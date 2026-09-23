"""Seed-clean full-parameter SFT of the hierarchical MiniMind skill planner."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from mouse_llm.training.train_skill_planner import (sanitize_sft_data, validate_supervision_coverage, validate_training_data)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-path', type=Path, required=True)
    parser.add_argument('--base-weight', type=Path, required=True)
    parser.add_argument('--tokenizer', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--batch-size', type=int, default=2)
    parser.add_argument('--gradient-accumulation', type=int, default=8)
    parser.add_argument('--learning-rate', type=float, default=1e-5)
    parser.add_argument('--max-seq-len', type=int, default=1024)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    if min(args.epochs, args.batch_size, args.gradient_accumulation) < 1 or args.learning_rate <= 0:
        parser.error('Invalid optimization parameter')
    from dataset.lm_dataset import SFTDataset
    from model.model_minimind import MiniMindConfig, MiniMindForCausalLM
    from transformers import AutoTokenizer

    count = validate_training_data(args.data_path)
    args.output_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
    private_data = args.output_dir / 'skill_planner_sft_private.jsonl'
    if sanitize_sft_data(args.data_path, private_data) != count:
        raise ValueError('Sanitized planner row count changed')
    coverage = validate_supervision_coverage(private_data, tokenizer_path=args.tokenizer, max_seq_len=args.max_seq_len)
    torch.manual_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    dataset = SFTDataset(str(private_data), tokenizer, max_length=args.max_seq_len)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    model = MiniMindForCausalLM(MiniMindConfig(hidden_size=768, num_hidden_layers=8))
    model.load_state_dict(torch.load(args.base_weight, map_location='cpu', weights_only=True), strict=True)
    model.to(device).train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    optimizer.zero_grad(set_to_none=True)
    losses = []
    steps = 0
    for epoch in range(args.epochs):
        for index, (ids, labels) in enumerate(loader):
            ids, labels = ids.to(device), labels.to(device)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == 'cuda'):
                result = model(ids, labels=labels)
                loss = result.loss + result.aux_loss
            (loss / args.gradient_accumulation).backward()
            if (index + 1) % args.gradient_accumulation == 0 or index + 1 == len(loader):
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                steps += 1
            losses.append(float(loss.detach()))
        print(f'epoch={epoch + 1} loss={sum(losses) / len(losses):.4f} steps={steps}', flush=True)
    weight = args.output_dir / 'full_skill_planner_768.pth'
    torch.save({key: value.detach().cpu() for key, value in model.state_dict().items()}, weight)
    os.chmod(weight, 0o600)
    metadata = {'method': 'full_sft', 'train_rows': count, 'coverage': coverage, 'epochs': args.epochs, 'optimizer_steps': steps, 'learning_rate': args.learning_rate, 'mean_loss': sum(losses) / len(losses), 'initial_weight': str(args.base_weight), 'output_weight': str(weight)}
    (args.output_dir / 'training.json').write_text(json.dumps(metadata, indent=2, sort_keys=True) + '\n')
    print(json.dumps(metadata, sort_keys=True), flush=True)


if __name__ == '__main__':
    main()
