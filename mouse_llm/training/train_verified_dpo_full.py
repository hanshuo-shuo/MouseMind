"""Full-parameter DPO adaptation using the same verified skill pairs."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from dataset.lm_dataset import SFTDataset
from model.model_minimind import MiniMindConfig, MiniMindForCausalLM


class PairDataset(Dataset):
    def __init__(self, path: Path, tokenizer, max_seq_len: int):
        self.rows = [json.loads(line) for line in path.read_text().splitlines() if line]
        if not self.rows:
            raise ValueError("Empty DPO train split")
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.max_length = max_seq_len
        self.bos_id = tokenizer(f"{tokenizer.bos_token}assistant\n", add_special_tokens=False).input_ids
        self.eos_id = tokenizer(f"{tokenizer.eos_token}\n", add_special_tokens=False).input_ids
        self.padding = tokenizer.pad_token_id or 0

    generate_labels = SFTDataset.generate_labels

    def __len__(self):
        return len(self.rows)

    def encode(self, messages):
        text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        ids = self.tokenizer(text, truncation=True, max_length=self.max_seq_len, padding="max_length").input_ids
        labels = self.generate_labels(ids)
        mask = torch.tensor([value != -100 for value in labels[1:]], dtype=torch.bool)
        if not mask.any():
            raise ValueError("DPO response was truncated; increase max_seq_len")
        return torch.tensor(ids[:-1]), torch.tensor(ids[1:]), mask

    def __getitem__(self, index):
        row = self.rows[index]
        prompt = row["prompt"]
        chosen = self.encode([*prompt, {"role": "assistant", "content": row["chosen"]}])
        rejected = self.encode([*prompt, {"role": "assistant", "content": row["rejected"]}])
        return {"chosen": chosen, "rejected": rejected}


def completion_logps(model, ids, labels, mask):
    logits = model(ids).logits.float()
    token_logps = F.log_softmax(logits, dim=-1).gather(-1, labels.unsqueeze(-1)).squeeze(-1)
    return (token_logps * mask).sum(-1)


def dpo_loss(policy_chosen, policy_rejected, ref_chosen, ref_rejected, beta):
    # Same standard log-ratio objective used in MiniMind upstream train_dpo.py.
    logits = (policy_chosen - policy_rejected) - (ref_chosen - ref_rejected)
    return -F.logsigmoid(beta * logits).mean()


def make_model(config, sft_full_weight: Path, device, *, trainable: bool):
    model = MiniMindForCausalLM(config)
    model.load_state_dict(torch.load(sft_full_weight, map_location="cpu", weights_only=True), strict=True)
    model.to(device)
    for parameter in model.parameters():
        parameter.requires_grad_(trainable)
    return model


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-data", type=Path, required=True)
    parser.add_argument("--sft-full-weight", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--beta", type=float, default=0.1)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--gradient-accumulation", type=int, default=8)
    parser.add_argument("--max-seq-len", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.beta <= 0 or args.epochs < 1 or args.batch_size < 1:
        parser.error("beta, epochs and batch size must be positive")
    from transformers import AutoTokenizer
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    dataset = PairDataset(args.train_data, tokenizer, args.max_seq_len)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    config = MiniMindConfig(hidden_size=768, num_hidden_layers=8)
    policy = make_model(config, args.sft_full_weight, device, trainable=True)
    reference = make_model(config, args.sft_full_weight, device, trainable=False)
    reference.eval()
    policy.train()
    parameters = [p for p in policy.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(parameters, lr=args.learning_rate)
    use_amp = device.type == "cuda"
    steps = 0
    losses = []
    optimizer.zero_grad(set_to_none=True)
    for epoch in range(args.epochs):
        for batch_index, batch in enumerate(loader):
            c_ids, c_labels, c_mask = (x.to(device) for x in batch["chosen"])
            r_ids, r_labels, r_mask = (x.to(device) for x in batch["rejected"])
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_amp):
                with torch.no_grad():
                    ref_c = completion_logps(reference, c_ids, c_labels, c_mask)
                    ref_r = completion_logps(reference, r_ids, r_labels, r_mask)
                pi_c = completion_logps(policy, c_ids, c_labels, c_mask)
                pi_r = completion_logps(policy, r_ids, r_labels, r_mask)
                loss = dpo_loss(pi_c, pi_r, ref_c, ref_r, args.beta)
            (loss / args.gradient_accumulation).backward()
            if (batch_index + 1) % args.gradient_accumulation == 0 or batch_index + 1 == len(loader):
                torch.nn.utils.clip_grad_norm_(parameters, 1.0)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                steps += 1
            losses.append(float(loss.detach().cpu()))
            if (batch_index + 1) % 50 == 0:
                print(f"epoch={epoch + 1} batch={batch_index + 1}/{len(loader)} loss={losses[-1]:.4f}", flush=True)
    args.output_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
    weight = args.output_dir / "full_verified_dpo_768.pth"
    torch.save({key: value.detach().cpu() for key, value in policy.state_dict().items()}, weight)
    os.chmod(weight, 0o600)
    metadata = {"beta": args.beta, "epochs": args.epochs, "batch_size": args.batch_size, "gradient_accumulation": args.gradient_accumulation, "learning_rate": args.learning_rate, "train_pairs": len(dataset), "optimizer_steps": steps, "last_loss": losses[-1], "mean_loss": sum(losses) / len(losses), "initial_policy_and_reference": str(args.sft_full_weight), "method": "full_dpo"}
    (args.output_dir / "training.json").write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
    print(json.dumps(metadata, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
