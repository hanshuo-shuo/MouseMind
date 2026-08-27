"""Validate P2 skill data and launch the existing generic MiniMind LoRA engine."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

from mouse_llm.data.planner_schema import parse_skill


def validate_training_data(path: Path) -> int:
    count = 0
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            payload = json.loads(line)
            conversations = payload.get("conversations")
            if not isinstance(conversations, list) or len(conversations) != 3:
                raise ValueError(f"Line {line_number}: expected three planner messages")
            if conversations[-1].get("role") != "assistant" or parse_skill(
                conversations[-1].get("content", "")
            ) is None:
                raise ValueError(f"Line {line_number}: invalid exact skill target")
            if payload.get("instruction_split") != "train":
                raise ValueError(f"Line {line_number}: non-training instruction leaked")
            count += 1
    if count == 0:
        raise ValueError("No MiniMind skill-planner training examples")
    return count


def sanitize_sft_data(source: Path, destination: Path) -> int:
    rows = []
    with source.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append({"conversations": json.loads(line)["conversations"]})
    destination.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True, separators=(",", ":")))
            handle.write("\n")
    os.replace(temporary, destination)
    destination.chmod(0o600)
    return len(rows)


def validate_supervision_coverage(
    path: Path,
    *,
    tokenizer_path: Path,
    max_seq_len: int,
) -> dict[str, int]:
    from transformers import AutoTokenizer

    from dataset.lm_dataset import SFTDataset

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    dataset = SFTDataset(str(path), tokenizer, max_length=max_seq_len)
    supervised_counts = []
    for index in range(len(dataset)):
        _, labels = dataset[index]
        count = int((labels != -100).sum())
        if count == 0:
            raise ValueError(
                f"Planner sample {index} has no supervised target within "
                f"max_seq_len={max_seq_len}; increase the limit or compress context"
            )
        supervised_counts.append(count)
    return {
        "sample_count": len(supervised_counts),
        "minimum_supervised_tokens": min(supervised_counts),
        "maximum_supervised_tokens": max(supervised_counts),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Train MiniMind skill-planner LoRA")
    parser.add_argument("--data-path", type=Path, required=True)
    parser.add_argument("--save-dir", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--lora-name", default="lora_mousemind_skill_planner")
    parser.add_argument("--from-weight", default="full_sft")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--max-seq-len", type=int, default=1024)
    parser.add_argument("--tokenizer", type=Path)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()
    count = validate_training_data(args.data_path)
    args.save_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
    sanitized_data = args.save_dir / "skill_planner_sft_private.jsonl"
    sanitized_count = sanitize_sft_data(args.data_path, sanitized_data)
    if sanitized_count != count:
        raise ValueError("Sanitized planner SFT row count changed")
    tokenizer_path = args.tokenizer or args.repo_root / "model"
    coverage = validate_supervision_coverage(
        sanitized_data,
        tokenizer_path=tokenizer_path,
        max_seq_len=args.max_seq_len,
    )
    print(f"validated_skill_planner_rows={count}", flush=True)
    print(f"supervision_coverage={json.dumps(coverage, sort_keys=True)}", flush=True)
    if args.prepare_only:
        return
    command = [
        sys.executable,
        "train_lora.py",
        "--data_path",
        str(sanitized_data.resolve()),
        "--save_dir",
        str(args.save_dir.resolve()),
        "--lora_name",
        args.lora_name,
        "--from_weight",
        args.from_weight,
        "--epochs",
        str(args.epochs),
        "--batch_size",
        str(args.batch_size),
        "--learning_rate",
        str(args.learning_rate),
        "--max_seq_len",
        str(args.max_seq_len),
        "--num_workers",
        str(args.num_workers),
        "--device",
        args.device,
        "--save_interval",
        "1000",
    ]
    subprocess.run(
        command,
        cwd=args.repo_root.resolve() / "trainer",
        check=True,
    )


if __name__ == "__main__":
    main()
