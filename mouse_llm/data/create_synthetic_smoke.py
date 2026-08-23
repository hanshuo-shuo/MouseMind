"""Create a clearly labeled, non-research dataset for pipeline smoke tests."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from .schema import make_conversation


def observation(index: int) -> list[float]:
    return [
        round((((index + 1) * (feature + 3) * 17) % 200 - 100) / 100, 4)
        for feature in range(10)
    ]


def write_jsonl(path: Path, *, count: int, offset: int, action: int) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for index in range(offset, offset + count):
            sample = make_conversation(
                observation(index), action, action_count=295, precision=4
            )
            handle.write(json.dumps(sample, separators=(",", ":")) + "\n")
    os.replace(temporary, path)
    path.chmod(0o600)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--train-samples", type=int, default=64)
    parser.add_argument("--validation-samples", type=int, default=16)
    parser.add_argument("--test-samples", type=int, default=16)
    parser.add_argument("--constant-action", type=int, default=24)
    args = parser.parse_args()

    if not 0 <= args.constant_action < 295:
        raise ValueError("constant-action must be in [0, 294]")
    if min(args.train_samples, args.validation_samples, args.test_samples) <= 0:
        raise ValueError("all synthetic split sizes must be positive")

    args.output_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
    args.output_dir.chmod(0o700)
    offsets = {"train": 0, "validation": 10_000, "test": 20_000}
    counts = {
        "train": args.train_samples,
        "validation": args.validation_samples,
        "test": args.test_samples,
    }
    for split, count in counts.items():
        write_jsonl(
            args.output_dir / f"{split}.jsonl",
            count=count,
            offset=offsets[split],
            action=args.constant_action,
        )
    manifest = {
        "schema_version": 1,
        "synthetic": True,
        "research_evidence": False,
        "purpose": "engineering smoke test only",
        "constant_action": args.constant_action,
        "sample_counts": counts,
    }
    manifest_path = args.output_dir / "manifest.json"
    temporary = manifest_path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, manifest_path)
    manifest_path.chmod(0o600)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
