"""Upweight one corrective-data iteration without touching held-out splits."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def rows(path: Path) -> list[str]:
    return [line for line in path.read_text(encoding="utf-8").splitlines() if line]


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge frozen P2 and P2.1 training rows")
    parser.add_argument("--base-dir", type=Path, required=True)
    parser.add_argument("--corrective-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--corrective-weight", type=int, default=3)
    args = parser.parse_args()
    if args.corrective_weight <= 0:
        raise ValueError("corrective-weight must be positive")
    args.output_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
    counts = {}
    for stem in ("planner", "risk"):
        base_train = rows(args.base_dir / f"{stem}_train.jsonl")
        corrective_train = rows(args.corrective_dir / f"{stem}_train.jsonl")
        merged = [*base_train, *(corrective_train * args.corrective_weight)]
        (args.output_dir / f"{stem}_train.jsonl").write_text(
            "\n".join(merged) + "\n", encoding="utf-8"
        )
        counts[f"{stem}_base_train"] = len(base_train)
        counts[f"{stem}_corrective_unique"] = len(corrective_train)
        counts[f"{stem}_merged_train"] = len(merged)
    for name in (
        "planner_validation.jsonl",
        "planner_seen_test.jsonl",
        "planner_unseen_test.jsonl",
        "risk_validation.jsonl",
    ):
        (args.output_dir / name).write_text(
            (args.base_dir / name).read_text(encoding="utf-8"), encoding="utf-8"
        )
    manifest = {
        "schema_version": 1,
        "artifact": "p2_1_corrective_dataset_merge",
        "corrective_weight": args.corrective_weight,
        "held_out_splits_source": "unchanged P2.0 base dataset",
        "final_test_data_used": False,
        **counts,
    }
    (args.output_dir / "corrective_merge_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
