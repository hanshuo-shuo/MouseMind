"""Pair seed-matched SFT, DPO, and GRPO closed-loop runs without test leakage."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

from mouse_llm.evaluation.closed_loop import bootstrap_mean
from mouse_llm.evaluation.contracts import DEFAULT_P2_CONTRACT, contract_seeds


FIELDS = {
    "task_success_rate": "success",
    "clean_success_rate": "clean_success",
    "capture_rate": "captured",
    "captures_per_episode": "captures",
}
MATCHING_METADATA = (
    "contract_sha256", "seed_pool", "seed_sha256", "episode_count",
    "planner_horizon", "p1_planner_horizon", "risk_threshold",
    "ood_condition", "preference", "instruction_split", "environment_parameters",
)


def load_run(directory: Path, *, seed_pool: str, contract: Path) -> tuple[dict[str, Any], dict[int, dict[str, str]]]:
    report = json.loads((directory / "closed_loop_metrics.json").read_text())
    metadata = report["metadata"]
    if metadata["seed_pool"] != seed_pool:
        raise ValueError(f"{directory}: expected {seed_pool}, got {metadata['seed_pool']}")
    with (directory / "closed_loop_episodes.csv").open(newline="") as handle:
        rows = [row for row in csv.DictReader(handle) if row["policy"] == "minimind-learned"]
    by_seed = {int(row["seed"]): row for row in rows}
    expected = set(contract_seeds(seed_pool, contract))
    if len(rows) != len(by_seed) or set(by_seed) != expected:
        raise ValueError(f"{directory}: expected every seed in {seed_pool} exactly once")
    if metadata["episode_count"] != len(expected):
        raise ValueError(f"{directory}: benchmark metadata has incomplete episode count")
    return metadata, by_seed


def compare_runs(
    baseline: Path,
    candidates: dict[str, Path],
    *,
    seed_pool: str = "development",
    contract: Path = DEFAULT_P2_CONTRACT,
    bootstrap_seed: int = 42,
) -> dict[str, Any]:
    baseline_meta, baseline_rows = load_run(baseline, seed_pool=seed_pool, contract=contract)
    result: dict[str, Any] = {
        "seed_pool": seed_pool,
        "episode_count": len(baseline_rows),
        "baseline": str(baseline),
        "candidates": {},
    }
    for name, directory in candidates.items():
        metadata, rows = load_run(directory, seed_pool=seed_pool, contract=contract)
        for key in MATCHING_METADATA:
            if metadata[key] != baseline_meta[key]:
                raise ValueError(f"{name}: benchmark differs at {key}")
        estimates: dict[str, Any] = {}
        for offset, (metric, field) in enumerate(FIELDS.items()):
            base_values = [float(baseline_rows[seed][field]) for seed in sorted(rows)]
            candidate_values = [float(rows[seed][field]) for seed in sorted(rows)]
            estimates[metric] = {
                "baseline_mean": sum(base_values) / len(base_values),
                "candidate_mean": sum(candidate_values) / len(candidate_values),
                "candidate_minus_baseline": bootstrap_mean(
                    [candidate - base for base, candidate in zip(base_values, candidate_values)],
                    seed=bootstrap_seed + offset,
                ),
            }
        result["candidates"][name] = {
            "directory": str(directory),
            "metrics": estimates,
            "development_candidate": bool(
                estimates["task_success_rate"]["candidate_mean"] >= estimates["task_success_rate"]["baseline_mean"]
                and estimates["clean_success_rate"]["candidate_mean"] >= estimates["clean_success_rate"]["baseline_mean"]
                and estimates["capture_rate"]["candidate_mean"] <= estimates["capture_rate"]["baseline_mean"]
                and estimates["captures_per_episode"]["candidate_mean"] < estimates["captures_per_episode"]["baseline_mean"]
            ) if seed_pool == "development" else None,
        }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", action="append", required=True, metavar="NAME=DIR")
    parser.add_argument("--seed-pool", choices=("development", "final_id_test"), default="development")
    parser.add_argument("--contract", type=Path, default=DEFAULT_P2_CONTRACT)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    candidates: dict[str, Path] = {}
    for item in args.candidate:
        name, separator, directory = item.partition("=")
        if not separator or not name or not directory or name in candidates:
            parser.error(f"Invalid or duplicate --candidate {item!r}")
        candidates[name] = Path(directory)
    result = compare_runs(
        args.baseline, candidates, seed_pool=args.seed_pool, contract=args.contract
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({name: row["development_candidate"] for name, row in result["candidates"].items()}, sort_keys=True))


if __name__ == "__main__":
    main()
