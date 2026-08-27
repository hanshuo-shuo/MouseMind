"""Deterministically merge P2 episode shards and reject incomplete contracts."""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import fields
from pathlib import Path
from typing import Any

from mouse_llm.evaluation.closed_loop import (
    EpisodeResult,
    _write_outputs,
    build_report,
)
from mouse_llm.evaluation.contracts import DEFAULT_P2_CONTRACT, contract_seeds


MATCH_KEYS = (
    "environment",
    "world",
    "contract_name",
    "seed_pool",
    "max_steps",
    "control_budget_seconds",
    "ood_condition",
    "environment_parameters",
    "preference",
    "instruction_split",
    "planner_horizon",
    "p1_planner_horizon",
    "evade_distance",
    "risk_threshold",
)


INT_FIELDS = {
    "seed",
    "success",
    "captured",
    "captures",
    "survived",
    "steps",
    "first_predator_visible_step",
    "first_capture_step",
    "predator_visible_steps",
    "capture_near_occlusion",
    "capture_in_open_space",
    "planner_calls",
    "clean_success",
    "clean_success_steps",
    "time_to_first_capture",
}
OPTIONAL_FIELDS = {
    "first_predator_visible_step",
    "first_capture_step",
    "clean_success_steps",
    "time_to_first_capture",
    "captures_per_successful_episode",
    "dominant_skill",
}


def _episode(row: dict[str, str]) -> EpisodeResult:
    payload: dict[str, Any] = {}
    available = {field.name for field in fields(EpisodeResult)}
    for name in available:
        raw = row.get(name, "")
        if raw == "" and name in OPTIONAL_FIELDS:
            payload[name] = None
        elif name == "latency_samples_seconds":
            payload[name] = (float(row["latency_mean_seconds"]),)
        elif name in INT_FIELDS:
            payload[name] = int(float(raw or 0))
        elif name in {"policy", "failure_mode", "dominant_skill", "skill_counts"}:
            payload[name] = raw
        else:
            payload[name] = float(raw or 0.0)
    return EpisodeResult(**payload)


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge complete P2 seed shards")
    parser.add_argument("--shards", type=Path, nargs="+", required=True)
    parser.add_argument("--contract", type=Path, default=DEFAULT_P2_CONTRACT)
    parser.add_argument("--seed-pool", choices=("development", "final_id_test"), required=True)
    parser.add_argument("--reference-policy", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    expected = set(contract_seeds(args.seed_pool, args.contract))
    baseline_metadata: dict[str, Any] | None = None
    by_policy: dict[str, dict[int, EpisodeResult]] = {}
    for shard in args.shards:
        report = json.loads((shard / "closed_loop_metrics.json").read_text(encoding="utf-8"))
        metadata = report["metadata"]
        if metadata.get("seed_pool") != args.seed_pool:
            raise ValueError(f"Wrong seed pool in {shard}")
        if baseline_metadata is None:
            baseline_metadata = metadata
        else:
            for key in MATCH_KEYS:
                if metadata.get(key) != baseline_metadata.get(key):
                    raise ValueError(f"Shard config mismatch for {key}: {shard}")
        with (shard / "closed_loop_episodes.csv").open(
            "r", encoding="utf-8", newline=""
        ) as handle:
            for row in csv.DictReader(handle):
                episode = _episode(row)
                policy_rows = by_policy.setdefault(episode.policy, {})
                if episode.seed in policy_rows:
                    raise ValueError(
                        f"Duplicate seed {episode.seed} for {episode.policy}"
                    )
                policy_rows[episode.seed] = episode
    if baseline_metadata is None:
        raise ValueError("No shards supplied")
    for policy, rows in by_policy.items():
        actual = set(rows)
        if actual != expected:
            missing = sorted(expected - actual)
            extra = sorted(actual - expected)
            raise ValueError(
                f"Incomplete {policy}: missing={missing[:10]} extra={extra[:10]}"
            )
    results = {
        policy: [rows[seed] for seed in sorted(expected)]
        for policy, rows in by_policy.items()
    }
    metadata = dict(baseline_metadata)
    metadata.update(
        {
            "episode_count": len(expected),
            "full_pool_episode_count": len(expected),
            "merged_shards": [str(path) for path in args.shards],
            "seed_completeness_verified": True,
            "duplicate_seeds_verified_absent": True,
            "latency_merge_note": "action percentiles reconstructed from per-episode means; run main evidence unsharded for exact action percentiles",
            "research_evidence": True,
            "research_evidence_blockers": [],
        }
    )
    report = build_report(
        results,
        seed=min(expected),
        reference_policy=args.reference_policy,
        metadata=metadata,
    )
    _write_outputs(args.output_dir, report, results)
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
