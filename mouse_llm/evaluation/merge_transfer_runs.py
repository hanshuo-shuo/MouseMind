"""Merge complete policy-partitioned transfer runs and reject contract drift."""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import fields
from pathlib import Path
from typing import Any

from mouse_llm.evaluation.closed_loop import EpisodeResult, _write_outputs, build_report
from mouse_llm.evaluation.contracts import (
    DEFAULT_TRANSFER_CONTRACT,
    transfer_contract_seeds,
)
from mouse_llm.evaluation.transfer_benchmark import (
    _add_transfer_metrics,
    _paired_transfer_differences,
)


MATCH_KEYS = (
    "source_environment",
    "target_environment",
    "world",
    "contract_name",
    "contract_sha256",
    "compatibility_audit_sha256",
    "code_manifest_sha256",
    "code_manifest_files",
    "seed_pool",
    "seed_sha256",
    "condition",
    "instruction_split",
    "preference",
    "literal_low_level_transfer_compatible",
    "aligned_goal_controller",
    "target_adaptation_used",
    "goal_projection",
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
    "goals_completed",
    "goal_count",
    "return_completed",
    "objectives_completed",
    "objective_count",
}
OPTIONAL_FIELDS = {
    "first_predator_visible_step",
    "first_capture_step",
    "clean_success_steps",
    "time_to_first_capture",
    "captures_per_successful_episode",
    "dominant_skill",
}
TEXT_FIELDS = {"policy", "failure_mode", "dominant_skill", "skill_counts"}


def _episode(row: dict[str, str]) -> EpisodeResult:
    payload: dict[str, Any] = {}
    for field in fields(EpisodeResult):
        name = field.name
        raw = row.get(name, "")
        if raw == "" and name in OPTIONAL_FIELDS:
            payload[name] = None
        elif name == "latency_samples_seconds":
            payload[name] = (float(row["latency_mean_seconds"]),)
        elif name in INT_FIELDS:
            payload[name] = int(float(raw or 0))
        elif name in TEXT_FIELDS:
            payload[name] = raw
        else:
            payload[name] = float(raw or 0.0)
    return EpisodeResult(**payload)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=Path, nargs="+", required=True)
    parser.add_argument("--contract", type=Path, default=DEFAULT_TRANSFER_CONTRACT)
    parser.add_argument(
        "--seed-pool", choices=("development", "final_test"), required=True
    )
    parser.add_argument("--condition", required=True)
    parser.add_argument("--reference-policy", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    expected_seeds = set(transfer_contract_seeds(args.seed_pool, args.contract))
    baseline_metadata: dict[str, Any] | None = None
    compatibility: dict[str, Any] | None = None
    by_policy: dict[str, dict[int, EpisodeResult]] = {}
    exact_latency: dict[str, dict[str, Any]] = {}
    checkpoint_sha256: dict[str, str] = {}
    for run in args.runs:
        report = json.loads(
            (run / "closed_loop_metrics.json").read_text(encoding="utf-8")
        )
        metadata = report["metadata"]
        if (
            metadata.get("research_evidence") is not True
            or metadata.get("seed_pool") != args.seed_pool
            or metadata.get("condition") != args.condition
        ):
            raise ValueError(f"Run is not complete matching evidence: {run}")
        if baseline_metadata is None:
            baseline_metadata = metadata
            compatibility = report["compatibility"]
        else:
            for key in MATCH_KEYS:
                if metadata.get(key) != baseline_metadata.get(key):
                    raise ValueError(f"Transfer run mismatch for {key}: {run}")
            if report["compatibility"] != compatibility:
                raise ValueError(f"Compatibility payload mismatch: {run}")
        for policy, summary in report["policies"].items():
            if policy in exact_latency:
                raise ValueError(f"Policy appears in more than one run: {policy}")
            exact_latency[policy] = summary["latency_seconds"]
        for role, digest in metadata.get("checkpoint_sha256", {}).items():
            previous = checkpoint_sha256.setdefault(role, digest)
            if previous != digest:
                raise ValueError(f"Checkpoint hash mismatch for {role}: {run}")
        with (run / "closed_loop_episodes.csv").open(
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
    if baseline_metadata is None or compatibility is None:
        raise ValueError("No transfer runs supplied")
    if args.reference_policy not in by_policy:
        raise ValueError("Merged reference policy is absent")
    for policy, rows in by_policy.items():
        actual = set(rows)
        if actual != expected_seeds:
            raise ValueError(
                f"Incomplete {policy}: missing={sorted(expected_seeds - actual)[:10]} "
                f"extra={sorted(actual - expected_seeds)[:10]}"
            )
    results = {
        policy: [rows[seed] for seed in sorted(expected_seeds)]
        for policy, rows in by_policy.items()
    }
    metadata = dict(baseline_metadata)
    metadata.update(
        {
            "episode_count": len(expected_seeds),
            "full_pool_episode_count": len(expected_seeds),
            "merged_policy_runs": [str(path) for path in args.runs],
            "policy_partition_completeness_verified": True,
            "duplicate_policy_runs_verified_absent": True,
            "research_evidence": True,
            "research_evidence_blockers": [],
            "checkpoint_sha256": checkpoint_sha256,
        }
    )
    report = build_report(
        results,
        seed=min(expected_seeds),
        reference_policy=args.reference_policy,
        metadata=metadata,
    )
    report["experiment"] = "mousemind_frozen_cross_task_transfer"
    for policy, latency in exact_latency.items():
        report["policies"][policy]["latency_seconds"] = latency
    _add_transfer_metrics(
        report, results, bootstrap_seed=min(expected_seeds) + 5000
    )
    within_mode: dict[str, Any] = {}
    for reference, candidates in (
        (
            "literal-direct-mlp",
            ("literal-p1-rule", "literal-numeric", "literal-minimind"),
        ),
        (
            "aligned-goal-only",
            (
                "aligned-p1-rule",
                "aligned-numeric",
                "aligned-minimind",
                "aligned-minimind-no-history",
                "aligned-minimind-no-instruction",
            ),
        ),
    ):
        if reference not in results:
            continue
        for offset, candidate in enumerate(candidates):
            if candidate in results:
                within_mode[f"{candidate}_minus_{reference}"] = (
                    _paired_transfer_differences(
                        results[reference],
                        results[candidate],
                        seed=min(expected_seeds) + 10000 + offset * 100,
                    )
                )
    report["within_mode_paired_comparisons"] = within_mode
    report["compatibility"] = compatibility
    _write_outputs(args.output_dir, report, results)
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
