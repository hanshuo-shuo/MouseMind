"""Merge independently executed policies into one paired benchmark report."""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
from typing import Any

from mouse_llm.evaluation.closed_loop import (
    DEFAULT_OBSERVATION_AUDIT,
    SUMMARY_FIELDS,
    bootstrap_mean,
    load_verified_observation_audit,
)


METADATA_MATCH_KEYS = (
    "environment",
    "world",
    "episode_count",
    "seed_start",
    "seed_sha256",
    "max_steps",
    "control_budget_seconds",
)


def _load_run(run_dir: Path) -> tuple[dict[str, Any], list[dict[str, str]]]:
    report = json.loads(
        (run_dir / "closed_loop_metrics.json").read_text(encoding="utf-8")
    )
    with (run_dir / "closed_loop_episodes.csv").open(
        "r", encoding="utf-8", newline=""
    ) as handle:
        rows = list(csv.DictReader(handle))
    if report.get("experiment") != "mousemind_seeded_closed_loop" or not rows:
        raise ValueError(f"Not a closed-loop run: {run_dir}")
    return report, rows


def _paired_from_rows(
    reference: list[dict[str, str]],
    candidate: list[dict[str, str]],
    *,
    seed: int,
) -> dict[str, Any]:
    reference_by_seed = {int(row["seed"]): row for row in reference}
    candidate_by_seed = {int(row["seed"]): row for row in candidate}
    if reference_by_seed.keys() != candidate_by_seed.keys():
        raise ValueError("Policies do not contain identical paired seeds")
    seeds = sorted(reference_by_seed)
    result: dict[str, Any] = {
        "candidate_minus_reference": True,
        "episode_count": len(seeds),
    }
    for offset, (metric, field) in enumerate(SUMMARY_FIELDS.items()):
        differences = [
            float(candidate_by_seed[item].get(field, 0.0) or 0.0)
            - float(reference_by_seed[item].get(field, 0.0) or 0.0)
            for item in seeds
        ]
        result[metric] = bootstrap_mean(differences, seed=seed + offset)
    return result


def merge_runs(
    run_dirs: list[Path],
    *,
    reference_policy: str,
    seed: int,
    execution_note: str | None = None,
) -> tuple[dict[str, Any], list[dict[str, str]]]:
    loaded = [_load_run(path) for path in run_dirs]
    reference_metadata = loaded[0][0]["metadata"]
    policies: dict[str, Any] = {}
    rows_by_policy: dict[str, list[dict[str, str]]] = {}
    all_rows: list[dict[str, str]] = []
    systems: dict[str, Any] = {}
    hierarchical_config: dict[str, Any] | None = None
    for run_dir, (report, rows) in zip(run_dirs, loaded, strict=True):
        metadata = report["metadata"]
        for key in METADATA_MATCH_KEYS:
            if metadata.get(key) != reference_metadata.get(key):
                raise ValueError(f"Run metadata differs for {key}: {run_dir}")
        if metadata.get("research_evidence") is not True:
            raise ValueError(f"Run is not research evidence: {run_dir}")
        for policy_name, summary in report["policies"].items():
            if policy_name in policies:
                raise ValueError(f"Duplicate policy {policy_name!r}; merge full policy runs")
            policy_rows = [row for row in rows if row["policy"] == policy_name]
            if len(policy_rows) != metadata["episode_count"]:
                raise ValueError(f"Incomplete policy rows for {policy_name!r}")
            policies[policy_name] = summary
            rows_by_policy[policy_name] = policy_rows
            all_rows.extend(policy_rows)
        if metadata.get("hierarchical_policy"):
            hierarchical_config = metadata["hierarchical_policy"]
        systems[run_dir.name] = report.get("system", {})
    if reference_policy not in policies:
        raise ValueError(f"Missing reference policy {reference_policy!r}")
    paired = {}
    ordered_policies = [
        reference_policy,
        *(name for name in policies if name != reference_policy),
    ]
    comparison_index = 0
    for candidate_index in range(1, len(ordered_policies)):
        candidate = ordered_policies[candidate_index]
        for reference in ordered_policies[:candidate_index]:
            comparison_index += 1
            paired[f"{candidate}_minus_{reference}"] = _paired_from_rows(
                rows_by_policy[reference],
                rows_by_policy[candidate],
                seed=seed + comparison_index * 1000,
            )
    metadata = dict(reference_metadata)
    metadata["reference_policy"] = reference_policy
    metadata["merged_policy_runs"] = [path.name for path in run_dirs]
    metadata["observation_contract_audit"] = load_verified_observation_audit(
        DEFAULT_OBSERVATION_AUDIT
    )
    if hierarchical_config is not None:
        metadata["hierarchical_policy"] = hierarchical_config
    if execution_note:
        metadata["execution_note"] = execution_note
    report = {
        "schema_version": 1,
        "experiment": "mousemind_seeded_closed_loop",
        "metadata": metadata,
        "policies": policies,
        "paired_comparisons": paired,
        "failure_taxonomy": loaded[0][0]["failure_taxonomy"],
        "system_by_run": systems,
    }
    all_rows.sort(key=lambda row: (row["policy"], int(row["seed"])))
    return report, all_rows


def _write(output_dir: Path, report: dict[str, Any], rows: list[dict[str, str]]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
    metrics_path = output_dir / "closed_loop_metrics.json"
    metrics_tmp = metrics_path.with_suffix(".json.tmp")
    metrics_tmp.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(metrics_tmp, metrics_path)
    episodes_path = output_dir / "closed_loop_episodes.csv"
    episodes_tmp = episodes_path.with_suffix(".csv.tmp")
    fieldnames: list[str] = []
    for row in rows:
        for field in row:
            if field not in fieldnames:
                fieldnames.append(field)
    with episodes_tmp.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(episodes_tmp, episodes_path)
    metrics_path.chmod(0o600)
    episodes_path.chmod(0o600)


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge full paired policy runs")
    parser.add_argument("--runs", type=Path, nargs="+", required=True)
    parser.add_argument("--reference-policy", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bootstrap-seed", type=int, default=42)
    parser.add_argument("--execution-note")
    args = parser.parse_args()
    report, rows = merge_runs(
        args.runs,
        reference_policy=args.reference_policy,
        seed=args.bootstrap_seed,
        execution_note=args.execution_note,
    )
    _write(args.output_dir, report, rows)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
