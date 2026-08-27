"""Freeze planner horizon and verifier threshold using development results only."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _named(value: str) -> tuple[float, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("expected VALUE=PATH")
    name, path = value.split("=", 1)
    return float(name), Path(path)


def _load(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    metadata = payload.get("metadata", {})
    if metadata.get("seed_pool") != "development":
        raise ValueError(f"Selection input is not a development report: {path}")
    if metadata.get("research_evidence") is not True:
        raise ValueError(f"Selection input is not completed evidence: {path}")
    return payload


def _metrics(report: dict[str, Any], policy: str) -> dict[str, float]:
    summary = report["policies"][policy]
    return {
        "clean_success_rate": float(summary["clean_success_rate"]["mean"]),
        "task_success_rate": float(summary["success_rate"]["mean"]),
        "capture_rate": float(summary["capture_rate"]["mean"]),
        "captures_per_episode": float(summary["captures"]["mean"]),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Freeze P2 development choices")
    parser.add_argument("--horizon-report", action="append", type=_named, required=True)
    parser.add_argument("--threshold-report", action="append", type=_named, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    horizon_rows = []
    for horizon, path in args.horizon_report:
        report = _load(path)
        horizon_rows.append(
            {"horizon": int(horizon), **_metrics(report, "numeric-learned")}
        )
    selected_horizon = max(
        horizon_rows,
        key=lambda row: (
            row["clean_success_rate"],
            row["task_success_rate"],
            -row["capture_rate"],
            -row["horizon"],
        ),
    )
    threshold_rows = []
    for threshold, path in args.threshold_report:
        report = _load(path)
        verified = _metrics(report, "numeric-verified")
        unverified = _metrics(report, "numeric-learned")
        threshold_rows.append(
            {
                "threshold": threshold,
                **verified,
                "clean_success_delta_vs_unverified": verified["clean_success_rate"]
                - unverified["clean_success_rate"],
                "capture_rate_delta_vs_unverified": verified["capture_rate"]
                - unverified["capture_rate"],
                "task_success_delta_vs_unverified": verified["task_success_rate"]
                - unverified["task_success_rate"],
            }
        )
    # Primary objective is clean success. Ties prefer task completion, lower
    # capture rate, and the less aggressive (lower) threshold.
    selected_threshold = max(
        threshold_rows,
        key=lambda row: (
            row["clean_success_rate"],
            row["task_success_rate"],
            -row["capture_rate"],
            -row["threshold"],
        ),
    )
    verifier_promoted = bool(
        selected_threshold["clean_success_delta_vs_unverified"] > 0.0
        and selected_threshold["capture_rate_delta_vs_unverified"] < 0.0
        and selected_threshold["task_success_delta_vs_unverified"] >= -0.05
    )
    output = {
        "schema_version": 1,
        "artifact": "p2_development_operating_point",
        "research_evidence": True,
        "selection_pool": "development",
        "final_test_used": False,
        "selection_rule": "maximize clean success; tie-break by task success, lower capture rate, then simpler/lower operating value",
        "horizon_sweep": sorted(horizon_rows, key=lambda row: row["horizon"]),
        "threshold_sweep": sorted(threshold_rows, key=lambda row: row["threshold"]),
        "selected_planner_horizon": selected_horizon["horizon"],
        "selected_risk_threshold": selected_threshold["threshold"],
        "selected_system": (
            "numeric-verified" if verifier_promoted else "numeric-learned"
        ),
        "verifier_promoted": verifier_promoted,
        "verifier_deployment_status": (
            "promoted"
            if verifier_promoted
            else "not promoted: no swept operating point improved clean success and capture rate without material task-success loss"
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(output, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(output, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
