"""Create a deterministic replay manifest for targeted corrective data collection."""

from __future__ import annotations

import argparse
import csv
import json
import os
from collections import defaultdict
from pathlib import Path
from typing import Any


def severity(row: dict[str, str]) -> tuple[float, float, int]:
    mode = row["failure_mode"]
    if mode.startswith("capture") or mode in {
        "late_predator_response",
        "open_space_capture",
    }:
        primary = float(row["captures"])
    elif mode == "navigation_oscillation":
        primary = float(row["oscillation_score"])
    elif mode == "stuck_timeout":
        primary = -float(row["recent_path_length"])
    elif mode in {"wrong_way_navigation", "goal_overshoot"}:
        primary = float(row["goal_distance_end"]) - float(
            row["goal_distance_min"]
        )
    else:
        primary = -float(row["episode_return"])
    return primary, -float(row["episode_return"]), -int(row["seed"])


def build_manifest(
    rows: list[dict[str, str]],
    *,
    policy: str,
    per_mode: int,
    source_metrics: dict[str, Any],
) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        if row["policy"] == policy and row["failure_mode"] != "success":
            grouped[row["failure_mode"]].append(row)
    selections = []
    for mode, candidates in sorted(grouped.items()):
        ranked = sorted(candidates, key=severity, reverse=True)
        for rank, row in enumerate(ranked[:per_mode], start=1):
            selections.append(
                {
                    "seed": int(row["seed"]),
                    "failure_mode": mode,
                    "rank_within_mode": rank,
                    "captures": int(float(row["captures"])),
                    "episode_return": float(row["episode_return"]),
                    "first_predator_visible_step": (
                        int(float(row["first_predator_visible_step"]))
                        if row.get("first_predator_visible_step")
                        else None
                    ),
                    "first_capture_step": (
                        int(float(row["first_capture_step"]))
                        if row.get("first_capture_step")
                        else None
                    ),
                    "corrective_target": (
                        "expert action labels from first visibility through capture"
                        if "capture" in mode or mode == "late_predator_response"
                        else "expert recovery labels over the final 20 steps"
                    ),
                }
            )
    return {
        "schema_version": 1,
        "artifact": "failure_replay_manifest",
        "research_evidence": bool(
            source_metrics.get("metadata", {}).get("research_evidence")
        ),
        "policy": policy,
        "source_seed_sha256": source_metrics["metadata"]["seed_sha256"],
        "per_mode_limit": per_mode,
        "selected_episode_count": len(selections),
        "failure_mode_counts": {
            mode: len(candidates) for mode, candidates in sorted(grouped.items())
        },
        "replay_queue": selections,
        "next_step": (
            "replay these seeds, collect expert corrective actions around the "
            "named failure window, retrain, then evaluate on the original full seed set"
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Mine closed-loop failure episodes")
    parser.add_argument("--episodes", type=Path, required=True)
    parser.add_argument("--metrics", type=Path, required=True)
    parser.add_argument("--policy", required=True)
    parser.add_argument("--per-mode", type=int, default=20)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.per_mode <= 0:
        raise ValueError("per-mode must be positive")
    with args.episodes.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    metrics = json.loads(args.metrics.read_text(encoding="utf-8"))
    manifest = build_manifest(
        rows,
        policy=args.policy,
        per_mode=args.per_mode,
        source_metrics=metrics,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(temporary, args.output)
    args.output.chmod(0o600)
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
