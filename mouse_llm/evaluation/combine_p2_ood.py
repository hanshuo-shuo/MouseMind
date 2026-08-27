"""Combine completed OOD reports into one compact Git-safe aggregate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


POLICY_FIELDS = (
    "episode_count",
    "success_rate",
    "clean_success_rate",
    "capture_rate",
    "captures",
    "survival_rate",
    "return",
    "path_efficiency",
    "steps",
    "planner_override_rate",
    "skill_switch_rate",
    "latency_seconds",
)


def _named_path(raw: str) -> tuple[str, Path]:
    if "=" not in raw:
        raise argparse.ArgumentTypeError("expected NAME=PATH")
    name, path = raw.split("=", 1)
    return name, Path(path)


def _load(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("experiment") != "mousemind_seeded_closed_loop":
        raise ValueError(f"Not a MouseMind closed-loop report: {path}")
    metadata = payload.get("metadata", {})
    if metadata.get("research_evidence") is not True:
        raise ValueError(f"Not completed research evidence: {path}")
    if metadata.get("seed_pool") != "final_id_test":
        raise ValueError(f"Not a final-test OOD report: {path}")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Combine final P2 OOD reports")
    parser.add_argument("--report", action="append", type=_named_path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    reports = [(name, _load(path)) for name, path in args.report]
    reference = reports[0][1]["metadata"]
    conditions: dict[str, Any] = {}
    for name, report in reports:
        metadata = report["metadata"]
        for field in (
            "contract_name",
            "contract_sha256",
            "seed_sha256",
            "episode_count",
            "world",
            "planner_horizon",
            "p1_planner_horizon",
            "risk_threshold",
        ):
            if metadata.get(field) != reference.get(field):
                raise ValueError(f"OOD report mismatch for {field}: {name}")
        conditions[name] = {
            "ood_condition": metadata["ood_condition"],
            "environment_parameters": metadata["environment_parameters"],
            "instruction_split": metadata["instruction_split"],
            "policies": {
                policy: {
                    field: summary[field]
                    for field in POLICY_FIELDS
                    if field in summary
                }
                for policy, summary in report["policies"].items()
            },
        }
    output = {
        "schema_version": 1,
        "artifact": "p2_ood_closed_loop",
        "research_evidence": True,
        "contract_name": reference["contract_name"],
        "contract_sha256": reference["contract_sha256"],
        "seed_pool": reference["seed_pool"],
        "seed_sha256": reference["seed_sha256"],
        "episode_count_per_condition": reference["episode_count"],
        "selected_system": "numeric-learned",
        "verifier_promoted": False,
        "conditions": conditions,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(output, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(output, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
