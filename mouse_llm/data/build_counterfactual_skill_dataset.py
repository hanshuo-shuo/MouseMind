"""Build outcome-grounded planner and risk datasets from verified replay anchors."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

from mouse_llm.data.planner_schema import (
    Preference,
    instruction_for,
    planner_messages,
    preference_one_hot,
    select_skill,
)
from mouse_llm.evaluation.closed_loop import MLPCheckpointPolicy
from mouse_llm.evaluation.contracts import DEFAULT_P2_CONTRACT, contract_seeds
from mouse_llm.evaluation.evaluate_policy import load_action_catalog
from mouse_llm.hierarchical.counterfactual import branch_anchor
from mouse_llm.hierarchical.policy import Skill


DATASET_SCHEMA_VERSION = "mousemind_counterfactual_skill_dataset_v1"


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            payload = json.loads(line)
            if "prefix_actions" not in payload:
                raise ValueError(f"Line {line_number}: not a replay anchor")
            rows.append(payload)
    if not rows:
        raise ValueError("No replay anchors found")
    return rows


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True, separators=(",", ":")))
            handle.write("\n")
    os.replace(temporary, path)


def _partition(anchor_id: str) -> str:
    bucket = int(hashlib.sha256(anchor_id.encode("ascii")).hexdigest()[:8], 16) % 100
    if bucket < 80:
        return "train"
    if bucket < 90:
        return "validation"
    return "test"


def build_rows(
    counterfactuals: list[dict[str, Any]], *, label_horizon: int
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, list[dict[str, Any]]]]:
    planner_rows: dict[str, list[dict[str, Any]]] = {
        "train": [],
        "validation": [],
        "seen_test": [],
        "unseen_test": [],
    }
    risk_rows: dict[str, list[dict[str, Any]]] = {"train": [], "validation": []}
    for record in counterfactuals:
        anchor_id = record["anchor_id"]
        partition = _partition(anchor_id)
        branches = record["branches_by_horizon"][str(label_horizon)]
        for preference in Preference:
            target, utilities = select_skill(branches, preference)
            destinations = (
                (("train", "train"),)
                if partition == "train"
                else (("validation", "validation"),)
                if partition == "validation"
                else (("seen_test", "train"), ("unseen_test", "unseen_test"))
            )
            for output_split, instruction_split in destinations:
                instruction = instruction_for(
                    preference,
                    split=instruction_split,
                    stable_key=anchor_id,
                )
                feature_vector = [
                    *record["context_vector"],
                    *preference_one_hot(preference),
                ]
                planner_rows[output_split].append(
                    {
                        "schema_version": DATASET_SCHEMA_VERSION,
                        "anchor_id": anchor_id,
                        "horizon": label_horizon,
                        "preference": preference.value,
                        "instruction": instruction,
                        "instruction_split": instruction_split,
                        "context_vector": feature_vector,
                        "target_skill": target.value,
                        "utilities": utilities,
                        "conversations": planner_messages(
                            record["context"],
                            preference=preference,
                            instruction=instruction,
                            target_skill=target,
                        ),
                    }
                )
        risk_split = "train" if partition == "train" else "validation"
        for skill in Skill:
            outcome = branches[skill.value]
            risk_rows[risk_split].append(
                {
                    "schema_version": DATASET_SCHEMA_VERSION,
                    "anchor_id": anchor_id,
                    "horizon": label_horizon,
                    "candidate_skill": skill.value,
                    "context_vector": record["context_vector"],
                    "capture_within_h": int(outcome["capture_within_h"]),
                    "outcome": outcome,
                }
            )
    return planner_rows, risk_rows


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Counterfactually label every candidate high-level skill"
    )
    parser.add_argument("--anchors", type=Path, required=True)
    parser.add_argument("--contract", type=Path, default=DEFAULT_P2_CONTRACT)
    parser.add_argument(
        "--source-pool",
        choices=("data_collection", "development_corrective"),
        default="data_collection",
    )
    parser.add_argument("--research-evidence", action="store_true")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--aggregate-output", type=Path, required=True)
    parser.add_argument("--mlp-checkpoint", type=Path, required=True)
    parser.add_argument("--horizons", type=int, nargs="+", default=(4, 8))
    parser.add_argument("--label-horizon", type=int, default=8)
    parser.add_argument("--evade-distance", type=float, default=0.35)
    parser.add_argument("--replay-tolerance", type=float, default=1e-7)
    parser.add_argument("--world", default="21_05")
    parser.add_argument("--max-steps", type=int, default=300)
    parser.add_argument("--time-step", type=float, default=0.25)
    parser.add_argument("--predator-speed-ratio", type=float, default=0.15)
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--action-catalog",
        type=Path,
        default=Path(__file__).resolve().parents[1]
        / "envs/mice/assets/action_catalog_21_05.json",
    )
    args = parser.parse_args()
    horizons = tuple(sorted(set(args.horizons)))
    if args.label_horizon not in horizons:
        raise ValueError("label-horizon must be included in horizons")
    anchors = _read_jsonl(args.anchors)
    input_anchor_seeds = {int(anchor["seed"]) for anchor in anchors}
    if args.source_pool == "development_corrective":
        allowed_development = set(contract_seeds("development", args.contract))
        if not input_anchor_seeds <= allowed_development:
            raise ValueError(
                "Development-corrective branches may not contain collection or final seeds"
            )
    private_output = args.output_dir / "counterfactuals_private.jsonl"
    destinations = load_action_catalog(args.action_catalog)
    specialist = MLPCheckpointPolicy(
        args.mlp_checkpoint,
        device=args.device,
        observation_indices=tuple(range(10)),
        name="counterfactual-specialist",
    )
    from mouse_llm.envs.mice import BotEvadeEnv, custom_reward

    def env_factory():
        return BotEvadeEnv(
            world_name=args.world,
            use_lppos=False,
            use_predator=True,
            max_step=args.max_steps,
            reward_function=custom_reward,
            time_step=args.time_step,
            frame_stack_k=1,
            predator_prey_forward_speed_ratio=args.predator_speed_ratio,
        )

    counterfactuals: list[dict[str, Any]] = (
        [json.loads(line) for line in private_output.read_text(encoding="utf-8").splitlines() if line]
        if private_output.exists()
        else []
    )
    processed = {
        (int(record["seed"]), int(record["step_index"]))
        for record in counterfactuals
    }
    for record in counterfactuals:
        if set(record["branches_by_horizon"]) != {str(value) for value in horizons}:
            raise ValueError("Existing counterfactual checkpoint uses different horizons")
    if counterfactuals:
        print(f"resuming {len(counterfactuals)} completed anchors", flush=True)
    for index, anchor in enumerate(anchors, start=1):
        if (int(anchor["seed"]), int(anchor["step_index"])) in processed:
            continue
        by_horizon: dict[str, Any] = {}
        base_record: dict[str, Any] | None = None
        for horizon in horizons:
            branched = branch_anchor(
                env_factory,
                anchor=anchor,
                specialist=specialist,
                destinations=destinations,
                horizon=horizon,
                evade_distance=args.evade_distance,
                replay_tolerance=args.replay_tolerance,
            )
            base_record = branched
            by_horizon[str(horizon)] = branched["branches"]
        assert base_record is not None
        counterfactuals.append(
            {
                key: value
                for key, value in base_record.items()
                if key not in {"branches", "horizon"}
            }
            | {"branches_by_horizon": by_horizon}
        )
        # Private incremental checkpoint: an interrupted long cluster job can
        # resume without turning already verified branches into approximate or
        # duplicated labels.
        _write_jsonl(private_output, counterfactuals)
        print(f"counterfactuals: anchor {index}/{len(anchors)}", flush=True)
    planner_rows, risk_rows = build_rows(
        counterfactuals, label_horizon=args.label_horizon
    )
    _write_jsonl(private_output, counterfactuals)
    for split, rows in planner_rows.items():
        _write_jsonl(args.output_dir / f"planner_{split}.jsonl", rows)
    for split, rows in risk_rows.items():
        _write_jsonl(args.output_dir / f"risk_{split}.jsonl", rows)
    label_counts = Counter(
        row["target_skill"] for rows in planner_rows.values() for row in rows
    )
    capture_counts = Counter(
        int(outcome["capture_within_h"])
        for record in counterfactuals
        for horizon in record["branches_by_horizon"].values()
        for outcome in horizon.values()
    )
    anchor_seeds = {int(record["seed"]) for record in counterfactuals}
    full_collection = set(contract_seeds("data_collection", args.contract))
    development = set(contract_seeds("development", args.contract))
    complete_collection = anchor_seeds == full_collection
    corrective_valid = (
        args.source_pool == "development_corrective"
        and bool(anchor_seeds)
        and anchor_seeds <= development
        and args.research_evidence
    )
    research_evidence = complete_collection or corrective_valid
    aggregate = {
        "schema_version": 1,
        "artifact": "p2_counterfactual_dataset_aggregate",
        "research_evidence": research_evidence,
        "research_evidence_blockers": (
            []
            if research_evidence
            else [
                "incomplete data-collection pool or unverified development-corrective source"
            ]
        ),
        "source_pool": args.source_pool,
        "anchor_count": len(counterfactuals),
        "branch_count": len(counterfactuals) * len(horizons) * len(Skill),
        "horizons": list(horizons),
        "label_horizon": args.label_horizon,
        "planner_rows": {key: len(value) for key, value in planner_rows.items()},
        "risk_rows": {key: len(value) for key, value in risk_rows.items()},
        "target_skill_counts": dict(sorted(label_counts.items())),
        "counterfactual_skill_oracle": {
            "offline_skill_accuracy": 1.0,
            "counterfactual_regret_chosen_minus_oracle": 0.0,
            "scope": "short-horizon verified branches only; not a closed-loop result",
        },
        "branch_capture_class_counts": {
            str(key): value for key, value in sorted(capture_counts.items())
        },
        "all_replays_verified": True,
        "private_states_committed": False,
    }
    args.aggregate_output.parent.mkdir(parents=True, exist_ok=True)
    args.aggregate_output.write_text(
        json.dumps(aggregate, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(aggregate, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
