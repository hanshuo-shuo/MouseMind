"""Replay development failures and collect one P2.1 corrective anchor queue."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import numpy as np

from mouse_llm.baselines.planner_mlp import NumericSkillPlanner
from mouse_llm.data.planner_schema import Preference
from mouse_llm.evaluation.closed_loop import LEGACY_GYM_SOURCE_INDICES, MLPCheckpointPolicy
from mouse_llm.evaluation.contracts import DEFAULT_P2_CONTRACT, contract_seeds
from mouse_llm.evaluation.evaluate_policy import load_action_catalog
from mouse_llm.hierarchical.context import PlannerContextBuilder
from mouse_llm.hierarchical.risk_critic import RuntimeRiskCritic
from mouse_llm.hierarchical.verified_policy import ProposeVerifyPolicy


def collect(
    *,
    env_factory: Any,
    policy: ProposeVerifyPolicy,
    replay_queue: list[dict[str, Any]],
    max_anchors_per_seed: int,
) -> list[dict[str, Any]]:
    anchors: list[dict[str, Any]] = []
    for item in replay_queue:
        seed = int(item["seed"])
        target_capture = item.get("first_capture_step")
        target_start = max(int(target_capture or 20) - 8, 0)
        env = env_factory()
        observation, _ = env.reset(seed=seed)
        policy.reset(seed)
        builder = PlannerContextBuilder(temporal_window=8)
        prefix: list[int] = []
        candidates: list[dict[str, Any]] = []
        previous_visible = False
        try:
            for step_index in range(getattr(env, "max_step", 300)):
                values = np.asarray(observation, dtype=np.float64).reshape(-1)
                legacy = np.asarray(env.legacy_policy_observation(), dtype=np.float64)
                context = builder.observe(values)
                visible = bool(values[3])
                tags = []
                if target_start <= step_index <= int(target_capture or target_start + 8):
                    tags.append("development_failure_window")
                if visible and not previous_visible:
                    tags.append("first_predator_visibility")
                if bool(values[8]):
                    tags.append("near_occlusion")
                if step_index % policy.planner_horizon == 0:
                    tags.append("planner_decision")
                if tags:
                    candidates.append(
                        {
                            "schema_version": "mousemind_p2_1_corrective_anchor_v1",
                            "seed": seed,
                            "step_index": step_index,
                            "prefix_actions": list(prefix),
                            "source_policy": policy.name,
                            "failure_mode": item["failure_mode"],
                            "tags": tags,
                            "environment_observation": values.tolist(),
                            "legacy_observation": legacy.tolist(),
                            "context": context.payload(),
                            "context_vector": context.numeric().tolist(),
                        }
                    )
                decision = policy.act(values)
                skill = decision.metadata["executed_skill"]
                if decision.metadata.get("was_overridden") and candidates:
                    candidates[-1]["tags"].append("verifier_override")
                builder.record_decision(decision.action, skill)
                prefix.append(int(decision.action))
                observation, _, terminated, truncated, _ = env.step(decision.action)
                previous_visible = visible
                if terminated or truncated:
                    break
        finally:
            env.close()
        candidates.sort(
            key=lambda row: (
                "development_failure_window" not in row["tags"],
                "verifier_override" not in row["tags"],
                abs(row["step_index"] - int(target_capture or row["step_index"])),
                row["step_index"],
            )
        )
        anchors.extend(candidates[:max_anchors_per_seed])
        print(
            f"corrective anchors: seed={seed} selected={min(len(candidates), max_anchors_per_seed)}",
            flush=True,
        )
    return anchors


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect P2.1 development failure anchors")
    parser.add_argument("--failure-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--aggregate-output", type=Path, required=True)
    parser.add_argument("--mlp-checkpoint", type=Path, required=True)
    parser.add_argument("--planner-checkpoint", type=Path, required=True)
    parser.add_argument("--risk-checkpoint", type=Path)
    parser.add_argument("--risk-threshold", type=float, default=0.9)
    parser.add_argument("--use-verifier", action="store_true")
    parser.add_argument("--planner-horizon", type=int, required=True)
    parser.add_argument("--max-anchors-per-seed", type=int, default=4)
    parser.add_argument("--contract", type=Path, default=DEFAULT_P2_CONTRACT)
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--action-catalog",
        type=Path,
        default=Path(__file__).resolve().parents[1]
        / "envs/mice/assets/action_catalog_21_05.json",
    )
    args = parser.parse_args()
    manifest = json.loads(args.failure_manifest.read_text(encoding="utf-8"))
    queue = manifest["replay_queue"]
    development_seeds = set(contract_seeds("development", args.contract))
    selected_seeds = {int(item["seed"]) for item in queue}
    if not selected_seeds <= development_seeds:
        raise ValueError("P2.1 may only collect corrective data from development seeds")
    destinations = load_action_catalog(args.action_catalog)
    specialist = MLPCheckpointPolicy(
        args.mlp_checkpoint,
        device=args.device,
        observation_indices=LEGACY_GYM_SOURCE_INDICES,
        name="p2.1-specialist",
    )
    if args.use_verifier and args.risk_checkpoint is None:
        parser.error("--risk-checkpoint is required with --use-verifier")
    verifier = (
        RuntimeRiskCritic(args.risk_checkpoint, device=args.device)
        if args.use_verifier
        else None
    )
    policy = ProposeVerifyPolicy(
        specialist=specialist,
        destinations=destinations,
        planner=NumericSkillPlanner(args.planner_checkpoint, device=args.device),
        verifier=verifier,
        risk_threshold=args.risk_threshold,
        preference=Preference.SURVIVAL_FIRST,
        planner_horizon=args.planner_horizon,
        name="numeric-verified-p2.0" if args.use_verifier else "numeric-learned-p2.0",
    )
    from mouse_llm.envs.mice import BotEvadeEnv, custom_reward

    def env_factory():
        return BotEvadeEnv(
            world_name="21_05",
            use_lppos=False,
            use_predator=True,
            max_step=300,
            reward_function=custom_reward,
            time_step=0.25,
            frame_stack_k=1,
        )

    anchors = collect(
        env_factory=env_factory,
        policy=policy,
        replay_queue=queue,
        max_anchors_per_seed=args.max_anchors_per_seed,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in anchors:
            handle.write(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n")
    os.replace(temporary, args.output)
    research_evidence = bool(manifest.get("research_evidence"))
    aggregate = {
        "schema_version": 1,
        "artifact": "p2_1_corrective_anchor_aggregate",
        "research_evidence": research_evidence,
        "research_evidence_blockers": (
            [] if research_evidence else ["source failure report is a smoke run"]
        ),
        "source_pool": "development",
        "selected_failure_episode_count": len(queue),
        "selected_seed_count": len(selected_seeds),
        "anchor_count": len(anchors),
        "final_test_data_used": False,
    }
    args.aggregate_output.write_text(
        json.dumps(aggregate, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(aggregate, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
