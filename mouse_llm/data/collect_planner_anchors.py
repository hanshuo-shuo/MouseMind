"""Collect private strategic replay anchors from frozen P2 collection seeds."""

from __future__ import annotations

import argparse
import json
import os
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

from mouse_llm.evaluation.closed_loop import MLPCheckpointPolicy
from mouse_llm.evaluation.contracts import contract_seeds, DEFAULT_P2_CONTRACT
from mouse_llm.evaluation.evaluate_policy import load_action_catalog
from mouse_llm.hierarchical.context import PlannerContextBuilder
from mouse_llm.hierarchical.policy import (
    GeometricSkillController,
    HierarchicalPolicy,
    Skill,
)


ANCHOR_SCHEMA_VERSION = "mousemind_planner_anchor_v1"


def _inside_git(path: Path) -> bool:
    for candidate in (path, *path.parents):
        if (candidate / ".git").exists():
            return True
    return False


def _atomic_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def collect_anchors(
    *,
    env_factory: Any,
    specialist: Any,
    destinations: np.ndarray,
    seeds: tuple[int, ...],
    anchor_stride: int,
    max_anchors_per_seed: int,
    planner_horizon: int,
) -> list[dict[str, Any]]:
    if anchor_stride <= 0 or max_anchors_per_seed <= 0:
        raise ValueError("anchor_stride and max_anchors_per_seed must be positive")
    controller = GeometricSkillController(specialist, destinations)
    rule_policy = HierarchicalPolicy(
        specialist=specialist,
        destinations=destinations,
        instruction="Prioritize survival and avoid being captured.",
        planner_horizon=planner_horizon,
        name="p1-rule-hierarchy",
    )
    anchors: list[dict[str, Any]] = []
    for seed_index, seed in enumerate(seeds):
        env = env_factory()
        source_kind = ("p1-rule", "direct-mlp", "skill-perturbation")[seed_index % 3]
        specialist.reset(seed)
        rule_policy.reset(seed)
        builder = PlannerContextBuilder(temporal_window=8)
        prefix_actions: list[int] = []
        candidates: list[dict[str, Any]] = []
        previous_visible = False
        observation, _ = env.reset(seed=seed)
        try:
            for step_index in range(getattr(env, "max_step", 300)):
                values = np.asarray(observation, dtype=np.float64).reshape(-1)
                legacy = np.asarray(env.legacy_policy_observation(), dtype=np.float64)
                context = builder.observe(values)
                visible = bool(values[3]) if len(values) >= 15 else bool(
                    legacy[3] != 0.0 or legacy[4] != 0.0
                )
                tags: list[str] = []
                if step_index % anchor_stride == 0:
                    tags.append("strategic_stride")
                if visible and not previous_visible:
                    tags.append("first_predator_visibility")
                if len(values) >= 15 and bool(values[7]):
                    tags.append("near_wall")
                if len(values) >= 15 and bool(values[8]):
                    tags.append("near_occlusion")
                if len(values) >= 15 and bool(values[10]):
                    tags.append("recently_puffed")
                perturbation_skill = tuple(Skill)[
                    (seed + step_index // planner_horizon) % len(Skill)
                ]
                rule_plan = rule_policy.planner.plan(
                    rule_policy.instruction, legacy, tuple(rule_policy.history)
                )
                if perturbation_skill != rule_plan.skill:
                    tags.append("planner_disagreement")
                if tags:
                    candidates.append(
                        {
                            "schema_version": ANCHOR_SCHEMA_VERSION,
                            "seed": int(seed),
                            "step_index": step_index,
                            "prefix_actions": list(prefix_actions),
                            "source_policy": source_kind,
                            "tags": tags,
                            "environment_observation": values.tolist(),
                            "legacy_observation": legacy.tolist(),
                            "context": context.payload(),
                            "context_vector": context.numeric().tolist(),
                        }
                    )
                if source_kind == "p1-rule":
                    decision = rule_policy.act(legacy)
                    skill = Skill(decision.metadata["skill"])
                elif source_kind == "direct-mlp":
                    decision = specialist.act(legacy)
                    skill = Skill.GO_TO_GOAL
                else:
                    skill = perturbation_skill
                    decision = controller.act(skill, legacy)
                builder.record_decision(decision.action, skill)
                prefix_actions.append(int(decision.action))
                observation, _, terminated, truncated, info = env.step(decision.action)
                next_values = np.asarray(observation).reshape(-1)
                captured_now = bool(len(next_values) >= 15 and next_values[10])
                captured_now |= bool(info.get("captures", 0) > 0)
                if captured_now:
                    for candidate in candidates[-3:]:
                        if "pre_capture_window" not in candidate["tags"]:
                            candidate["tags"].append("pre_capture_window")
                previous_visible = visible
                if terminated or truncated:
                    break
        finally:
            env.close()
        priority = {
            "pre_capture_window": 8,
            "recently_puffed": 7,
            "first_predator_visibility": 6,
            "near_occlusion": 5,
            "planner_disagreement": 4,
            "near_wall": 2,
            "strategic_stride": 1,
        }
        candidates.sort(
            key=lambda row: (
                -sum(priority.get(tag, 0) for tag in set(row["tags"])),
                row["step_index"],
            )
        )
        hard_limit = max(1, max_anchors_per_seed // 2)
        selected_candidates = candidates[:hard_limit]
        selected_steps = {row["step_index"] for row in selected_candidates}
        coverage_candidates = sorted(
            (
                row
                for row in candidates
                if row["step_index"] not in selected_steps
                and "strategic_stride" in row["tags"]
            ),
            key=lambda row: row["step_index"],
        )
        remaining = max_anchors_per_seed - len(selected_candidates)
        if coverage_candidates and remaining > 0:
            coverage_indices = np.linspace(
                0, len(coverage_candidates) - 1, num=min(remaining, len(coverage_candidates))
            ).round().astype(int)
            selected_candidates.extend(
                coverage_candidates[index] for index in sorted(set(coverage_indices))
            )
        if len(selected_candidates) < max_anchors_per_seed:
            selected_ids = {id(row) for row in selected_candidates}
            selected_candidates.extend(
                row
                for row in candidates
                if id(row) not in selected_ids
            )
        selected = sorted(
            selected_candidates[:max_anchors_per_seed], key=lambda row: row["step_index"]
        )
        anchors.extend(selected)
        print(
            f"anchors: seed={seed} source={source_kind} selected={len(selected)}",
            flush=True,
        )
    return anchors


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect private P2 replay anchors")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--aggregate-output", type=Path, required=True)
    parser.add_argument("--mlp-checkpoint", type=Path, required=True)
    parser.add_argument("--contract", type=Path, default=DEFAULT_P2_CONTRACT)
    parser.add_argument("--seed-limit", type=int, default=0)
    parser.add_argument("--anchor-stride", type=int, default=8)
    parser.add_argument("--max-anchors-per-seed", type=int, default=6)
    parser.add_argument("--planner-horizon", type=int, default=4)
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
    parser.add_argument("--allow-repo-output", action="store_true")
    args = parser.parse_args()
    if _inside_git(args.output.resolve()) and not args.allow_repo_output:
        raise ValueError("Private replay anchors must be written outside Git")
    full_collection_seeds = contract_seeds("data_collection", args.contract)
    seeds = full_collection_seeds
    if args.seed_limit > 0:
        seeds = seeds[: args.seed_limit]
    destinations = load_action_catalog(args.action_catalog)
    specialist = MLPCheckpointPolicy(
        args.mlp_checkpoint,
        device=args.device,
        observation_indices=tuple(range(10)),
        name="p2-anchor-mlp",
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

    anchors = collect_anchors(
        env_factory=env_factory,
        specialist=specialist,
        destinations=destinations,
        seeds=seeds,
        anchor_stride=args.anchor_stride,
        max_anchors_per_seed=args.max_anchors_per_seed,
        planner_horizon=args.planner_horizon,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for anchor in anchors:
            handle.write(json.dumps(anchor, sort_keys=True, separators=(",", ":")))
            handle.write("\n")
    os.replace(temporary, args.output)
    tags = Counter(tag for anchor in anchors for tag in anchor["tags"])
    sources = Counter(anchor["source_policy"] for anchor in anchors)
    aggregate = {
        "schema_version": 1,
        "artifact": "p2_anchor_collection_aggregate",
        "research_evidence": seeds == full_collection_seeds,
        "seed_pool": "data_collection",
        "seed_count": len(seeds),
        "anchor_count": len(anchors),
        "source_policy_counts": dict(sorted(sources.items())),
        "tag_counts": dict(sorted(tags.items())),
        "private_replay_records_committed": False,
        "research_evidence_blockers": (
            [] if seeds == full_collection_seeds else ["partial collection smoke"]
        ),
    }
    _atomic_json(args.aggregate_output, aggregate)
    print(json.dumps(aggregate, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
