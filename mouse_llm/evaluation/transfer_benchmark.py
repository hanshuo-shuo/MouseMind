"""Frozen BotEvade-to-Oasis full-stack and planner-isolation benchmark."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from mouse_llm.baselines.planner_mlp import NumericSkillPlanner
from mouse_llm.data.planner_schema import Preference
from mouse_llm.evaluation.closed_loop import (
    LEGACY_GYM_SOURCE_INDICES,
    EpisodeResult,
    MLPCheckpointPolicy,
    Policy,
    PolicyDecision,
    RandomPolicy,
    _write_outputs,
    bootstrap_mean,
    build_report,
    paired_differences,
    run_policy,
)
from mouse_llm.evaluation.contracts import (
    DEFAULT_TRANSFER_CONTRACT,
    load_transfer_contract,
    transfer_contract_seeds,
)
from mouse_llm.evaluation.evaluate_policy import load_action_catalog
from mouse_llm.hierarchical.context import PlannerContext
from mouse_llm.hierarchical.policy import (
    HierarchicalPolicy,
    PlannerDecision,
    Skill,
)
from mouse_llm.hierarchical.verified_policy import ProposeVerifyPolicy


DEFAULT_ACTION_CATALOG = Path(
    "mouse_llm/envs/mice/assets/action_catalog_21_05.json"
)
SOURCE_MANIFEST_FILES = (
    "mouse_llm/baselines/planner_mlp.py",
    "mouse_llm/envs/mice/oasis.py",
    "mouse_llm/envs/mice/_vendor/cellworld_game/tasks/oasis.py",
    "mouse_llm/evaluation/audit_transfer_compatibility.py",
    "mouse_llm/evaluation/closed_loop.py",
    "mouse_llm/evaluation/contracts.py",
    "mouse_llm/evaluation/contracts/cross_task_transfer_v1.json",
    "mouse_llm/evaluation/transfer_benchmark.py",
    "mouse_llm/hierarchical/context.py",
    "mouse_llm/hierarchical/minimind_planner.py",
    "mouse_llm/hierarchical/policy.py",
    "mouse_llm/hierarchical/verified_policy.py",
)
TRANSFER_SUMMARY_FIELDS = {
    "ordered_goals_completed": "goals_completed",
    "goal_completion_rate": "goal_completion_rate",
    "return_completed_rate": "return_completed",
    "objectives_completed": "objectives_completed",
}


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def source_manifest_sha256(repo_root: Path | None = None) -> str:
    root = repo_root or Path(__file__).resolve().parents[2]
    digest = hashlib.sha256()
    for relative in SOURCE_MANIFEST_FILES:
        path = root / relative
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


class RuleContextPlanner:
    """P1 decision rule expressed through the frozen learned-planner API."""

    def plan_context(
        self, context: PlannerContext, preference: Preference
    ) -> PlannerDecision:
        if preference == Preference.HOLD:
            return PlannerDecision(Skill.HOLD_POSITION, "hold_preference")
        if (
            context.predator_visible
            and preference in {Preference.SURVIVAL_FIRST, Preference.BALANCED}
        ):
            return PlannerDecision(Skill.EVADE_PREDATOR, "visible_predator")
        return PlannerDecision(Skill.GO_TO_GOAL, "goal_progress")


class GoalCoordinatePolicy:
    """Parameter-free target-task controller used to isolate planner transfer."""

    observation_mode = "transfer"

    def __init__(self, destinations: np.ndarray, *, name: str = "aligned-goal-only"):
        self.name = name
        self.destinations = np.asarray(destinations, dtype=np.float64)
        self.action_count = len(self.destinations)

    def reset(self, seed: int) -> None:
        del seed

    def act(self, observation: np.ndarray) -> PolicyDecision:
        values = np.asarray(observation, dtype=np.float64).reshape(-1)
        if len(values) < 17 or not np.isfinite(values).all():
            raise ValueError("Goal-coordinate policy requires transfer observation")
        goal = values[15:17]
        action = int(np.linalg.norm(self.destinations - goal, axis=1).argmin())
        return PolicyDecision(
            action,
            metadata={
                "skill": Skill.GO_TO_GOAL.value,
                "replanned": False,
                "transfer_adapter": "active_goal_coordinate",
            },
        )


class _PlannerIsolationSpecialist:
    """Guard proving that aligned GO_TO_GOAL never calls the P2 specialist."""

    name = "planner-isolation-specialist"

    def reset(self, seed: int) -> None:
        del seed

    def act(self, observation: np.ndarray) -> PolicyDecision:
        del observation
        raise RuntimeError(
            "Planner-isolation transfer must use the active-goal controller"
        )


def _load_compatibility_audit(
    path: Path, *, contract_path: Path
) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    expected_contract_sha = hashlib.sha256(contract_path.read_bytes()).hexdigest()
    if (
        payload.get("schema_version") != 1
        or payload.get("artifact") != "cross_task_transfer_compatibility_audit"
        or payload.get("research_evidence") is not True
        or payload.get("contract_sha256") != expected_contract_sha
        or payload.get("action_contract", {}).get("verified") is not True
        or payload.get("observation_contract", {}).get("verified") is not True
        or payload.get("reset_contract", {}).get("verified") is not True
        or payload.get("terminal_contract", {}).get("verified") is not True
    ):
        raise ValueError(f"Transfer compatibility audit is not verified: {path}")
    return payload


def _add_transfer_metrics(
    report: dict[str, Any],
    results: dict[str, list[EpisodeResult]],
    *,
    bootstrap_seed: int,
) -> None:
    for policy_offset, (name, rows) in enumerate(results.items()):
        summary = report["policies"][name]
        for metric_offset, (metric, field) in enumerate(
            TRANSFER_SUMMARY_FIELDS.items()
        ):
            summary[metric] = bootstrap_mean(
                [float(getattr(row, field)) for row in rows],
                seed=bootstrap_seed + policy_offset * 100 + metric_offset,
            )


def _paired_transfer_differences(
    reference: Sequence[EpisodeResult],
    candidate: Sequence[EpisodeResult],
    *,
    seed: int,
) -> dict[str, Any]:
    result = paired_differences(reference, candidate, seed=seed)
    reference_by_seed = {row.seed: row for row in reference}
    candidate_by_seed = {row.seed: row for row in candidate}
    for offset, (name, field) in enumerate(TRANSFER_SUMMARY_FIELDS.items()):
        differences = [
            float(getattr(candidate_by_seed[item], field))
            - float(getattr(reference_by_seed[item], field))
            for item in sorted(reference_by_seed)
        ]
        result[name] = bootstrap_mean(differences, seed=seed + 100 + offset)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, default=DEFAULT_TRANSFER_CONTRACT)
    parser.add_argument("--compatibility-audit", type=Path, required=True)
    parser.add_argument("--action-catalog", type=Path, default=DEFAULT_ACTION_CATALOG)
    parser.add_argument(
        "--policies",
        nargs="+",
        default=[
            "random",
            "literal-direct-mlp",
            "literal-p1-rule",
            "literal-numeric",
            "literal-minimind",
            "aligned-goal-only",
            "aligned-p1-rule",
            "aligned-numeric",
            "aligned-minimind",
        ],
        choices=(
            "random",
            "literal-direct-mlp",
            "literal-p1-rule",
            "literal-numeric",
            "literal-minimind",
            "aligned-goal-only",
            "aligned-p1-rule",
            "aligned-numeric",
            "aligned-minimind",
            "aligned-minimind-no-history",
            "aligned-minimind-no-instruction",
        ),
    )
    parser.add_argument(
        "--reference-policy", default="aligned-goal-only"
    )
    parser.add_argument(
        "--seed-pool", choices=("development", "final_test"), default="development"
    )
    parser.add_argument("--condition", choices=("seen_instruction", "unseen_instruction"), default="seen_instruction")
    parser.add_argument("--seed-limit", type=int, default=0)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--mlp-checkpoint", type=Path)
    parser.add_argument("--planner-checkpoint", type=Path)
    parser.add_argument("--base-weight", type=Path)
    parser.add_argument("--skill-lora-weight", type=Path)
    parser.add_argument("--tokenizer", type=Path, default=Path("model"))
    parser.add_argument("--hidden-size", type=int, default=768)
    parser.add_argument("--num-hidden-layers", type=int, default=8)
    parser.add_argument("--max-seq-len", type=int, default=1024)
    parser.add_argument("--max-new-tokens", type=int, default=24)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    contract = load_transfer_contract(args.contract)
    compatibility = _load_compatibility_audit(
        args.compatibility_audit, contract_path=args.contract
    )
    seeds = transfer_contract_seeds(args.seed_pool, args.contract)
    full_pool_count = len(seeds)
    if args.seed_limit:
        if args.seed_limit <= 0:
            parser.error("--seed-limit must be positive")
        if not args.smoke:
            parser.error("Partial contract runs require --smoke")
        seeds = seeds[: args.seed_limit]
    destinations = load_action_catalog(args.action_catalog)
    selection = contract["selection"]
    preference = Preference(selection["preference"])
    instruction_split = (
        "unseen_test" if args.condition == "unseen_instruction" else "train"
    )

    from mouse_llm.envs.mice import OasisEnv, oasis_reward

    def env_factory():
        return OasisEnv(
            world_name=contract["target"]["world"],
            goal_locations=[
                tuple(values) for values in contract["target"]["goal_locations"]
            ],
            goal_threshold=contract["target"]["goal_threshold"],
            use_lppos=False,
            use_predator=True,
            frame_stack_k=1,
            max_step=contract["max_steps"],
            time_step=contract["time_step"],
            reward_function=oasis_reward(),
        )

    checkpoint_policies = {
        "literal-direct-mlp",
        "literal-p1-rule",
        "literal-numeric",
        "literal-minimind",
    }
    if checkpoint_policies.intersection(args.policies) and args.mlp_checkpoint is None:
        parser.error("Literal frozen-stack policies require --mlp-checkpoint")

    def specialist(name: str) -> MLPCheckpointPolicy:
        assert args.mlp_checkpoint is not None
        return MLPCheckpointPolicy(
            args.mlp_checkpoint,
            device=args.device,
            observation_indices=LEGACY_GYM_SOURCE_INDICES,
            name=name,
        )

    def numeric_planner():
        if args.planner_checkpoint is None:
            parser.error("Numeric policies require --planner-checkpoint")
        return NumericSkillPlanner(args.planner_checkpoint, device=args.device)

    def minimind_planner(ablation: str | None = None):
        if args.base_weight is None or args.skill_lora_weight is None:
            parser.error("MiniMind policies require base and skill LoRA weights")
        from mouse_llm.hierarchical.minimind_planner import MiniMindSkillPlanner

        return MiniMindSkillPlanner(
            base_weight=args.base_weight,
            lora_weight=args.skill_lora_weight,
            tokenizer_path=args.tokenizer,
            device=args.device,
            hidden_size=args.hidden_size,
            num_hidden_layers=args.num_hidden_layers,
            max_seq_len=args.max_seq_len,
            max_new_tokens=args.max_new_tokens,
            instruction_split=instruction_split,
            ablation=ablation,
        )

    def learned_policy(
        *, name: str, planner: Any, aligned: bool
    ) -> ProposeVerifyPolicy:
        return ProposeVerifyPolicy(
            specialist=(
                _PlannerIsolationSpecialist()
                if aligned
                else specialist(f"{name}-specialist")
            ),
            destinations=destinations,
            planner=planner,
            preference=preference,
            planner_horizon=(
                selection["rule_planner_horizon"]
                if name.endswith("p1-rule")
                else selection["planner_horizon"]
            ),
            evade_distance=selection["evade_distance"],
            name=name,
            observation_mode="transfer",
            goal_destination_indices=(15, 16) if aligned else None,
        )

    policies: list[Policy] = []
    for name in args.policies:
        if name == "random":
            policies.append(RandomPolicy(action_count=len(destinations)))
        elif name == "literal-direct-mlp":
            policies.append(specialist(name))
        elif name == "literal-p1-rule":
            policies.append(
                HierarchicalPolicy(
                    specialist=specialist(f"{name}-specialist"),
                    destinations=destinations,
                    instruction="Prioritize survival and avoid being captured.",
                    planner_horizon=selection["rule_planner_horizon"],
                    evade_distance=selection["evade_distance"],
                    name=name,
                )
            )
        elif name == "literal-numeric":
            policies.append(
                learned_policy(name=name, planner=numeric_planner(), aligned=False)
            )
        elif name == "literal-minimind":
            policies.append(
                learned_policy(name=name, planner=minimind_planner(), aligned=False)
            )
        elif name == "aligned-goal-only":
            policies.append(GoalCoordinatePolicy(destinations, name=name))
        elif name == "aligned-p1-rule":
            policies.append(
                learned_policy(name=name, planner=RuleContextPlanner(), aligned=True)
            )
        elif name == "aligned-numeric":
            policies.append(
                learned_policy(name=name, planner=numeric_planner(), aligned=True)
            )
        elif name.startswith("aligned-minimind"):
            ablation = (
                "no_temporal_history"
                if name.endswith("no-history")
                else "instruction_removed"
                if name.endswith("no-instruction")
                else None
            )
            policies.append(
                learned_policy(
                    name=name, planner=minimind_planner(ablation), aligned=True
                )
            )

    if args.reference_policy not in {policy.name for policy in policies}:
        parser.error("--reference-policy must be included in --policies")
    results = {
        policy.name: run_policy(
            env_factory,
            policy,
            seeds=seeds,
            control_budget_seconds=contract["time_step"],
            warmup_actions=3,
        )
        for policy in policies
    }
    seed_digest = hashlib.sha256(
        ",".join(map(str, seeds)).encode("ascii")
    ).hexdigest()
    checkpoint_sha256: dict[str, str] = {}
    if checkpoint_policies.intersection(args.policies):
        assert args.mlp_checkpoint is not None
        checkpoint_sha256["bot_evade_mlp_specialist"] = _file_sha256(
            args.mlp_checkpoint
        )
    if any("numeric" in name for name in args.policies):
        assert args.planner_checkpoint is not None
        checkpoint_sha256["numeric_skill_planner"] = _file_sha256(
            args.planner_checkpoint
        )
    if any("minimind" in name for name in args.policies):
        assert args.base_weight is not None and args.skill_lora_weight is not None
        checkpoint_sha256["minimind_base"] = _file_sha256(args.base_weight)
        checkpoint_sha256["minimind_skill_lora"] = _file_sha256(
            args.skill_lora_weight
        )
    metadata = {
        "source_environment": contract["source"]["environment"],
        "target_environment": contract["target"]["environment"],
        "world": contract["target"]["world"],
        "contract_name": contract["contract_name"],
        "contract_sha256": hashlib.sha256(args.contract.read_bytes()).hexdigest(),
        "compatibility_audit_sha256": hashlib.sha256(
            args.compatibility_audit.read_bytes()
        ).hexdigest(),
        "code_manifest_sha256": source_manifest_sha256(),
        "code_manifest_files": list(SOURCE_MANIFEST_FILES),
        "checkpoint_sha256": checkpoint_sha256,
        "seed_pool": args.seed_pool,
        "episode_count": len(seeds),
        "full_pool_episode_count": full_pool_count,
        "seed_start": seeds[0],
        "seed_sha256": seed_digest,
        "condition": args.condition,
        "instruction_split": instruction_split,
        "preference": preference.value,
        "literal_low_level_transfer_compatible": False,
        "aligned_goal_controller": "parameter-free nearest action to active goal",
        "target_adaptation_used": False,
        "goal_projection": contract["target"]["discrete_goal_projection"],
        "synthetic": False,
        "research_evidence": not args.smoke,
        "research_evidence_blockers": ["partial smoke run"] if args.smoke else [],
    }
    report = build_report(
        results,
        seed=seeds[0],
        reference_policy=args.reference_policy,
        metadata=metadata,
    )
    report["experiment"] = "mousemind_frozen_cross_task_transfer"
    _add_transfer_metrics(report, results, bootstrap_seed=seeds[0] + 5000)
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
                        seed=seeds[0] + 10000 + offset * 100,
                    )
                )
    report["within_mode_paired_comparisons"] = within_mode
    report["compatibility"] = {
        "action_contract": compatibility["action_contract"],
        "observation_contract": compatibility["observation_contract"],
        "reset_contract": compatibility["reset_contract"],
        "terminal_contract": compatibility["terminal_contract"],
    }
    _write_outputs(args.output_dir, report, results)
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
