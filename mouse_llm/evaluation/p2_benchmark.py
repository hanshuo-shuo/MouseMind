"""Frozen-contract ID/OOD benchmark for learned and verified hierarchies."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from mouse_llm.baselines.planner_mlp import NumericSkillPlanner
from mouse_llm.data.planner_schema import Preference
from mouse_llm.evaluation.closed_loop import (
    LEGACY_GYM_SOURCE_INDICES,
    MLPCheckpointPolicy,
    MiniMindPolicy,
    Policy,
    RandomPolicy,
    _write_outputs,
    build_report,
    run_policy,
)
from mouse_llm.evaluation.contracts import (
    DEFAULT_P2_CONTRACT,
    contract_seeds,
    load_p2_contract,
)
from mouse_llm.evaluation.evaluate_policy import load_action_catalog
from mouse_llm.hierarchical.policy import HierarchicalPolicy
from mouse_llm.hierarchical.risk_critic import RuntimeRiskCritic
from mouse_llm.hierarchical.verified_policy import ProposeVerifyPolicy


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the frozen MouseMind P2 benchmark")
    parser.add_argument(
        "--policies",
        nargs="+",
        choices=(
            "random",
            "direct-mlp",
            "direct-minimind-base",
            "direct-minimind-lora",
            "p1-rule",
            "numeric-learned",
            "numeric-verified",
            "minimind-learned",
            "minimind-verified",
            "minimind-no-history",
            "minimind-no-instruction",
        ),
        required=True,
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--contract", type=Path, default=DEFAULT_P2_CONTRACT)
    parser.add_argument(
        "--seed-pool", choices=("development", "final_id_test"), required=True
    )
    parser.add_argument("--seed-limit", type=int, default=0)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--reference-policy")
    parser.add_argument("--mlp-checkpoint", type=Path, required=True)
    parser.add_argument("--planner-checkpoint", type=Path)
    parser.add_argument("--risk-checkpoint", type=Path)
    parser.add_argument("--risk-threshold", type=float, default=0.5)
    parser.add_argument("--planner-horizon", type=int, default=4)
    parser.add_argument("--p1-planner-horizon", type=int, default=4)
    parser.add_argument("--evade-distance", type=float, default=0.35)
    parser.add_argument(
        "--preference",
        choices=tuple(item.value for item in Preference),
        default=Preference.SURVIVAL_FIRST.value,
    )
    parser.add_argument(
        "--ood",
        choices=("id", "faster_predator_020", "faster_predator_025", "shorter_los_070", "unseen_language"),
        default="id",
    )
    parser.add_argument("--base-weight", type=Path)
    parser.add_argument("--skill-lora-weight", type=Path)
    parser.add_argument("--direct-lora-weight", type=Path)
    parser.add_argument("--tokenizer", type=Path, default=Path("model"))
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--hidden-size", type=int, default=768)
    parser.add_argument("--num-hidden-layers", type=int, default=8)
    parser.add_argument("--max-seq-len", type=int, default=1024)
    parser.add_argument("--max-new-tokens", type=int, default=24)
    parser.add_argument(
        "--action-catalog",
        type=Path,
        default=Path(__file__).resolve().parents[1]
        / "envs/mice/assets/action_catalog_21_05.json",
    )
    args = parser.parse_args()
    if len(set(args.policies)) != len(args.policies):
        raise ValueError("policies must be unique")
    if args.planner_horizon <= 0 or args.p1_planner_horizon <= 0:
        raise ValueError("planner horizons must be positive")
    if not 0.0 <= args.risk_threshold <= 1.0:
        raise ValueError("risk-threshold must be in [0, 1]")
    contract = load_p2_contract(args.contract)
    seeds = contract_seeds(args.seed_pool, args.contract)
    full_pool_count = len(seeds)
    if args.seed_limit > 0:
        if not args.smoke:
            raise ValueError("Partial contract runs must be explicitly marked --smoke")
        seeds = seeds[: args.seed_limit]
    destinations = load_action_catalog(args.action_catalog)
    preference = Preference(args.preference)
    if args.ood == "unseen_language":
        instruction_split = "unseen_test"
    else:
        instruction_split = "train"
    speed_ratio = {
        "faster_predator_020": 0.20,
        "faster_predator_025": 0.25,
    }.get(args.ood, 0.15)
    max_los = 0.70 if args.ood == "shorter_los_070" else 1.0
    from mouse_llm.envs.mice import BotEvadeEnv, custom_reward

    def env_factory():
        return BotEvadeEnv(
            world_name=contract["world"],
            use_lppos=False,
            use_predator=True,
            max_step=contract["max_steps"],
            reward_function=custom_reward,
            time_step=contract["time_step"],
            frame_stack_k=1,
            predator_prey_forward_speed_ratio=speed_ratio,
            max_line_of_sight_distance=max_los,
        )

    def specialist(name: str) -> MLPCheckpointPolicy:
        return MLPCheckpointPolicy(
            args.mlp_checkpoint,
            device=args.device,
            observation_indices=LEGACY_GYM_SOURCE_INDICES,
            name=name,
        )

    verifier = (
        RuntimeRiskCritic(args.risk_checkpoint, device=args.device)
        if args.risk_checkpoint is not None
        else None
    )
    policies: list[Policy] = []
    for name in args.policies:
        if name == "random":
            policies.append(RandomPolicy(action_count=len(destinations)))
        elif name == "direct-mlp":
            policies.append(specialist("direct-mlp"))
        elif name in {"direct-minimind-base", "direct-minimind-lora"}:
            if args.base_weight is None:
                parser.error(f"--base-weight is required for {name}")
            direct_lora = None
            if name == "direct-minimind-lora":
                if args.direct_lora_weight is None:
                    parser.error("--direct-lora-weight is required for direct-minimind-lora")
                direct_lora = args.direct_lora_weight
            policies.append(
                MiniMindPolicy(
                    name=name,
                    base_weight=args.base_weight,
                    lora_weight=direct_lora,
                    tokenizer_path=args.tokenizer,
                    action_count=len(destinations),
                    hidden_size=args.hidden_size,
                    num_hidden_layers=args.num_hidden_layers,
                    max_seq_len=256,
                    max_new_tokens=24,
                    device=args.device,
                    observation_indices=LEGACY_GYM_SOURCE_INDICES,
                    decode_mode="json-constrained",
                    fallback_action=0,
                )
            )
        elif name == "p1-rule":
            policies.append(
                HierarchicalPolicy(
                    specialist=specialist("p1-specialist"),
                    destinations=destinations,
                    instruction="Prioritize survival and avoid being captured.",
                    planner_horizon=args.p1_planner_horizon,
                    evade_distance=args.evade_distance,
                    name="p1-rule",
                )
            )
        elif name.startswith("numeric"):
            if args.planner_checkpoint is None:
                parser.error(f"--planner-checkpoint is required for {name}")
            learned_planner = NumericSkillPlanner(
                args.planner_checkpoint, device=args.device
            )
            verified = name.endswith("verified")
            if verified and verifier is None:
                parser.error("--risk-checkpoint is required for numeric-verified")
            policies.append(
                ProposeVerifyPolicy(
                    specialist=specialist(f"{name}-specialist"),
                    destinations=destinations,
                    planner=learned_planner,
                    preference=preference,
                    planner_horizon=args.planner_horizon,
                    verifier=verifier if verified else None,
                    risk_threshold=args.risk_threshold,
                    evade_distance=args.evade_distance,
                    name=name,
                )
            )
        else:
            if args.base_weight is None or args.skill_lora_weight is None:
                parser.error("MiniMind hierarchy requires base and skill LoRA weights")
            from mouse_llm.hierarchical.minimind_planner import MiniMindSkillPlanner

            learned_planner = MiniMindSkillPlanner(
                base_weight=args.base_weight,
                lora_weight=args.skill_lora_weight,
                tokenizer_path=args.tokenizer,
                device=args.device,
                hidden_size=args.hidden_size,
                num_hidden_layers=args.num_hidden_layers,
                max_seq_len=args.max_seq_len,
                max_new_tokens=args.max_new_tokens,
                instruction_split=instruction_split,
                ablation=(
                    "no_temporal_history"
                    if name == "minimind-no-history"
                    else "instruction_removed"
                    if name == "minimind-no-instruction"
                    else None
                ),
            )
            verified = name.endswith("verified")
            if verified and verifier is None:
                parser.error("--risk-checkpoint is required for minimind-verified")
            policies.append(
                ProposeVerifyPolicy(
                    specialist=specialist(f"{name}-specialist"),
                    destinations=destinations,
                    planner=learned_planner,
                    preference=preference,
                    planner_horizon=args.planner_horizon,
                    verifier=verifier if verified else None,
                    risk_threshold=args.risk_threshold,
                    evade_distance=args.evade_distance,
                    name=name,
                )
            )
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
    reference = args.reference_policy or policies[0].name
    seed_digest = hashlib.sha256(",".join(map(str, seeds)).encode("ascii")).hexdigest()
    metadata = {
        "environment": contract["environment"],
        "world": contract["world"],
        "contract_name": contract["contract_name"],
        "contract_sha256": hashlib.sha256(args.contract.read_bytes()).hexdigest(),
        "seed_pool": args.seed_pool,
        "episode_count": len(seeds),
        "full_pool_episode_count": full_pool_count,
        "seed_start": seeds[0],
        "seed_sha256": seed_digest,
        "max_steps": contract["max_steps"],
        "control_budget_seconds": contract["time_step"],
        "ood_condition": args.ood,
        "environment_parameters": {
            "predator_prey_forward_speed_ratio": speed_ratio,
            "max_line_of_sight_distance": max_los,
        },
        "preference": preference.value,
        "instruction_split": instruction_split,
        "planner_horizon": args.planner_horizon,
        "p1_planner_horizon": args.p1_planner_horizon,
        "evade_distance": args.evade_distance,
        "risk_threshold": args.risk_threshold,
        "reference_policy": reference,
        "synthetic": False,
        "research_evidence": not args.smoke,
        "research_evidence_blockers": ["partial smoke run"] if args.smoke else [],
    }
    report = build_report(
        results,
        seed=seeds[0],
        reference_policy=reference,
        metadata=metadata,
    )
    _write_outputs(args.output_dir, report, results)
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
