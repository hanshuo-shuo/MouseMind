"""Compare SFT and DPO on identical final rollout states using existing exact replay."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

from mouse_llm.data.planner_schema import Preference, select_skill
from mouse_llm.evaluation.closed_loop import MLPCheckpointPolicy
from mouse_llm.evaluation.contracts import DEFAULT_P2_CONTRACT, contract_seeds, load_p2_contract
from mouse_llm.evaluation.evaluate_policy import load_action_catalog
from mouse_llm.hierarchical.context import PlannerContext
from mouse_llm.hierarchical.counterfactual import branch_anchor
from mouse_llm.hierarchical.minimind_planner import MiniMindSkillPlanner
from mouse_llm.hierarchical.policy import Skill


def restore_context(payload: dict) -> PlannerContext:
    current = payload["current"]
    history = payload["history"]
    return PlannerContext(
        **current,
        recent_prey_positions=tuple(tuple(p) for p in history["recent_prey_positions"]),
        recent_predator_visibility=tuple(history["recent_predator_visibility"]),
        recent_low_level_actions=tuple(history["recent_low_level_actions"]),
        recent_high_level_skills=tuple(history["recent_high_level_skills"]),
        temporal_window=payload["temporal_window"],
        schema_version=payload["schema_version"],
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--base-weight", type=Path, required=True)
    parser.add_argument("--sft-lora-weight", type=Path, required=True)
    parser.add_argument("--mlp-checkpoint", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--contract", type=Path, default=DEFAULT_P2_CONTRACT)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    rows = [json.loads(line) for line in args.trace.read_text().splitlines() if line]
    final_seeds = set(contract_seeds("final_id_test", args.contract))
    if {int(row["seed"]) for row in rows} != final_seeds:
        raise ValueError("Diagnostic trace does not cover the frozen final seed pool")
    contract = load_p2_contract(args.contract)
    destinations = load_action_catalog(Path(__file__).resolve().parents[1] / "envs/mice/assets/action_catalog_21_05.json")
    from mouse_llm.envs.mice import BotEvadeEnv, custom_reward

    def env_factory():
        return BotEvadeEnv(
            world_name=contract["world"], use_lppos=False, use_predator=True,
            max_step=contract["max_steps"], reward_function=custom_reward,
            time_step=contract["time_step"], frame_stack_k=1,
            predator_prey_forward_speed_ratio=0.15,
        )

    specialist = MLPCheckpointPolicy(args.mlp_checkpoint, device="cpu", observation_indices=tuple(range(10)), name="dpo-diagnostic-specialist")
    sft = MiniMindSkillPlanner(base_weight=args.base_weight, lora_weight=args.sft_lora_weight, tokenizer_path=args.tokenizer, device=args.device)
    counts: dict[str, Counter] = defaultdict(Counter)
    for index, row in enumerate(rows, 1):
        if index % 100 == 0:
            print(f"diagnostic_states={index}/{len(rows)}", flush=True)
        context = restore_context(row["context"])
        sft_skill = sft.plan_context(context, Preference.SURVIVAL_FIRST).skill
        dpo_skill = Skill(row["dpo_skill"])
        groups = ["all"]
        if context.predator_visible:
            groups.append("predator_visible")
        if context.near_occlusion:
            groups.append("near_occlusion")
        for group in groups:
            counts[group]["decisions"] += 1
        if sft_skill == dpo_skill:
            continue
        for group in groups:
            counts[group]["changed"] += 1
        try:
            branched = branch_anchor(env_factory, anchor=row, specialist=specialist, destinations=destinations, horizon=8)
        except RuntimeError as exc:
            if "Deterministic replay verification failed" not in str(exc):
                raise
            for group in groups:
                counts[group]["unverified_changed"] += 1
            print(f"replay_unverified_index={index}", flush=True)
            continue
        oracle, _ = select_skill(branched["branches"], Preference.SURVIVAL_FIRST)
        for group in groups:
            if sft_skill != oracle and dpo_skill == oracle:
                counts[group]["sft_wrong_dpo_correct"] += 1
            elif sft_skill == oracle and dpo_skill != oracle:
                counts[group]["sft_correct_dpo_wrong"] += 1
            else:
                counts[group]["both_wrong"] += 1
    report = {"schema_version": 1, "scope": "DPO final rollout replan states; exact-state horizon-8 oracle on changed decisions", "trace_states": len(rows), "groups": {name: dict(counter) for name, counter in counts.items()}}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
