import pytest

from mouse_llm.data.build_verified_dpo import build_pairs, seed_split
from mouse_llm.data.planner_schema import Preference, select_skill


def _outcome(capture, progress):
    return {
        "capture_within_h": capture,
        "captures": capture,
        "goal_distance_change": progress,
        "predator_exposure_steps": 0,
        "path_length": 1,
        "near_occlusion_exposure": 0,
        "terminal_goal_event": 0,
    }


def _record(seed):
    return {
        "anchor_id": f"anchor-{seed}", "seed": seed, "context": {},
        "replay_verification": {"verified": True},
        "branches_by_horizon": {"8": {
            "go_to_goal": _outcome(1, 1),
            "evade_predator": _outcome(0, 0.5),
            "hold_position": _outcome(0, 0),
        }},
    }


def test_best_worst_pairs_preserve_seed_isolation_and_exact_target():
    seeds = {split: next(seed for seed in range(1000) if seed_split(seed) == split)
             for split in ("train", "validation", "test")}
    rows = build_pairs([_record(seed) for seed in seeds.values()], final_seeds=set(), label_horizon=8)
    assert {row["seed"] for row in rows["train"]} == {seeds["train"]}
    assert {row["seed"] for row in rows["validation"]} == {seeds["validation"]}
    assert {row["seed"] for row in rows["seen_test"]} == {seeds["test"]}
    assert {row["seed"] for row in rows["unseen_test"]} == {seeds["test"]}
    row = next(row for row in rows["train"] if row["preference"] == Preference.SURVIVAL_FIRST.value)
    target, utilities = select_skill(_record(seeds["train"])["branches_by_horizon"]["8"], Preference.SURVIVAL_FIRST)
    assert row["target_skill"] == target.value
    assert row["chosen_utility"] == max(utilities.values())
    assert row["rejected_utility"] == min(utilities.values())
    assert row["margin"] > 0


def test_final_seed_and_unverified_replay_fail_closed():
    record = _record(1)
    with pytest.raises(ValueError, match="Final evaluation seed"):
        build_pairs([record], final_seeds={1}, label_horizon=8)
    record["replay_verification"]["verified"] = False
    with pytest.raises(ValueError, match="Unverified"):
        build_pairs([record], final_seeds=set(), label_horizon=8)


def test_dpo_loss_is_neutral_at_sft_initialization_and_prefers_chosen():
    import torch
    from mouse_llm.training.train_verified_dpo import dpo_loss

    chosen = torch.tensor([1.0], requires_grad=True)
    rejected = torch.tensor([0.0], requires_grad=True)
    loss = dpo_loss(chosen, rejected, chosen.detach(), rejected.detach(), 0.1)
    assert loss.item() == pytest.approx(0.69314718, abs=1e-6)
    loss.backward()
    assert chosen.grad.item() < 0
    assert rejected.grad.item() > 0


def test_final_trace_context_round_trip():
    from mouse_llm.evaluation.diagnose_verified_dpo import restore_context
    from mouse_llm.hierarchical.context import PlannerContext

    context = PlannerContext(
        prey_x=0.1, prey_y=0.2, prey_direction=0.3, predator_visible=True,
        predator_x=0.4, predator_y=0.5, predator_direction=0.6,
        time_since_predator_seen=0, near_wall=False, near_occlusion=True,
        puffed=False, puff_cooled_down=True, goal_distance=0.7,
        recent_prey_positions=((0.1, 0.2),), recent_predator_visibility=(1,),
        recent_low_level_actions=(2,), recent_high_level_skills=("evade_predator",),
    )
    assert restore_context(context.payload()).serialize() == context.serialize()
