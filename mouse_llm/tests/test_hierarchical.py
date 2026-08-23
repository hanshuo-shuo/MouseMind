from __future__ import annotations

import numpy as np

from mouse_llm.evaluation.closed_loop import PolicyDecision
from mouse_llm.hierarchical import HierarchicalPolicy, InstructionSkillPlanner, Skill


class _Specialist:
    name = "specialist"

    def reset(self, seed):
        self.seed = seed

    def act(self, observation):
        return PolicyDecision(1)


def _state(*, predator_visible: bool):
    return np.asarray(
        [
            0.5,
            0.5,
            0.0,
            0.6 if predator_visible else 0.0,
            0.5 if predator_visible else 0.0,
            0.0,
            0.5,
            0.0,
            0.0,
            0.0,
        ],
        dtype=np.float32,
    )


def test_paraphrased_safety_instruction_selects_evasion():
    planner = InstructionSkillPlanner()
    for instruction in (
        "Prioritize safety.",
        "Avoid unnecessary exposure to the predator.",
        "Favor survival over speed.",
    ):
        decision = planner.plan(instruction, _state(predator_visible=True), ())
        assert decision.skill == Skill.EVADE_PREDATOR


def test_same_state_changes_behavior_with_instruction():
    destinations = np.asarray(
        [[0.5, 0.5], [1.0, 0.5], [0.1, 0.5]], dtype=np.float64
    )
    safe = HierarchicalPolicy(
        specialist=_Specialist(),
        destinations=destinations,
        instruction="Favor survival over speed.",
        planner_horizon=8,
    )
    fast = HierarchicalPolicy(
        specialist=_Specialist(),
        destinations=destinations,
        instruction="Reach the goal as quickly as possible.",
        planner_horizon=8,
    )
    observation = _state(predator_visible=True)
    safe.reset(7)
    fast.reset(7)
    safe_decision = safe.act(observation)
    fast_decision = fast.act(observation)
    assert safe_decision.action == 2
    assert safe_decision.metadata["skill"] == "evade_predator"
    assert fast_decision.action == 1
    assert fast_decision.metadata["skill"] == "go_to_goal"


def test_visibility_interrupts_goal_plan_before_horizon():
    destinations = np.asarray(
        [[0.5, 0.5], [1.0, 0.5], [0.1, 0.5]], dtype=np.float64
    )
    policy = HierarchicalPolicy(
        specialist=_Specialist(),
        destinations=destinations,
        instruction="Move cautiously and avoid the predator.",
        planner_horizon=16,
    )
    policy.reset(3)
    first = policy.act(_state(predator_visible=False))
    second = policy.act(_state(predator_visible=True))
    assert first.metadata["skill"] == "go_to_goal"
    assert second.metadata["skill"] == "evade_predator"
    assert second.metadata["replanned"] is True
