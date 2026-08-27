from dataclasses import replace

import numpy as np

from mouse_llm.data.planner_schema import Preference
from mouse_llm.evaluation.closed_loop import PolicyDecision
from mouse_llm.hierarchical.context import PlannerContext
from mouse_llm.hierarchical.policy import PlannerDecision, Skill
from mouse_llm.hierarchical.verified_policy import ProposeVerifyPolicy, verify_proposal


class _Verifier:
    scores = {
        Skill.GO_TO_GOAL: 0.9,
        Skill.EVADE_PREDATOR: 0.1,
        Skill.HOLD_POSITION: 0.12,
    }

    def score(self, context, skill):
        return self.scores[skill]


def _context():
    return PlannerContext(
        prey_x=0.2,
        prey_y=0.3,
        prey_direction=0.0,
        predator_visible=True,
        predator_x=0.5,
        predator_y=0.5,
        predator_direction=0.0,
        time_since_predator_seen=0,
        near_wall=False,
        near_occlusion=False,
        puffed=False,
        puff_cooled_down=True,
        goal_distance=0.7,
        recent_prey_positions=(),
        recent_predator_visibility=(),
        recent_low_level_actions=(),
        recent_high_level_skills=(),
    )


def test_verifier_overrides_risky_proposal_with_lowest_risk_skill():
    decision = verify_proposal(
        context=_context(),
        proposed_skill=Skill.GO_TO_GOAL,
        verifier=_Verifier(),
        threshold=0.5,
        preference=Preference.SURVIVAL_FIRST,
    )
    assert decision.was_overridden is True
    assert decision.executed_skill == Skill.EVADE_PREDATOR
    accepted = verify_proposal(
        context=replace(_context(), predator_visible=False),
        proposed_skill=Skill.EVADE_PREDATOR,
        verifier=_Verifier(),
        threshold=0.5,
        preference=Preference.BALANCED,
    )
    assert accepted.was_overridden is False
    balanced_override = verify_proposal(
        context=replace(_context(), predator_visible=False),
        proposed_skill=Skill.GO_TO_GOAL,
        verifier=_Verifier(),
        threshold=0.05,
        preference=Preference.BALANCED,
    )
    assert balanced_override.executed_skill == Skill.EVADE_PREDATOR


def test_propose_verify_policy_records_auditable_decision_metadata():
    class _Specialist:
        name = "specialist"

        def reset(self, seed):
            self.seed = seed

        def act(self, observation):
            return PolicyDecision(0)

    class _Planner:
        def plan_context(self, context, preference):
            return PlannerDecision(Skill.GO_TO_GOAL, "test")

    policy = ProposeVerifyPolicy(
        specialist=_Specialist(),
        destinations=np.asarray([[0.2, 0.3], [0.8, 0.8], [0.1, 0.1]]),
        planner=_Planner(),
        verifier=_Verifier(),
        risk_threshold=0.5,
        preference=Preference.SURVIVAL_FIRST,
    )
    policy.reset(7)
    observation = np.asarray(
        [0.2, 0.3, 0.0, 1.0, 0.5, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.7]
    )
    decision = policy.act(observation)
    assert decision.metadata["proposed_skill"] == "go_to_goal"
    assert decision.metadata["executed_skill"] == "evade_predator"
    assert decision.metadata["was_overridden"] is True
    assert decision.metadata["override_reason"] == "proposed_capture_risk_above_threshold"
    assert decision.metadata["replan_reason"] == "horizon"
    assert decision.metadata["planner_latency_seconds"] >= 0.0
    assert decision.metadata["verifier_latency_seconds"] >= 0.0
