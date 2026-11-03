"""Only the GO executor changes in the legacy-planner/continuous-BC hybrid.

All observations and decisions below are synthetic. No simulator, tokenizer,
pretrained model, private trajectory, or network service is used.
"""

import copy
import json

import numpy as np
import pytest

from mouse_llm.continuous.contracts import ContinuousDecision
from mouse_llm.continuous.legacy_hybrid import LegacyPlannerContinuousBCPolicy
from mouse_llm.data.planner_schema import Preference, skill_target
from mouse_llm.evaluation.closed_loop import (
    LEGACY_GYM_SOURCE_INDICES,
    PolicyDecision,
    policy_observation,
)
from mouse_llm.hierarchical.policy import GeometricSkillController, PlannerDecision, Skill
from mouse_llm.hierarchical.verified_policy import ProposeVerifyPolicy


CATALOG = np.asarray([
    [0.1, 0.1], [0.9, 0.1], [0.1, 0.9], [0.9, 0.9],
    [0.5, 0.5], [0.2, 0.5], [0.8, 0.5], [0.6, 0.7],
], dtype=np.float64)


def observation(*, x=0.5, y=0.5, predator=None, near_wall=False, near_occlusion=False):
    values = np.zeros(15, dtype=np.float64)
    values[:3] = x, y, 6.1
    values[7], values[8] = int(near_wall), int(near_occlusion)
    values[11], values[14] = 1.0, 0.7
    if predator is not None:
        values[3] = 1.0
        values[4:7] = *predator, 5.8
    return values


class RecordingPlanner:
    def __init__(self, skills=(Skill.GO_TO_GOAL,), *, parse_valid=True, raw_response=None):
        self.skills = tuple(skills)
        self.parse_valid = parse_valid
        self.raw_response = raw_response
        self.resets = []
        self.calls = []
        self.last_valid = None
        self.last_response = None

    def reset(self, seed):
        self.resets.append(seed)
        self.calls.clear()
        self.last_valid = None
        self.last_response = None

    def plan_context(self, context, preference):
        index = min(len(self.calls), len(self.skills) - 1)
        skill = self.skills[index]
        self.calls.append((copy.deepcopy(context.payload()), preference))
        self.last_valid = self.parse_valid
        self.last_response = self.raw_response if self.raw_response is not None else skill_target(skill)
        return PlannerDecision(skill, "synthetic_recording_planner")


class DiscreteSpecialist:
    def __init__(self, *, action=2, valid=True):
        self.action, self.valid = action, valid
        self.resets, self.calls = [], []

    def reset(self, seed):
        self.resets.append(seed)
        self.calls.clear()

    def act(self, state):
        self.calls.append(np.asarray(state).copy())
        return PolicyDecision(self.action, valid=self.valid, raw_response="synthetic specialist response")


class ContinuousSpecialist:
    def __init__(self, xy=tuple(CATALOG[2]), *, valid=True):
        self.xy, self.valid = tuple(xy), valid
        self.resets, self.calls = [], []

    def reset(self, seed):
        self.resets.append(seed)
        self.calls.clear()

    def act(self, state):
        self.calls.append(np.asarray(state).copy())
        return ContinuousDecision(self.xy, valid=self.valid,
                                  metadata={"label_mode": "commanded_destination"})


def hybrid(specialist=None, planner=None, *, destinations=CATALOG, horizon=4,
           preference=Preference.SURVIVAL_FIRST, temporal_window=3):
    return LegacyPlannerContinuousBCPolicy(
        specialist=specialist or ContinuousSpecialist(),
        planner=planner or RecordingPlanner(),
        destinations=destinations,
        preference=preference,
        planner_horizon=horizon,
        temporal_window=temporal_window,
    )


def test_catalog_equivalent_go_preserves_legacy_context_actions_and_emergency_cadence():
    skills = (Skill.GO_TO_GOAL, Skill.EVADE_PREDATOR, Skill.HOLD_POSITION, Skill.GO_TO_GOAL)
    old_planner, new_planner = RecordingPlanner(skills), RecordingPlanner(skills)
    old_specialist, new_specialist = DiscreteSpecialist(), ContinuousSpecialist()
    old = ProposeVerifyPolicy(
        specialist=old_specialist, destinations=CATALOG, planner=old_planner,
        preference=Preference.SURVIVAL_FIRST, planner_horizon=4, temporal_window=3,
    )
    new = hybrid(new_specialist, new_planner)
    old.reset(42)
    new.reset(42)
    snapshots = [
        observation(x=0.42), observation(x=0.43),
        observation(x=0.44, predator=(0.8, 0.5)),  # Interrupt GO before horizon.
        observation(x=0.45), observation(x=0.46, predator=(0.8, 0.5)),
        observation(x=0.47), observation(x=0.48, near_occlusion=True),
        observation(x=0.49, predator=(0.8, 0.5)),  # HOLD is not interrupted.
        observation(x=0.50), observation(x=0.51), observation(x=0.52),
    ]
    replan_steps = []
    shared_metadata = (
        "skill", "proposed_skill", "executed_skill", "risk_score", "was_overridden",
        "override_reason", "replanned", "replan_reason",
    )
    for step, state in enumerate(snapshots):
        before = state.copy()
        old_decision, new_decision = old.act(state), new.act(state)
        np.testing.assert_array_equal(state, before)
        assert new_decision.action == tuple(CATALOG[old_decision.action])
        assert new_decision.valid == old_decision.valid
        assert new.last_context.payload() == old.last_context.payload()
        assert old_planner.calls == new_planner.calls
        for key in shared_metadata:
            assert new_decision.metadata[key] == old_decision.metadata[key]
        if new_decision.metadata["replanned"]:
            replan_steps.append(step)
    assert replan_steps == [0, 2, 6, 10]
    assert old_specialist.resets == new_specialist.resets == [42]
    assert len(old_specialist.calls) == len(new_specialist.calls)
    for left, right in zip(old_specialist.calls, new_specialist.calls):
        np.testing.assert_array_equal(left, right)


@pytest.mark.parametrize("preference,expected", [
    (Preference.SURVIVAL_FIRST, [True, True]),
    (Preference.BALANCED, [True, True]),
    (Preference.GOAL_FIRST, [True, False]),
    (Preference.HOLD, [True, False]),
])
def test_visibility_interrupt_uses_exact_legacy_preference_gate(preference, expected):
    policy = hybrid(preference=preference, horizon=8)
    policy.reset(1)
    decisions = [policy.act(observation()), policy.act(observation(predator=(0.8, 0.5)))]
    assert [decision.metadata["replanned"] for decision in decisions] == expected
    if expected[1]:
        assert decisions[1].metadata["replan_reason"] == "emergency_predator_visibility_interrupt"


def test_off_catalog_go_is_executed_exactly_and_only_history_uses_nearest_id():
    xy = (0.241234567, 0.738912345)
    specialist, planner = ContinuousSpecialist(xy), RecordingPlanner()
    policy = hybrid(specialist, planner, horizon=1)
    policy.reset(7)
    first = policy.act(observation())
    proxy = int(np.linalg.norm(CATALOG - np.asarray(xy), axis=1).argmin())
    assert first.action == xy
    assert first.action != tuple(CATALOG[proxy])
    assert first.metadata["history_proxy_action"] == proxy
    assert first.metadata["history_action_proxy"] == "nearest_catalog_id_for_history_only"
    assert first.metadata["go_execution_source"] == "continuous_bc_exact_xy"
    assert first.metadata["legacy_context_schema"] == "mousemind_planner_context_v1"
    policy.act(observation(x=0.53))
    history = planner.calls[-1][0]["history"]
    assert history["recent_low_level_actions"] == [proxy]
    assert "recent_xy_actions" not in history
    assert all(isinstance(value, int) for value in history["recent_low_level_actions"])
    assert planner.calls[-1][0]["schema_version"] == "mousemind_planner_context_v1"


def test_nearest_history_proxy_tie_uses_first_catalog_entry_without_quantizing_output():
    catalog = np.asarray([[0.25, 0.5], [0.75, 0.5]], dtype=np.float64)
    policy = hybrid(ContinuousSpecialist((0.5, 0.5)), destinations=catalog, horizon=1)
    policy.reset(4)
    decision = policy.act(observation())
    assert decision.action == (0.5, 0.5)
    assert decision.metadata["history_proxy_action"] == 0


@pytest.mark.parametrize("skill,state", [
    (Skill.HOLD_POSITION, observation(x=0.51, y=0.52)),
    (Skill.EVADE_PREDATOR, observation(predator=(0.8, 0.5))),
    (Skill.EVADE_PREDATOR, observation()),
    (Skill.EVADE_PREDATOR, observation(predator=(0.5, 0.5))),
])
def test_non_goal_actions_remain_legacy_catalog_geometry(skill, state):
    specialist = ContinuousSpecialist((0.321, 0.654))
    policy = hybrid(specialist, RecordingPlanner((skill,)), horizon=1)
    policy.reset(11)
    expected_controller = GeometricSkillController(DiscreteSpecialist(), CATALOG)
    legacy = policy_observation(state, indices=LEGACY_GYM_SOURCE_INDICES)
    expected = expected_controller.act(skill, legacy)
    decision = policy.act(state)
    assert decision.action == tuple(CATALOG[expected.action])
    assert decision.metadata["history_proxy_action"] == expected.action
    assert decision.metadata["non_goal_execution_source"] == "legacy_geometric_catalog"
    assert specialist.calls == []
    if skill == Skill.HOLD_POSITION:
        assert decision.action != tuple(state[:2])  # Preserve original snapped HOLD.


def test_hidden_predator_keeps_legacy_origin_semantics_instead_of_new_threat_memory():
    policy = hybrid(planner=RecordingPlanner((Skill.EVADE_PREDATOR,)), horizon=1)
    policy.reset(1)
    visible = policy.act(observation(predator=(0.8, 0.5)))
    hidden = policy.act(observation())
    assert visible.action[0] < 0.5
    assert hidden.action[0] > 0.5 and hidden.action[1] > 0.5
    payload = policy.last_context.payload()
    assert payload["current"]["predator_x"] == 0
    assert payload["current"]["predator_y"] == 0
    assert "last_seen_predator_x" not in payload["current"]


def test_legacy_evade_clipping_stays_at_point_zero_two_and_point_nine_eight():
    catalog = np.asarray([[0.98, 0.98], [1.0, 1.0], [0.02, 0.02], [0.0, 0.0]])
    policy = hybrid(planner=RecordingPlanner((Skill.EVADE_PREDATOR,)), destinations=catalog, horizon=1)
    policy.reset(2)
    assert policy.act(observation(x=0.9, y=0.9, predator=(0.1, 0.1))).action == (0.98, 0.98)
    assert policy.act(observation(x=0.1, y=0.1, predator=(0.9, 0.9))).action == (0.02, 0.02)


@pytest.mark.parametrize("planner_valid,specialist_valid,raw", [
    (False, True, "invalid planner text preserved verbatim"),
    (True, False, '{"skill":"go_to_goal"}'),
])
def test_planner_parse_validity_is_not_substituted_for_specialist_action_validity(planner_valid, specialist_valid, raw):
    planner = RecordingPlanner(parse_valid=planner_valid, raw_response=raw)
    specialist = ContinuousSpecialist(valid=specialist_valid)
    policy = hybrid(specialist, planner)
    policy.reset(3)
    decision = policy.act(observation())
    assert decision.valid is specialist_valid
    assert decision.metadata["planner_last_valid"] is planner_valid
    assert decision.metadata["planner_raw_response"] == raw
    assert decision.metadata["skill"] == Skill.GO_TO_GOAL.value


def test_switching_away_from_go_does_not_reuse_its_xy_or_invalid_flag():
    specialist = ContinuousSpecialist((0.321, 0.654), valid=False)
    planner = RecordingPlanner((Skill.GO_TO_GOAL, Skill.HOLD_POSITION, Skill.EVADE_PREDATOR))
    policy = hybrid(specialist, planner, horizon=1)
    policy.reset(3)
    go = policy.act(observation())
    hold = policy.act(observation())
    evade = policy.act(observation(predator=(0.8, 0.5)))
    assert go.valid is False and go.action == specialist.xy
    assert hold.valid is True and hold.action == (0.5, 0.5)
    assert evade.valid is True and evade.action == (0.2, 0.5)
    assert len(specialist.calls) == 1


def test_reset_resets_planner_specialist_and_context_history_before_next_episode():
    specialist, planner = ContinuousSpecialist(), RecordingPlanner()
    policy = hybrid(specialist, planner, horizon=8)
    state = observation()
    policy.reset(10)
    first = policy.act(state)
    first_context = copy.deepcopy(policy.last_context.payload())
    policy.act(observation(predator=(0.8, 0.5)))
    policy.reset(20)
    assert policy.last_context is None
    second = policy.act(state)
    assert specialist.resets == planner.resets == [10, 20]
    assert len(specialist.calls) == len(planner.calls) == 1
    assert first.action == second.action
    assert second.metadata["replanned"] is True
    assert policy.last_context.payload() == first_context
    assert policy.last_context.payload()["history"]["recent_low_level_actions"] == []


def test_only_public_observation_enters_planner_and_specialist():
    planner, specialist = RecordingPlanner(), ContinuousSpecialist()
    policy = hybrid(specialist, planner, horizon=1)
    state = observation(near_wall=True, near_occlusion=True)
    policy.reset(42)
    first = policy.act(state.copy())
    first_payload = copy.deepcopy(planner.calls[-1][0])
    np.testing.assert_array_equal(specialist.calls[-1][3:6], [0.0, 0.0, 0.0])
    assert specialist.calls[-1].shape == (10,)
    assert first_payload["current"]["predator_visible"] is False
    assert first_payload["current"]["predator_x"] == first_payload["current"]["predator_y"] == 0
    serialized = json.dumps(first_payload)
    for forbidden in ("replay_fingerprint", "simulator_state", "hidden_predator", "privileged_info"):
        assert forbidden not in serialized
    # A fresh episode with the same supplied public state has identical inputs;
    # no environment or evaluator-info object is part of the policy API.
    policy.reset(42)
    assert policy.act(state.copy()).action == first.action
    assert planner.calls[-1][0] == first_payload
