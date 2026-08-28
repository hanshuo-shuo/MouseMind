from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from mouse_llm.evaluation.audit_transfer_compatibility import coordinate_sha256
from mouse_llm.evaluation.contracts import (
    load_transfer_contract,
    transfer_contract_seeds,
)
from mouse_llm.evaluation.evaluate_policy import load_action_catalog
from mouse_llm.evaluation.transfer_benchmark import (
    GoalCoordinatePolicy,
    RuleContextPlanner,
    _PlannerIsolationSpecialist,
    source_manifest_sha256,
)
from mouse_llm.data.planner_schema import Preference
from mouse_llm.hierarchical.policy import GeometricSkillController, Skill
from mouse_llm.hierarchical.verified_policy import ProposeVerifyPolicy


class _Specialist:
    name = "specialist"

    def reset(self, seed):
        self.seed = seed

    def act(self, observation):
        raise AssertionError("goal destination should bypass the frozen specialist")


def test_transfer_seed_sets_are_frozen_disjoint_and_fail_closed():
    contract = load_transfer_contract()
    pools = {
        name: set(transfer_contract_seeds(name))
        for name in contract["seed_pools"]
    }
    assert not pools["compatibility_audit"] & pools["development"]
    assert not pools["compatibility_audit"] & pools["final_test"]
    assert not pools["development"] & pools["final_test"]
    assert contract["observation_contract"]["literal_low_level_transfer"] == {
        "compatible": False,
        "reason": "The frozen BotEvade 10D specialist observes goal distance but not the active goal coordinates required by Oasis.",
    }


def test_committed_action_catalog_matches_transfer_coordinate_hash():
    contract = load_transfer_contract()
    catalog = load_action_catalog(
        Path("mouse_llm/envs/mice/assets/action_catalog_21_05.json")
    )
    assert len(catalog) == contract["action_contract"]["action_count"]
    assert coordinate_sha256(catalog) == contract["action_contract"][
        "coordinate_sha256"
    ]
    frozen_goals = np.asarray(
        [*contract["target"]["goal_locations"], [0.05, 0.5]],
        dtype=np.float64,
    )
    frozen_errors = [
        float(np.linalg.norm(catalog - goal, axis=1).min())
        for goal in frozen_goals
    ]
    assert max(frozen_errors) <= contract["target"]["goal_threshold"]
    default_goal = np.asarray(
        contract["target"]["discrete_goal_projection"]["default_location"]
    )
    default_error = float(np.linalg.norm(catalog - default_goal, axis=1).min())
    assert default_error > contract["target"]["goal_threshold"]
    assert default_error == contract["target"]["discrete_goal_projection"][
        "projection_distance"
    ]


def test_goal_conditioned_controller_uses_active_target_without_retraining():
    destinations = np.asarray([[0.0, 0.0], [0.7, 0.2], [1.0, 1.0]])
    controller = GeometricSkillController(_Specialist(), destinations)
    decision = controller.act(
        Skill.GO_TO_GOAL,
        np.zeros(10, dtype=np.float32),
        goal_destination=np.asarray([0.69, 0.21]),
    )
    assert decision.action == 1


def test_transfer_contract_is_canonical_json():
    path = Path("mouse_llm/evaluation/contracts/cross_task_transfer_v1.json")
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["contract_name"] == "mousemind_cross_task_transfer_v1"
    assert payload["selection"]["target_adaptation_allowed"] is False


def test_transfer_source_manifest_is_stable_and_complete():
    digest = source_manifest_sha256()
    assert len(digest) == 64
    assert digest == source_manifest_sha256(Path.cwd())


def _transfer_observation(*, predator_visible: bool = False):
    values = np.zeros(18, dtype=np.float32)
    values[0:3] = [0.05, 0.5, 0.0]
    values[3] = float(predator_visible)
    if predator_visible:
        values[4:7] = [0.2, 0.5, 0.0]
    values[11] = 1.0
    values[14] = 0.7
    values[15:18] = [0.7, 0.2, 4.0]
    return values


def test_aligned_policy_freezes_planner_but_uses_target_goal_interface():
    destinations = np.asarray([[0.0, 0.0], [0.7, 0.2], [1.0, 1.0]])
    policy = ProposeVerifyPolicy(
        specialist=_PlannerIsolationSpecialist(),
        destinations=destinations,
        planner=RuleContextPlanner(),
        preference=Preference.SURVIVAL_FIRST,
        observation_mode="transfer",
        goal_destination_indices=(15, 16),
        name="aligned-rule",
    )
    policy.reset(9)
    decision = policy.act(_transfer_observation())
    assert decision.action == 1
    assert decision.metadata["skill"] == "go_to_goal"


def test_goal_only_policy_declares_transfer_observation_mode():
    destinations = np.asarray([[0.0, 0.0], [0.7, 0.2], [1.0, 1.0]])
    policy = GoalCoordinatePolicy(destinations)
    policy.reset(1)
    assert policy.observation_mode == "transfer"
    assert policy.act(_transfer_observation()).action == 1
