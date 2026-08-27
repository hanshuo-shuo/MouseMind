from types import SimpleNamespace

import numpy as np

from mouse_llm.evaluation.closed_loop import PolicyDecision
from mouse_llm.hierarchical.counterfactual import branch_anchor, verify_anchor_replay


class _Specialist:
    name = "fixed"

    def reset(self, seed):
        self.seed = seed

    def act(self, observation):
        return PolicyDecision(1)


class _ReplayEnv:
    max_step = 20

    def __init__(self):
        self.model = SimpleNamespace(
            prey_data=SimpleNamespace(puff_count=0, prey_goal_distance=1.0)
        )

    def reset(self, seed=None):
        self.x = (seed % 3) * 0.01
        self.steps = 0
        self.model.prey_data.puff_count = 0
        self.model.prey_data.prey_goal_distance = 1.0 - self.x
        return self._observation(), {}

    def _observation(self):
        return np.asarray(
            [
                self.x,
                0.5,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                -1.0,
                0.0,
                1.0,
                0.0,
                0.0,
                1.0 - self.x,
            ],
            dtype=np.float64,
        )

    def legacy_policy_observation(self):
        return np.asarray(
            [self.x, 0.5, 0.0, 0.0, 0.0, 0.0, 1.0 - self.x, 0.0, 1.0, 0.0]
        )

    def step(self, action):
        self.steps += 1
        if action == 1:
            self.x = min(1.0, self.x + 0.1)
        self.model.prey_data.prey_goal_distance = 1.0 - self.x
        success = self.x >= 0.95
        return self._observation(), 0.0, success, False, {
            "captures": 0,
            "is_success": int(success),
        } if success else {}

    def close(self):
        pass


def _anchor():
    env = _ReplayEnv()
    observation, _ = env.reset(seed=10)
    observation, *_ = env.step(1)
    return {
        "seed": 10,
        "step_index": 1,
        "prefix_actions": [1],
        "source_policy": "test",
        "tags": ["strategic_stride"],
        "environment_observation": observation.tolist(),
        "legacy_observation": env.legacy_policy_observation().tolist(),
        "context": {"schema_version": "test"},
        "context_vector": [0.0, 1.0],
    }


def test_replay_verification_rejects_drift_and_branches_every_skill():
    anchor = _anchor()
    verification = verify_anchor_replay(_ReplayEnv, anchor)
    assert verification["verified"] is True
    drifted = dict(anchor, environment_observation=[99.0] * 15)
    assert verify_anchor_replay(_ReplayEnv, drifted)["verified"] is False
    result = branch_anchor(
        _ReplayEnv,
        anchor=anchor,
        specialist=_Specialist(),
        destinations=np.asarray([[0.0, 0.5], [1.0, 0.5], [0.5, 0.5]]),
        horizon=4,
    )
    assert set(result["branches"]) == {
        "go_to_goal",
        "evade_predator",
        "hold_position",
    }
    assert result["replay_verification"]["verified"] is True
