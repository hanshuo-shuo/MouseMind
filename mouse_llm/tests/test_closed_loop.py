from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from mouse_llm.evaluation.closed_loop import (
    LEGACY_GYM_SOURCE_INDICES,
    PolicyDecision,
    bootstrap_mean,
    build_report,
    classify_failure,
    load_verified_observation_audit,
    policy_observation,
    run_policy,
)


class _FixedPolicy:
    def __init__(self, name: str, action: int):
        self.name = name
        self.action = action

    def reset(self, seed: int) -> None:
        self.seed = seed

    def act(self, observation):
        return PolicyDecision(self.action)


class _TinyEnv:
    time_step = 0.25

    def reset(self, seed=None):
        self.x = 0.0
        self.steps = 0
        return self._observation(), {}

    def _observation(self):
        return np.asarray([self.x, 0.5, *([0.0] * 8)], dtype=np.float32)

    def goal_distance(self, observation):
        return 1.0 - float(observation[0])

    def step(self, action):
        self.steps += 1
        self.x = min(1.0, self.x + (0.5 if action == 1 else 0.0))
        success = self.x >= 1.0
        truncated = self.steps >= 3 and not success
        info = {}
        if success or truncated:
            info = {
                "captures": int(action == 2),
                "is_success": int(success),
                "survived": int(action != 2),
            }
        return self._observation(), float(success), success, truncated, info

    def close(self):
        pass


def test_policy_observation_adapter_is_explicit_and_bounded():
    observation = np.arange(15, dtype=np.float32)
    selected = policy_observation(observation, indices=LEGACY_GYM_SOURCE_INDICES)
    assert selected[[0, 1, 3, 4, 6, 7, 8, 9]].tolist() == [
        0.0,
        1.0,
        4.0,
        5.0,
        14.0,
        10.0,
        11.0,
        12.0,
    ]
    assert selected[2] == pytest.approx(2.0)
    assert selected[5] == pytest.approx(6.0 - 2 * np.pi)
    with pytest.raises(ValueError, match="cannot satisfy"):
        policy_observation(observation, indices=tuple(range(6, 16)))


def test_seeded_rollouts_and_paired_report():
    seeds = [10, 11, 12]
    random_like = run_policy(
        _TinyEnv,
        _FixedPolicy("stationary", 0),
        seeds=seeds,
        control_budget_seconds=1.0,
    )
    specialist = run_policy(
        _TinyEnv,
        _FixedPolicy("goal", 1),
        seeds=seeds,
        control_budget_seconds=1.0,
    )
    report = build_report(
        {"stationary": random_like, "goal": specialist},
        seed=42,
        reference_policy="stationary",
        metadata={"synthetic": True},
    )
    assert report["policies"]["stationary"]["success_rate"]["mean"] == 0.0
    assert report["policies"]["goal"]["success_rate"]["mean"] == 1.0
    paired = report["paired_comparisons"]["goal_minus_stationary"]
    assert paired["success_rate"]["mean"] == 1.0
    assert [row.seed for row in specialist] == seeds
    assert all(row.path_efficiency == pytest.approx(1.0) for row in specialist)
    assert report["policies"]["stationary"]["failure_taxonomy"]["counts"] == {
        "stuck_timeout": 3
    }


def test_bootstrap_is_deterministic():
    first = bootstrap_mean([0.0, 1.0, 1.0], seed=7, iterations=100)
    second = bootstrap_mean([0.0, 1.0, 1.0], seed=7, iterations=100)
    assert first == second


def test_failure_taxonomy_prioritizes_capture_timing():
    mode = classify_failure(
        success=0,
        captures=1,
        first_predator_visible_step=10,
        first_capture_step=12,
        capture_near_occlusion=1,
        capture_in_open_space=0,
        oscillation_score=0.0,
        recent_path_length=1.0,
        goal_distance_start=0.9,
        goal_distance_min=0.5,
        goal_distance_end=0.6,
    )
    assert mode == "late_predator_response"


def test_checked_in_observation_audit_is_verified():
    audit = load_verified_observation_audit(
        Path("mouse_llm/reports/observation_contract_audit.json")
    )
    assert audit["verified"] is True
    assert audit["dataset_rows"] == 118861
