"""Verified exact-state branching with per-anchor duplicate-skill caching."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Any

import numpy as np

from mouse_llm.hierarchical.counterfactual import (
    BranchOutcome,
    branch_skill,
    verify_anchor_replay,
)
from mouse_llm.hierarchical.policy import Skill

from .reward import verified_reward


class ReplayVerificationError(RuntimeError):
    """No exact replay was obtained from fresh environment instances."""


class _VerifiedBranchEnv:
    """Check the replayed branch environment at the exact anchor boundary."""

    def __init__(self, env: Any, anchor: Mapping[str, Any], tolerance: float):
        self._env = env
        self._prefix_length = len(anchor["prefix_actions"])
        self._steps = 0
        self._expected_environment = np.asarray(anchor["environment_observation"], dtype=np.float64)
        self._expected_legacy = np.asarray(anchor["legacy_observation"], dtype=np.float64)
        self._tolerance = tolerance

    def __getattr__(self, name: str) -> Any:
        return getattr(self._env, name)

    def _check(self, observation: Any) -> None:
        environment_error = float(np.max(np.abs(np.asarray(observation, dtype=np.float64) - self._expected_environment)))
        legacy_error = float(np.max(np.abs(np.asarray(self._env.legacy_policy_observation(), dtype=np.float64) - self._expected_legacy)))
        if environment_error > self._tolerance or legacy_error > self._tolerance:
            raise ReplayVerificationError(
                f"Branch prefix state mismatch: environment={environment_error}, "
                f"legacy={legacy_error}, tolerance={self._tolerance}"
            )

    def reset(self, *args: Any, **kwargs: Any) -> Any:
        result = self._env.reset(*args, **kwargs)
        self._steps = 0
        if self._prefix_length == 0:
            self._check(result[0])
        return result

    def step(self, action: int) -> Any:
        result = self._env.step(action)
        self._steps += 1
        if self._steps == self._prefix_length:
            self._check(result[0])
        return result


class RolloutVerifier:
    def __init__(
        self,
        env_factory: Callable[[], Any],
        specialist: Any,
        destinations: np.ndarray,
        *,
        horizon: int = 8,
        evade_distance: float = 0.35,
        replay_tolerance: float = 1e-7,
        replay_attempts: int = 3,
    ):
        if horizon <= 0:
            raise ValueError("horizon must be positive")
        if replay_attempts < 1:
            raise ValueError("replay_attempts must be positive")
        self.env_factory = env_factory
        self.specialist = specialist
        self.destinations = destinations
        self.horizon = horizon
        self.evade_distance = evade_distance
        self.replay_tolerance = replay_tolerance
        self.replay_attempts = replay_attempts
        self.last_replay_attempts = 0
        self._anchor_key: tuple[int, tuple[int, ...]] | None = None
        self._outcomes: dict[Skill, BranchOutcome] = {}

    def _prepare(self, anchor: Mapping[str, Any]) -> None:
        key = (int(anchor["seed"]), tuple(int(a) for a in anchor["prefix_actions"]))
        if key == self._anchor_key:
            return
        # A failed verification must never reuse outcomes from a previous anchor.
        self._anchor_key = None
        self._outcomes = {}
        errors = []
        for attempt in range(1, self.replay_attempts + 1):
            verification = verify_anchor_replay(
                self.env_factory, anchor, tolerance=self.replay_tolerance
            )
            self.last_replay_attempts = attempt
            if verification["verified"]:
                break
            errors.append(verification)
        else:
            raise ReplayVerificationError(
                f"Exact-state replay failed on {self.replay_attempts} fresh environments: "
                f"{errors}"
            )
        self._anchor_key = key

    def verify_anchor(self, anchor: Mapping[str, Any]) -> None:
        self._prepare(anchor)

    def evaluate_skill(
        self, anchor: Mapping[str, Any], skill: Skill | str, horizon: int | None = None
    ) -> BranchOutcome:
        if horizon is not None and horizon != self.horizon:
            raise ValueError("Construct a verifier with the requested horizon")
        self._prepare(anchor)
        skill = Skill(skill)
        if skill not in self._outcomes:
            errors = []
            for _ in range(self.replay_attempts):
                try:
                    outcome = branch_skill(
                        lambda: _VerifiedBranchEnv(
                            self.env_factory(), anchor, self.replay_tolerance
                        ),
                        anchor=anchor,
                        skill=skill,
                        specialist=self.specialist,
                        destinations=self.destinations,
                        horizon=self.horizon,
                        evade_distance=self.evade_distance,
                    )
                    self._outcomes[skill] = outcome
                    break
                except ReplayVerificationError as exc:
                    errors.append(str(exc))
            else:
                raise ReplayVerificationError(
                    f"Branch {skill.value} failed exact replay on "
                    f"{self.replay_attempts} fresh environments: {errors}"
                )
        return self._outcomes[skill]

    def evaluate_reward(
        self, anchor: Mapping[str, Any], skill: Skill | str, preference: str
    ) -> float:
        return verified_reward(self.evaluate_skill(anchor, skill), preference)

    def evaluate_group(
        self, anchor: Mapping[str, Any], skills: Sequence[Skill | str], preference: str
    ) -> tuple[list[BranchOutcome], list[float]]:
        outcomes = [self.evaluate_skill(anchor, skill) for skill in skills]
        return outcomes, [verified_reward(outcome, preference) for outcome in outcomes]
