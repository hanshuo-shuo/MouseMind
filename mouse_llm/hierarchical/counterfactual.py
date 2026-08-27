"""Exact seeded replay and short-horizon counterfactual skill branching."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, asdict
from typing import Any

import numpy as np

from mouse_llm.hierarchical.policy import GeometricSkillController, Skill


COUNTERFACTUAL_SCHEMA_VERSION = "mousemind_counterfactual_branch_v1"


def _legacy_observation(env: Any, observation: Sequence[float]) -> np.ndarray:
    return np.asarray(
        env.legacy_policy_observation()
        if hasattr(env, "legacy_policy_observation")
        else observation,
        dtype=np.float64,
    ).reshape(-1)


def _capture_count(env: Any, info: Mapping[str, Any] | None = None) -> int:
    if info and "captures" in info:
        return int(info["captures"])
    try:
        return int(env.model.prey_data.puff_count)
    except AttributeError:
        try:
            return int(env.model.puff_count)
        except AttributeError:
            return 0


def _goal_distance(env: Any, observation: Sequence[float]) -> float:
    values = np.asarray(observation).reshape(-1)
    if len(values) >= 15:
        return float(values[14])
    if len(values) >= 7:
        return float(values[6])
    try:
        return float(env.model.prey_data.prey_goal_distance)
    except AttributeError:
        return 0.0


@dataclass(frozen=True)
class BranchOutcome:
    skill: str
    horizon: int
    steps_executed: int
    capture_within_h: int
    captures: int
    goal_distance_change: float
    predator_exposure_steps: int
    path_length: float
    near_occlusion_exposure: int
    terminal_goal_event: int


def replay_prefix(
    env: Any,
    *,
    seed: int,
    prefix_actions: Sequence[int],
) -> tuple[np.ndarray, dict[str, Any], bool]:
    observation, _ = env.reset(seed=int(seed))
    info: dict[str, Any] = {}
    done = False
    for action in prefix_actions:
        observation, _, terminated, truncated, info = env.step(int(action))
        done = bool(terminated or truncated)
        if done:
            break
    return np.asarray(observation).copy(), info, done


def verify_anchor_replay(
    env_factory: Callable[[], Any],
    anchor: Mapping[str, Any],
    *,
    tolerance: float = 1e-7,
) -> dict[str, float | bool]:
    env = env_factory()
    try:
        observation, _, done = replay_prefix(
            env,
            seed=int(anchor["seed"]),
            prefix_actions=[int(value) for value in anchor["prefix_actions"]],
        )
        expected_environment = np.asarray(
            anchor["environment_observation"], dtype=np.float64
        )
        expected_legacy = np.asarray(anchor["legacy_observation"], dtype=np.float64)
        actual_legacy = _legacy_observation(env, observation)
        environment_error = float(
            np.max(np.abs(np.asarray(observation) - expected_environment))
        )
        legacy_error = float(np.max(np.abs(actual_legacy - expected_legacy)))
        verified = bool(
            not done
            and environment_error <= tolerance
            and legacy_error <= tolerance
        )
        return {
            "verified": verified,
            "environment_max_abs_error": environment_error,
            "legacy_max_abs_error": legacy_error,
            "tolerance": tolerance,
        }
    finally:
        env.close()


def branch_skill(
    env_factory: Callable[[], Any],
    *,
    anchor: Mapping[str, Any],
    skill: Skill,
    specialist: Any,
    destinations: np.ndarray,
    horizon: int,
    evade_distance: float,
) -> BranchOutcome:
    if horizon <= 0:
        raise ValueError("Counterfactual horizon must be positive")
    env = env_factory()
    try:
        observation, info, done = replay_prefix(
            env,
            seed=int(anchor["seed"]),
            prefix_actions=[int(value) for value in anchor["prefix_actions"]],
        )
        if done:
            raise ValueError("Anchor prefix reaches a terminal state")
        specialist.reset(int(anchor["seed"]))
        controller = GeometricSkillController(
            specialist,
            destinations,
            evade_distance=evade_distance,
        )
        start_goal_distance = _goal_distance(env, observation)
        start_captures = _capture_count(env, info)
        previous_position = np.asarray(observation, dtype=np.float64)[:2]
        path_length = 0.0
        exposure_steps = 0
        near_occlusion_exposure = 0
        terminal_goal = 0
        steps_executed = 0
        capture_observed = False
        for _ in range(horizon):
            legacy = _legacy_observation(env, observation)
            decision = controller.act(skill, legacy)
            observation, _, terminated, truncated, info = env.step(decision.action)
            values = np.asarray(observation, dtype=np.float64).reshape(-1)
            position = values[:2]
            path_length += float(np.linalg.norm(position - previous_position))
            previous_position = position
            if len(values) >= 15:
                visible = bool(values[3])
                exposure_steps += int(visible)
                near_occlusion_exposure += int(visible and bool(values[8]))
                capture_observed |= bool(values[10])
            steps_executed += 1
            if terminated or truncated:
                terminal_goal = int(bool(info.get("is_success", False)))
                break
        captures = max(_capture_count(env, info) - start_captures, 0)
        end_goal_distance = _goal_distance(env, observation)
        return BranchOutcome(
            skill=skill.value,
            horizon=horizon,
            steps_executed=steps_executed,
            capture_within_h=int(capture_observed or captures > 0),
            captures=captures,
            goal_distance_change=float(start_goal_distance - end_goal_distance),
            predator_exposure_steps=exposure_steps,
            path_length=path_length,
            near_occlusion_exposure=near_occlusion_exposure,
            terminal_goal_event=terminal_goal,
        )
    finally:
        env.close()


def branch_anchor(
    env_factory: Callable[[], Any],
    *,
    anchor: Mapping[str, Any],
    specialist: Any,
    destinations: np.ndarray,
    horizon: int = 8,
    evade_distance: float = 0.35,
    replay_tolerance: float = 1e-7,
) -> dict[str, Any]:
    verification = verify_anchor_replay(
        env_factory, anchor, tolerance=replay_tolerance
    )
    if not verification["verified"]:
        raise RuntimeError(
            "Deterministic replay verification failed; refusing approximate "
            f"counterfactual labels: {verification}"
        )
    branches = {
        skill.value: asdict(
            branch_skill(
                env_factory,
                anchor=anchor,
                skill=skill,
                specialist=specialist,
                destinations=destinations,
                horizon=horizon,
                evade_distance=evade_distance,
            )
        )
        for skill in Skill
    }
    anchor_key = json.dumps(
        {
            "seed": int(anchor["seed"]),
            "step_index": int(anchor["step_index"]),
            "prefix_actions": list(anchor["prefix_actions"]),
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    return {
        "schema_version": COUNTERFACTUAL_SCHEMA_VERSION,
        "anchor_id": hashlib.sha256(anchor_key.encode("utf-8")).hexdigest()[:20],
        "seed": int(anchor["seed"]),
        "step_index": int(anchor["step_index"]),
        "source_policy": anchor.get("source_policy", "unknown"),
        "tags": sorted(set(anchor.get("tags", []))),
        "context": anchor["context"],
        "context_vector": anchor["context_vector"],
        "replay_verification": verification,
        "horizon": horizon,
        "branches": branches,
    }
