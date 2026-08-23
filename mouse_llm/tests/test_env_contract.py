from __future__ import annotations

import os
import math
from pathlib import Path

import pytest


def test_botevade_reset_and_step_contract():
    if not os.environ.get("CELLWORLD_CACHE"):
        pytest.skip("CELLWORLD_CACHE is required for the offline environment smoke test")
    required_path = (
        Path(os.environ["CELLWORLD_CACHE"])
        / "paths/hexagonal.21_05.astar.robot"
    )
    if not required_path.is_file():
        pytest.skip(f"offline cache is missing {required_path}")
    pytest.importorskip("gymnasium")
    pytest.importorskip("cellworld")
    from mouse_llm.envs.mice import BotEvadeEnv

    env = BotEvadeEnv(
        world_name="21_05",
        use_lppos=False,
        use_predator=True,
        frame_stack_k=1,
        max_step=2,
    )
    try:
        observation, info = env.reset(seed=7)
        first_predator_location = tuple(env.model.predator.state.location)
        repeated_observation, _ = env.reset(seed=7)
        assert tuple(env.model.predator.state.location) == first_predator_location
        assert (observation == repeated_observation).all()
        legacy = env.legacy_policy_observation()
        assert legacy.shape == (10,)
        assert legacy[0] == pytest.approx(env.model.prey.state.location[0])
        assert legacy[2] == pytest.approx(
            math.radians(env.model.prey.state.direction)
        )
        assert env.observation_space.contains(observation)
        assert env.action_space.n == 295
        next_observation, reward, terminated, truncated, info = env.step(0)
        assert env.observation_space.contains(next_observation)
        assert isinstance(float(reward), float)
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)
        next_observation, reward, terminated, truncated, info = env.step(0)
        assert terminated or truncated
        assert {"captures", "is_success", "survived", "termination_reason"} <= info.keys()
        assert info["survived"] == int(info["captures"] == 0)
        assert info["is_success"] == int(env.model.prey_data.goal_achieved)
    finally:
        env.close()
