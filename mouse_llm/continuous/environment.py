"""Native two-coordinate destination actions over the unchanged simulator."""

from __future__ import annotations

from typing import Any
import math

import numpy as np
import gymnasium as gym
from gymnasium import spaces

from .contracts import ACTION_SCHEMA, validate_xy

RESET_CONTRACT = "fresh_navigation_episode_reset_v1"


class ContinuousBotEvadeEnv(gym.Wrapper):
    """Expose ``[x, y]`` while retaining Cellworld's native navigation.

    The original continuous adapter accepts ``[x, y, peek]``. This contract
    fixes peek to zero; holding is a destination at the current position.
    A task-local prey navigation adapter retains the exact terminal waypoint;
    the native graph router otherwise replaces it with a cell center. Invalid
    map destinations are explicitly projected into the current connected free
    region by default. Reject mode holds instead. Failed projection also holds.
    An injected environment is useful for contract tests and must expose the
    same three-coordinate native action space.
    """

    action_schema = ACTION_SCHEMA
    reset_contract = RESET_CONTRACT

    def __init__(self, env: gym.Env | None = None, *, destination_mode: str = "project",
                 projection_epsilon: float = 1e-5, **kwargs: Any):
        if destination_mode not in {"project", "reject"}:
            raise ValueError("destination_mode must be project or reject")
        if (isinstance(projection_epsilon, bool) or not math.isfinite(projection_epsilon)
                or not 0.0 < projection_epsilon <= 0.01):
            raise ValueError("projection_epsilon must be finite and in (0, 0.01]")
        self.destination_mode = destination_mode
        self.projection_epsilon = float(projection_epsilon)
        if env is not None and kwargs:
            raise ValueError("Pass an existing environment or constructor settings, not both")
        if env is None:
            from mouse_llm.envs.mice import BotEvadeEnv, custom_reward

            defaults = {
                "world_name": "21_05", "use_lppos": False, "use_predator": True,
                "max_step": 300, "time_step": 0.25, "frame_stack_k": 1,
                "reward_function": custom_reward,
                "action_type": BotEvadeEnv.ActionType.CONTINUOUS,
            }
            defaults.update(kwargs)
            if defaults["action_type"] != BotEvadeEnv.ActionType.CONTINUOUS:
                raise ValueError("XY destinations require the native continuous action type")
            if defaults["frame_stack_k"] != 1:
                raise ValueError("Continuous policy observations require frame_stack_k=1")
            env = BotEvadeEnv(**defaults)
        if getattr(env.action_space, "shape", None) != (3,):
            raise ValueError("The wrapped environment must accept native [x, y, peek] actions")
        if getattr(env, "frame_stack_k", 1) != 1:
            raise ValueError("Continuous policy observations require frame_stack_k=1")
        super().__init__(env)
        self.action_space = spaces.Box(0.0, 1.0, shape=(2,), dtype=np.float32)
        self.exact_navigation = None
        if hasattr(env, "model"):
            from .navigation import install_exact_xy_navigation

            self.exact_navigation = install_exact_xy_navigation(
                env.model, destination_mode=destination_mode, projection_epsilon=projection_epsilon)
        self.navigation_contract = (
            self.exact_navigation.contract if self.exact_navigation is not None
            else "injected_environment_unverified_navigation"
        )

    def reset(self, *, seed=None, options=None):
        model = getattr(self.env, "model", None)
        if model is not None:
            # Native NavigationAgent.reset leaves queued destinations, timing,
            # and dynamics from the previous episode. Restore constructor-like
            # transient state before the native seeded task reset runs.
            model.time = 0.0
            model.paused = False
            for name in ("prey", "predator"):
                agent = getattr(model, name, None)
                if agent is None:
                    continue
                for attribute, value in (
                    ("new_destination", None), ("destination", None),
                    ("destination_wait", 0), ("navigation_plan_update_wait", 0),
                    ("active_navigation", True),
                ):
                    if hasattr(agent, attribute):
                        setattr(agent, attribute, value)
                if hasattr(agent, "path"):
                    agent.path = []
                if hasattr(agent, "last_destination_time"):
                    agent.last_destination_time = 0
                agent.stop_navigation()
            prey_data = getattr(model, "prey_data", None)
            if prey_data is not None and callable(getattr(prey_data, "reset", None)):
                prey_data.reset()
            if self.exact_navigation is not None:
                self.exact_navigation.last_invalid_reason = None
        observation, info = self.env.reset(seed=seed, options=options)
        info = dict(info)
        info.update({"navigation_contract": self.navigation_contract,
                     "reset_contract": RESET_CONTRACT,
                     "destination_mode": self.destination_mode,
                     "projection_epsilon": self.projection_epsilon})
        return observation, info

    def step(self, action):
        x, y = validate_xy(action)
        destination = (x, y)
        resolution = {
            "requested_xy": [x, y], "executed_xy": [x, y],
            "requested_invalid_destination": False, "original_invalid_reason": None,
            "was_projected": False, "projection_distance": 0.0,
            "projection_attempted": False,
            "invalid_destination": False, "invalid_destination_reason": None,
        }
        if self.exact_navigation is not None:
            prey = self.env.model.prey
            resolution = self.exact_navigation.resolve_destination(prey.state.location, destination)
            destination = tuple(resolution["executed_xy"])
            if resolution["invalid_destination"]:
                # Clear queued and current navigation so an invalid request
                # cannot leave the previous action running for another window.
                prey.new_destination = None
                prey.destination = None
                prey.path = []
                prey.stop_navigation()
                destination = tuple(prey.state.location)
        observation, reward, terminated, truncated, info = self.env.step(
            np.asarray([*destination, 0.0], dtype=np.float64)
        )
        info = dict(info)
        info.update({"navigation_contract": self.navigation_contract,
                     "reset_contract": RESET_CONTRACT,
                     "destination_mode": self.destination_mode,
                     "projection_epsilon": self.projection_epsilon,
                     **resolution})
        return observation, reward, terminated, truncated, info

    def legacy_policy_observation(self) -> np.ndarray:
        return np.asarray(self.env.legacy_policy_observation(), dtype=np.float32).copy()
