"""Observable temporal state for the continuous-control planner.

XY action history and last *observed* predator locations have their own schema;
continuous coordinates are never encoded as historical discrete action IDs.
"""

from __future__ import annotations

import copy
import json
from collections import deque
from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np

from .contracts import ACTION_SCHEMA, validate_xy

CONTEXT_SCHEMA = "mousemind_continuous_context_v1"
SKILLS = ("go_to_goal", "evade_predator", "hold_position")


@dataclass(frozen=True)
class ContinuousContext:
    current: dict[str, Any]
    history: dict[str, Any]
    temporal_window: int

    @property
    def predator_visible(self) -> bool:
        return bool(self.current["predator_visible"])

    @property
    def near_occlusion(self) -> bool:
        return bool(self.current["near_occlusion"])

    def payload(self) -> dict[str, Any]:
        return copy.deepcopy({
            "schema_version": CONTEXT_SCHEMA,
            "action_schema": ACTION_SCHEMA,
            "temporal_window": self.temporal_window,
            "current": self.current,
            "history": self.history,
        })

    def serialize(self) -> str:
        return json.dumps(self.payload(), sort_keys=True, separators=(",", ":"), allow_nan=False)


class ContinuousContextBuilder:
    def __init__(self, temporal_window: int = 8):
        if temporal_window < 1:
            raise ValueError("temporal_window must be positive")
        self.temporal_window = temporal_window
        self.reset()

    def reset(self) -> None:
        self.positions = deque(maxlen=self.temporal_window)
        self.visibility = deque(maxlen=self.temporal_window)
        self.actions = deque(maxlen=self.temporal_window)
        self.skills = deque(maxlen=self.temporal_window)
        self.step_index = 0
        self.last_seen_step: int | None = None
        self.last_seen_xy: tuple[float, float] | None = None

    def observe(self, observation: Sequence[float]) -> ContinuousContext:
        obs = np.asarray(observation, dtype=np.float64)
        if obs.ndim != 1 or len(obs) not in (10, 15) or not np.isfinite(obs).all():
            raise ValueError("Continuous context needs finite legacy 10D or unstacked 15D observations")
        if len(obs) == 10:
            visible = bool(obs[3] != 0 or obs[4] != 0)
            predator = obs[3:6]
            near_wall = near_occlusion = False
            goal_distance, puffed, cooled = obs[6], bool(obs[7]), bool(obs[8])
        else:
            visible = bool(obs[3])
            predator = obs[4:7]
            near_wall, near_occlusion = bool(obs[7]), bool(obs[8])
            goal_distance, puffed, cooled = obs[14], bool(obs[10]), bool(obs[11])
        if visible:
            self.last_seen_step = self.step_index
            self.last_seen_xy = (float(predator[0]), float(predator[1]))
        age = None if self.last_seen_step is None else self.step_index - self.last_seen_step
        self.positions.append([round(float(x), 5) for x in obs[:2]])
        self.visibility.append(int(visible))
        current = {
            "prey_x": round(float(obs[0]), 5),
            "prey_y": round(float(obs[1]), 5),
            "prey_direction": round(float(obs[2]), 5),
            "predator_visible": visible,
            "predator_x": round(float(predator[0]), 5) if visible else None,
            "predator_y": round(float(predator[1]), 5) if visible else None,
            "predator_direction": round(float(predator[2]), 5) if visible else None,
            "last_seen_predator_x": None if self.last_seen_xy is None else round(self.last_seen_xy[0], 5),
            "last_seen_predator_y": None if self.last_seen_xy is None else round(self.last_seen_xy[1], 5),
            "predator_memory_age": age,
            "time_since_predator_seen": age,
            "near_wall": near_wall, "near_occlusion": near_occlusion,
            "puffed": puffed, "puff_cooled_down": cooled,
            "goal_distance": round(float(goal_distance), 5),
        }
        context = ContinuousContext(current, {
            "recent_prey_positions": list(self.positions),
            "recent_predator_visibility": list(self.visibility),
            "recent_xy_actions": list(self.actions),
            "recent_high_level_skills": list(self.skills),
        }, self.temporal_window)
        self.step_index += 1
        return context

    def record_decision(self, action_xy: Sequence[float], skill: Any) -> None:
        value = getattr(skill, "value", skill)
        if value not in (*SKILLS, "unlabeled"):
            raise ValueError(f"Unknown continuous skill: {value}")
        self.actions.append(list(validate_xy(action_xy)))
        self.skills.append(value)
