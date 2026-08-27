"""Semantic temporal context shared by learned planners and risk critics."""

from __future__ import annotations

import json
import math
from collections import deque
from dataclasses import dataclass
from typing import Sequence

import numpy as np

from mouse_llm.hierarchical.policy import Skill


SCHEMA_VERSION = "mousemind_planner_context_v1"
SKILL_ORDER = tuple(Skill)


def _finite(value: float, name: str) -> float:
    numeric = float(value)
    if not math.isfinite(numeric):
        raise ValueError(f"{name} must be finite")
    return numeric


def _rounded(value: float) -> int | float:
    rounded = round(_finite(value, "context value"), 5)
    return 0 if rounded == 0 else rounded


@dataclass(frozen=True)
class PlannerContext:
    prey_x: float
    prey_y: float
    prey_direction: float
    predator_visible: bool
    predator_x: float
    predator_y: float
    predator_direction: float
    time_since_predator_seen: int
    near_wall: bool
    near_occlusion: bool
    puffed: bool
    puff_cooled_down: bool
    goal_distance: float
    recent_prey_positions: tuple[tuple[float, float], ...]
    recent_predator_visibility: tuple[int, ...]
    recent_low_level_actions: tuple[int, ...]
    recent_high_level_skills: tuple[str, ...]
    temporal_window: int = 8
    schema_version: str = SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != SCHEMA_VERSION:
            raise ValueError(f"Unsupported planner context {self.schema_version}")
        if self.temporal_window <= 0:
            raise ValueError("temporal_window must be positive")
        for sequence in (
            self.recent_prey_positions,
            self.recent_predator_visibility,
            self.recent_low_level_actions,
            self.recent_high_level_skills,
        ):
            if len(sequence) > self.temporal_window:
                raise ValueError("Temporal history exceeds the declared window")
        if self.time_since_predator_seen < 0:
            raise ValueError("time_since_predator_seen must be non-negative")

    def payload(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "temporal_window": self.temporal_window,
            "current": {
                "prey_x": _rounded(self.prey_x),
                "prey_y": _rounded(self.prey_y),
                "prey_direction": _rounded(self.prey_direction),
                "predator_visible": self.predator_visible,
                "predator_x": _rounded(self.predator_x),
                "predator_y": _rounded(self.predator_y),
                "predator_direction": _rounded(self.predator_direction),
                "time_since_predator_seen": self.time_since_predator_seen,
                "near_wall": self.near_wall,
                "near_occlusion": self.near_occlusion,
                "puffed": self.puffed,
                "puff_cooled_down": self.puff_cooled_down,
                "goal_distance": _rounded(self.goal_distance),
            },
            "history": {
                "recent_prey_positions": [
                    [_rounded(x), _rounded(y)]
                    for x, y in self.recent_prey_positions
                ],
                "recent_predator_visibility": list(
                    self.recent_predator_visibility
                ),
                "recent_low_level_actions": list(self.recent_low_level_actions),
                "recent_high_level_skills": list(self.recent_high_level_skills),
            },
        }

    def serialize(self) -> str:
        return json.dumps(
            self.payload(), sort_keys=True, separators=(",", ":"), allow_nan=False
        )

    def numeric(self, *, action_count: int = 295) -> np.ndarray:
        """Return a fixed-width vector for numeric planner/risk baselines."""
        if action_count <= 0:
            raise ValueError("action_count must be positive")
        scalars = [
            self.prey_x,
            self.prey_y,
            self.prey_direction / math.pi,
            float(self.predator_visible),
            self.predator_x,
            self.predator_y,
            self.predator_direction / math.pi,
            min(self.time_since_predator_seen, 100) / 100.0,
            float(self.near_wall),
            float(self.near_occlusion),
            float(self.puffed),
            float(self.puff_cooled_down),
            self.goal_distance,
        ]
        prey_history = list(self.recent_prey_positions)[-self.temporal_window :]
        visibility_history = list(self.recent_predator_visibility)[
            -self.temporal_window :
        ]
        action_history = list(self.recent_low_level_actions)[-self.temporal_window :]
        skill_history = list(self.recent_high_level_skills)[-self.temporal_window :]
        prey_history = [(0.0, 0.0)] * (
            self.temporal_window - len(prey_history)
        ) + prey_history
        visibility_history = [0] * (
            self.temporal_window - len(visibility_history)
        ) + visibility_history
        action_history = [-1] * (
            self.temporal_window - len(action_history)
        ) + action_history
        skill_history = [""] * (
            self.temporal_window - len(skill_history)
        ) + skill_history
        vector = list(scalars)
        vector.extend(value for point in prey_history for value in point)
        vector.extend(float(value) for value in visibility_history)
        vector.extend(
            (float(value) + 1.0) / (action_count + 1.0) for value in action_history
        )
        for value in skill_history:
            vector.extend(float(value == skill.value) for skill in SKILL_ORDER)
        result = np.asarray(vector, dtype=np.float32)
        if not np.isfinite(result).all():
            raise ValueError("Planner context produced non-finite numeric features")
        return result


class PlannerContextBuilder:
    """Stateful, deterministic builder for an eight-step semantic history."""

    def __init__(self, *, temporal_window: int = 8):
        if temporal_window <= 0:
            raise ValueError("temporal_window must be positive")
        self.temporal_window = temporal_window
        self.prey_positions: deque[tuple[float, float]] = deque(
            maxlen=temporal_window
        )
        self.predator_visibility: deque[int] = deque(maxlen=temporal_window)
        self.actions: deque[int] = deque(maxlen=temporal_window)
        self.skills: deque[str] = deque(maxlen=temporal_window)
        self.step_index = 0
        self.last_predator_seen_step: int | None = None

    def reset(self) -> None:
        self.prey_positions.clear()
        self.predator_visibility.clear()
        self.actions.clear()
        self.skills.clear()
        self.step_index = 0
        self.last_predator_seen_step = None

    def observe(self, observation: Sequence[float]) -> PlannerContext:
        values = np.asarray(observation, dtype=np.float64).reshape(-1)
        if len(values) == 10:
            prey_x, prey_y, prey_direction = values[:3]
            predator_x, predator_y, predator_direction = values[3:6]
            predator_visible = bool(predator_x != 0.0 or predator_y != 0.0)
            near_wall = False
            near_occlusion = False
            puffed = bool(values[7])
            puff_cooled_down = bool(values[8])
            goal_distance = values[6]
        elif len(values) >= 15:
            prey_x, prey_y, prey_direction = values[:3]
            predator_visible = bool(values[3])
            predator_x, predator_y, predator_direction = values[4:7]
            near_wall = bool(values[7])
            near_occlusion = bool(values[8])
            puffed = bool(values[10])
            puff_cooled_down = bool(values[11])
            goal_distance = values[14]
        else:
            raise ValueError("Planner context expects legacy 10D or BotEvade 15D state")
        if not np.isfinite(values).all():
            raise ValueError("Planner observation contains non-finite values")
        if predator_visible:
            self.last_predator_seen_step = self.step_index
        time_since = (
            self.temporal_window + 1
            if self.last_predator_seen_step is None
            else self.step_index - self.last_predator_seen_step
        )
        self.prey_positions.append((float(prey_x), float(prey_y)))
        self.predator_visibility.append(int(predator_visible))
        context = PlannerContext(
            prey_x=_finite(prey_x, "prey_x"),
            prey_y=_finite(prey_y, "prey_y"),
            prey_direction=_finite(prey_direction, "prey_direction"),
            predator_visible=predator_visible,
            predator_x=_finite(predator_x, "predator_x"),
            predator_y=_finite(predator_y, "predator_y"),
            predator_direction=_finite(predator_direction, "predator_direction"),
            time_since_predator_seen=time_since,
            near_wall=near_wall,
            near_occlusion=near_occlusion,
            puffed=puffed,
            puff_cooled_down=puff_cooled_down,
            goal_distance=_finite(goal_distance, "goal_distance"),
            recent_prey_positions=tuple(self.prey_positions),
            recent_predator_visibility=tuple(self.predator_visibility),
            recent_low_level_actions=tuple(self.actions),
            recent_high_level_skills=tuple(self.skills),
            temporal_window=self.temporal_window,
        )
        self.step_index += 1
        return context

    def record_decision(self, action: int, skill: Skill | str) -> None:
        if int(action) < 0:
            raise ValueError("action must be non-negative")
        skill_value = skill.value if isinstance(skill, Skill) else str(skill)
        if skill_value not in {item.value for item in SKILL_ORDER}:
            raise ValueError(f"Unknown high-level skill {skill_value!r}")
        self.actions.append(int(action))
        self.skills.append(skill_value)
