"""Language-conditioned high-level skills over a fast specialist controller."""

from __future__ import annotations

import enum
from collections import deque
from dataclasses import dataclass
from typing import Any, Protocol, Sequence

import numpy as np

from mouse_llm.evaluation.closed_loop import PolicyDecision


class Skill(str, enum.Enum):
    GO_TO_GOAL = "go_to_goal"
    EVADE_PREDATOR = "evade_predator"
    HOLD_POSITION = "hold_position"


@dataclass(frozen=True)
class PlannerDecision:
    skill: Skill
    reason: str


class SpecialistPolicy(Protocol):
    name: str

    def reset(self, seed: int) -> None: ...

    def act(self, observation: np.ndarray) -> PolicyDecision: ...


SAFETY_PHRASES = (
    "avoid",
    "safe",
    "safety",
    "survive",
    "survival",
    "cautious",
    "cautiously",
    "exposure",
    "favor survival",
    "prioritize hiding",
)
HOLD_PHRASES = ("hold", "wait", "stay still", "remain here")


def predator_visible(observation: Sequence[float]) -> bool:
    values = np.asarray(observation).reshape(-1)
    if len(values) != 10:
        raise ValueError("Hierarchical policy expects the verified legacy 10D state")
    return bool(values[3] != 0.0 or values[4] != 0.0)


class InstructionSkillPlanner:
    """Auditable language baseline for the high-level planner interface.

    This is intentionally not presented as the final MiniMind planner. It
    establishes instruction semantics, paraphrase tests, replanning cadence,
    and a replaceable contract before skill-level MiniMind post-training.
    """

    def plan(
        self,
        instruction: str,
        observation: np.ndarray,
        history: Sequence[tuple[np.ndarray, int]],
    ) -> PlannerDecision:
        del history
        normalized = " ".join(instruction.lower().split())
        if any(phrase in normalized for phrase in HOLD_PHRASES):
            return PlannerDecision(Skill.HOLD_POSITION, "instruction_requests_hold")
        safety_requested = any(phrase in normalized for phrase in SAFETY_PHRASES)
        if safety_requested and predator_visible(observation):
            return PlannerDecision(
                Skill.EVADE_PREDATOR,
                "safety_instruction_and_predator_visible",
            )
        return PlannerDecision(Skill.GO_TO_GOAL, "goal_progress_default")


class GeometricSkillController:
    def __init__(
        self,
        specialist: SpecialistPolicy,
        destinations: np.ndarray,
        *,
        evade_distance: float = 0.35,
    ):
        if destinations.ndim != 2 or destinations.shape[1] != 2:
            raise ValueError("destinations must have shape [actions, 2]")
        self.specialist = specialist
        self.destinations = np.asarray(destinations, dtype=np.float64)
        self.evade_distance = evade_distance
        self.action_count = len(destinations)

    def _nearest_action(self, destination: np.ndarray) -> int:
        return int(np.linalg.norm(self.destinations - destination, axis=1).argmin())

    def act(
        self,
        skill: Skill,
        observation: np.ndarray,
        *,
        goal_destination: np.ndarray | None = None,
    ) -> PolicyDecision:
        state = np.asarray(observation, dtype=np.float64)
        if skill == Skill.GO_TO_GOAL:
            if goal_destination is not None:
                goal = np.asarray(goal_destination, dtype=np.float64).reshape(-1)
                if len(goal) != 2 or not np.isfinite(goal).all():
                    raise ValueError("goal_destination must contain finite x/y")
                return PolicyDecision(self._nearest_action(goal))
            return self.specialist.act(observation)
        prey = state[:2]
        if skill == Skill.HOLD_POSITION:
            return PolicyDecision(self._nearest_action(prey))
        predator = state[3:5]
        delta = prey - predator
        distance = float(np.linalg.norm(delta))
        if distance <= 1e-8:
            # Deterministic escape direction at the singularity.
            delta = np.asarray([-1.0, 0.0])
            distance = 1.0
        desired = np.clip(
            prey + delta / distance * self.evade_distance,
            np.asarray([0.02, 0.02]),
            np.asarray([0.98, 0.98]),
        )
        return PolicyDecision(self._nearest_action(desired))


class HierarchicalPolicy:
    def __init__(
        self,
        *,
        specialist: SpecialistPolicy,
        destinations: np.ndarray,
        instruction: str,
        planner_horizon: int = 4,
        history_length: int = 8,
        evade_distance: float = 0.35,
        name: str = "hierarchical-mlp",
        planner: InstructionSkillPlanner | None = None,
    ):
        if planner_horizon <= 0 or history_length <= 0:
            raise ValueError("planner_horizon and history_length must be positive")
        self.name = name
        self.action_count = len(destinations)
        self.specialist = specialist
        self.planner = planner or InstructionSkillPlanner()
        self.controller = GeometricSkillController(
            specialist, destinations, evade_distance=evade_distance
        )
        self.instruction = instruction
        self.planner_horizon = planner_horizon
        self.history: deque[tuple[np.ndarray, int]] = deque(maxlen=history_length)
        self.current_plan: PlannerDecision | None = None
        self.steps_until_replan = 0

    def reset(self, seed: int) -> None:
        self.specialist.reset(seed)
        self.history.clear()
        self.current_plan = None
        self.steps_until_replan = 0

    def act(self, observation: np.ndarray) -> PolicyDecision:
        replan = self.current_plan is None or self.steps_until_replan <= 0
        # Safety events are allowed to interrupt a stale goal plan immediately.
        if (
            self.current_plan is not None
            and self.current_plan.skill == Skill.GO_TO_GOAL
            and predator_visible(observation)
            and any(
                phrase in " ".join(self.instruction.lower().split())
                for phrase in SAFETY_PHRASES
            )
        ):
            replan = True
        if replan:
            self.current_plan = self.planner.plan(
                self.instruction,
                observation,
                tuple(self.history),
            )
            self.steps_until_replan = self.planner_horizon
        decision = self.controller.act(self.current_plan.skill, observation)
        self.steps_until_replan -= 1
        self.history.append((np.asarray(observation).copy(), decision.action))
        metadata: dict[str, Any] = {
            "skill": self.current_plan.skill.value,
            "planner_reason": self.current_plan.reason,
            "replanned": replan,
        }
        return PolicyDecision(
            action=decision.action,
            valid=decision.valid,
            raw_response=decision.raw_response,
            metadata=metadata,
        )
