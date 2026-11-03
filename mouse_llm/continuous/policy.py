"""Continuous execution of the existing three high-level skills."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

from mouse_llm.data.planner_schema import Preference
from mouse_llm.hierarchical.policy import PlannerDecision, Skill
from mouse_llm.evaluation.closed_loop import LEGACY_GYM_SOURCE_INDICES, policy_observation

from .contracts import ACTION_SCHEMA, ContinuousDecision, validate_xy


def legacy_observation(observation: Sequence[float]) -> np.ndarray:
    """Use only the public 10D / unstacked 15D observation contract."""
    values = np.asarray(observation, dtype=np.float64).reshape(-1)
    if len(values) not in (10, 15):
        raise ValueError("Continuous policies require legacy 10D or unstacked BotEvade 15D")
    return policy_observation(values, indices=LEGACY_GYM_SOURCE_INDICES)


class ContinuousSkillController:
    """An XY specialist plus visible / remembered threat geometry.

    Memory is updated on every call, including GO and HOLD. Hidden zeros are
    never interpreted as a predator position. Memory age counts controller
    decisions; exact branch replay must restore it by replaying the prefix.
    """

    action_schema = ACTION_SCHEMA

    def __init__(self, specialist: Any, *, evade_distance: float = 0.35, memory_steps: int = 8):
        if not np.isfinite(evade_distance) or evade_distance <= 0:
            raise ValueError("evade_distance must be positive and finite")
        if isinstance(memory_steps, bool) or int(memory_steps) != memory_steps or memory_steps < 0:
            raise ValueError("memory_steps must be a non-negative integer")
        self.specialist = specialist
        self.evade_distance = float(evade_distance)
        self.memory_steps = int(memory_steps)
        self._last_seen: np.ndarray | None = None
        self._age: int | None = None

    def reset(self, seed: int) -> None:
        self.specialist.reset(int(seed))
        self._last_seen = None
        self._age = None

    def observe(self, observation: Sequence[float]) -> np.ndarray:
        state = legacy_observation(observation)
        if state[3] != 0.0 or state[4] != 0.0:
            self._last_seen = state[3:5].astype(np.float64).copy()
            self._age = 0
        elif self._age is not None:
            self._age += 1
        return state

    def memory_state(self) -> dict[str, Any]:
        return {"last_seen_predator_xy": None if self._last_seen is None else self._last_seen.tolist(),
                "predator_memory_age": self._age}

    def restore_memory(self, state: dict[str, Any]) -> None:
        xy, age = state.get("last_seen_predator_xy"), state.get("predator_memory_age")
        if xy is None:
            if age is not None:
                raise ValueError("Memory age requires a remembered predator position")
            self._last_seen, self._age = None, None
            return
        if isinstance(age, bool) or not isinstance(age, int) or age < 0:
            raise ValueError("Remembered predator age must be a non-negative integer")
        self._last_seen = np.asarray(validate_xy(xy), dtype=np.float64)
        self._age = age

    def act(self, skill: Skill | str, observation: Sequence[float]) -> ContinuousDecision:
        skill = Skill(skill)
        state = self.observe(observation)
        metadata = {"skill": skill.value, "action_schema": ACTION_SCHEMA, **self.memory_state()}
        if skill == Skill.GO_TO_GOAL:
            decision = self.specialist.act(state)
            return ContinuousDecision(decision.action, decision.valid, {**decision.metadata, **metadata})
        prey = np.asarray(validate_xy(state[:2]), dtype=np.float64)
        if skill == Skill.HOLD_POSITION:
            return ContinuousDecision(tuple(prey), metadata=metadata)
        if self._last_seen is None or self._age is None or self._age > self.memory_steps:
            decision = self.specialist.act(state)
            return ContinuousDecision(decision.action, decision.valid, {
                **decision.metadata, **metadata, "executed_skill": Skill.GO_TO_GOAL.value,
                "fallback_reason": "no_unexpired_public_predator_memory",
            })
        delta = prey - self._last_seen
        distance = float(np.linalg.norm(delta))
        if distance <= 1e-8:
            delta, distance = np.asarray([-1.0, 0.0]), 1.0
            metadata["escape_singularity"] = True
        destination = np.clip(prey + self.evade_distance * delta / distance, 0.0, 1.0)
        metadata["predator_source"] = "visible" if self._age == 0 else "public_memory"
        return ContinuousDecision(tuple(destination), metadata=metadata)


class RuleContinuousPlanner:
    """A transparent same-interface reference, never a trained LLM result."""

    def plan_context(self, context, preference: Preference) -> PlannerDecision:
        preference = Preference(preference)
        if preference == Preference.HOLD:
            return PlannerDecision(Skill.HOLD_POSITION, "hold_preference")
        if context.predator_visible and preference in (Preference.SURVIVAL_FIRST, Preference.BALANCED):
            return PlannerDecision(Skill.EVADE_PREDATOR, "visible_predator")
        return PlannerDecision(Skill.GO_TO_GOAL, "goal_default")


class ContinuousHierarchicalPolicy:
    observation_mode = "environment"
    action_schema = ACTION_SCHEMA

    def __init__(self, *, specialist, planner=None, preference=Preference.SURVIVAL_FIRST,
                 planner_horizon: int = 8, temporal_window: int = 8,
                 evade_distance: float = 0.35, memory_steps: int = 8,
                 name: str = "continuous-hierarchy"):
        from .context import ContinuousContextBuilder

        if isinstance(planner_horizon, bool) or int(planner_horizon) != planner_horizon or planner_horizon <= 0:
            raise ValueError("planner_horizon must be a positive integer")
        self.name = name
        self.planner = planner or RuleContinuousPlanner()
        self.preference = Preference(preference)
        self.planner_horizon = int(planner_horizon)
        self.controller = ContinuousSkillController(specialist, evade_distance=evade_distance, memory_steps=memory_steps)
        self.context_builder = ContinuousContextBuilder(temporal_window=temporal_window)
        self.current_skill: Skill | None = None
        self.steps_until_replan = 0
        self.last_context = None

    def reset(self, seed: int) -> None:
        self.controller.reset(seed)
        reset = getattr(self.planner, "reset", None)
        if callable(reset):
            reset(seed)
        self.context_builder.reset()
        self.current_skill, self.last_context = None, None
        self.steps_until_replan = 0

    def act(self, observation: Sequence[float]) -> ContinuousDecision:
        context = self.context_builder.observe(observation)
        self.last_context = context
        replan = self.current_skill is None or self.steps_until_replan <= 0
        reason = "horizon"
        if (not replan and context.predator_visible and self.current_skill == Skill.GO_TO_GOAL
                and self.preference in (Preference.SURVIVAL_FIRST, Preference.BALANCED)):
            replan, reason = True, "visible_predator_interrupt"
        if replan:
            proposal = self.planner.plan_context(context, self.preference)
            self.current_skill = Skill(proposal.skill)
            self.steps_until_replan = self.planner_horizon
        decision = self.controller.act(self.current_skill, legacy_observation(observation))
        self.context_builder.record_decision(decision.action, self.current_skill)
        self.steps_until_replan -= 1
        return ContinuousDecision(decision.action, decision.valid, {
            **decision.metadata, "skill": self.current_skill.value,
            "replanned": replan, "replan_reason": reason if replan else "continue_plan",
        })


class DirectContinuousPolicy:
    observation_mode = "environment"
    action_schema = ACTION_SCHEMA

    def __init__(self, specialist, *, name="continuous-bc"):
        self.specialist, self.name = specialist, name

    def reset(self, seed):
        self.specialist.reset(seed)

    def act(self, observation):
        return self.specialist.act(legacy_observation(observation))


class DiscreteDestinationAdapter:
    """Execute an existing discrete policy's destinations as an explicit control.

    This comparator retains the catalog-constrained policy; it is never called
    a learned continuous policy. XY coordinates pass unchanged to the native
    continuous environment.
    """

    observation_mode = "environment"
    action_schema = ACTION_SCHEMA

    def __init__(self, policy, destinations, *, name="discrete-catalog-reference"):
        array = np.asarray(destinations, dtype=np.float64)
        if array.ndim != 2 or array.shape[1] != 2 or not len(array):
            raise ValueError("Catalog must have shape [actions, 2]")
        self.destinations = np.asarray([validate_xy(row) for row in array])
        self.policy, self.name = policy, name

    def reset(self, seed):
        self.policy.reset(seed)

    def act(self, observation):
        policy_input = (
            np.asarray(observation).copy()
            if getattr(self.policy, "observation_mode", None) == "environment"
            else legacy_observation(observation)
        )
        decision = self.policy.act(policy_input)
        action = decision.action
        if isinstance(action, bool) or int(action) != action or not 0 <= action < len(self.destinations):
            raise ValueError("Discrete reference emitted an invalid catalog action")
        return ContinuousDecision(tuple(self.destinations[int(action)]), decision.valid, {
            **(decision.metadata or {}), "discrete_reference_action": int(action),
            "catalog_constrained_reference": True,
        })
