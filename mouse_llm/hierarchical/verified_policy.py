"""Learned propose-verify hierarchy over the existing low-level specialist."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np

from mouse_llm.data.planner_schema import Preference
from mouse_llm.evaluation.closed_loop import (
    LEGACY_GYM_SOURCE_INDICES,
    PolicyDecision,
    policy_observation,
)
from mouse_llm.hierarchical.context import PlannerContext, PlannerContextBuilder
from mouse_llm.hierarchical.policy import (
    GeometricSkillController,
    PlannerDecision,
    Skill,
    SpecialistPolicy,
)


class LearnedPlanner(Protocol):
    def plan_context(
        self, context: PlannerContext, preference: Preference
    ) -> PlannerDecision: ...


class RiskVerifier(Protocol):
    def score(self, context: PlannerContext, skill: Skill) -> float: ...


@dataclass(frozen=True)
class VerificationDecision:
    proposed_skill: Skill
    executed_skill: Skill
    proposed_risk: float
    executed_risk: float
    was_overridden: bool
    override_reason: str | None


def verify_proposal(
    *,
    context: PlannerContext,
    proposed_skill: Skill,
    verifier: RiskVerifier,
    threshold: float,
    preference: Preference,
    near_tie: float = 0.02,
) -> VerificationDecision:
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("Risk threshold must be in [0, 1]")
    proposed_risk = float(verifier.score(context, proposed_skill))
    if proposed_risk <= threshold:
        return VerificationDecision(
            proposed_skill=proposed_skill,
            executed_skill=proposed_skill,
            proposed_risk=proposed_risk,
            executed_risk=proposed_risk,
            was_overridden=False,
            override_reason=None,
        )
    scores = {skill: float(verifier.score(context, skill)) for skill in Skill}
    minimum = min(scores.values())
    candidates = [skill for skill in Skill if scores[skill] <= minimum + near_tie]
    preference_order = {
        Preference.SURVIVAL_FIRST: (
            Skill.EVADE_PREDATOR,
            Skill.HOLD_POSITION,
            Skill.GO_TO_GOAL,
        ),
        Preference.BALANCED: (
            (Skill.EVADE_PREDATOR, Skill.GO_TO_GOAL, Skill.HOLD_POSITION)
            if context.predator_visible
            else (Skill.GO_TO_GOAL, Skill.EVADE_PREDATOR, Skill.HOLD_POSITION)
        ),
        Preference.GOAL_FIRST: (
            Skill.GO_TO_GOAL,
            Skill.EVADE_PREDATOR,
            Skill.HOLD_POSITION,
        ),
        Preference.HOLD: (
            Skill.HOLD_POSITION,
            Skill.EVADE_PREDATOR,
            Skill.GO_TO_GOAL,
        ),
    }[preference]
    executed = next(skill for skill in preference_order if skill in candidates)
    return VerificationDecision(
        proposed_skill=proposed_skill,
        executed_skill=executed,
        proposed_risk=proposed_risk,
        executed_risk=scores[executed],
        was_overridden=executed != proposed_skill,
        override_reason="proposed_capture_risk_above_threshold",
    )


class ProposeVerifyPolicy:
    observation_mode = "environment"

    def __init__(
        self,
        *,
        specialist: SpecialistPolicy,
        destinations: np.ndarray,
        planner: LearnedPlanner,
        preference: Preference,
        planner_horizon: int = 4,
        verifier: RiskVerifier | None = None,
        risk_threshold: float = 0.5,
        evade_distance: float = 0.35,
        temporal_window: int = 8,
        name: str = "learned-hierarchical",
    ):
        if planner_horizon <= 0:
            raise ValueError("planner_horizon must be positive")
        self.name = name
        self.action_count = len(destinations)
        self.specialist = specialist
        self.controller = GeometricSkillController(
            specialist, destinations, evade_distance=evade_distance
        )
        self.planner = planner
        self.preference = preference
        self.planner_horizon = planner_horizon
        self.verifier = verifier
        self.risk_threshold = risk_threshold
        self.context_builder = PlannerContextBuilder(temporal_window=temporal_window)
        self.current_skill: Skill | None = None
        self.current_proposal: Skill | None = None
        self.current_risk = 0.0
        self.steps_until_replan = 0

    def reset(self, seed: int) -> None:
        self.specialist.reset(seed)
        reset = getattr(self.planner, "reset", None)
        if callable(reset):
            reset(seed)
        self.context_builder.reset()
        self.current_skill = None
        self.current_proposal = None
        self.current_risk = 0.0
        self.steps_until_replan = 0

    def act(self, observation: np.ndarray) -> PolicyDecision:
        context = self.context_builder.observe(observation)
        replan_reason = "horizon"
        replan = self.current_skill is None or self.steps_until_replan <= 0
        if (
            not replan
            and context.predator_visible
            and self.current_skill == Skill.GO_TO_GOAL
            and self.preference in {Preference.SURVIVAL_FIRST, Preference.BALANCED}
        ):
            replan = True
            replan_reason = "emergency_predator_visibility_interrupt"
        planner_latency = 0.0
        verifier_latency = 0.0
        was_overridden = False
        override_reason = None
        if replan:
            planner_start = time.perf_counter()
            proposal = self.planner.plan_context(context, self.preference)
            planner_latency = time.perf_counter() - planner_start
            self.current_proposal = proposal.skill
            self.current_skill = proposal.skill
            self.current_risk = 0.0
            if self.verifier is not None:
                verifier_start = time.perf_counter()
                verified = verify_proposal(
                    context=context,
                    proposed_skill=proposal.skill,
                    verifier=self.verifier,
                    threshold=self.risk_threshold,
                    preference=self.preference,
                )
                verifier_latency = time.perf_counter() - verifier_start
                self.current_skill = verified.executed_skill
                self.current_risk = verified.executed_risk
                was_overridden = verified.was_overridden
                override_reason = verified.override_reason
            self.steps_until_replan = self.planner_horizon
        assert self.current_skill is not None
        legacy = policy_observation(
            observation, indices=LEGACY_GYM_SOURCE_INDICES
        )
        decision = self.controller.act(self.current_skill, legacy)
        self.context_builder.record_decision(decision.action, self.current_skill)
        self.steps_until_replan -= 1
        metadata: dict[str, Any] = {
            "skill": self.current_skill.value,
            "proposed_skill": (
                self.current_proposal.value if self.current_proposal else self.current_skill.value
            ),
            "executed_skill": self.current_skill.value,
            "risk_score": self.current_risk,
            "was_overridden": was_overridden,
            "override_reason": override_reason,
            "replanned": replan,
            "replan_reason": replan_reason if replan else "continue_plan",
            "planner_latency_seconds": planner_latency,
            "verifier_latency_seconds": verifier_latency,
        }
        return PolicyDecision(
            action=decision.action,
            valid=decision.valid,
            raw_response=decision.raw_response,
            metadata=metadata,
        )
