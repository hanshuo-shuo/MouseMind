"""Frozen legacy high-level planning with only GO execution replaced by XY BC.

The old planner context still expects discrete action history. A nearest-catalog
ID is therefore recorded as an explicitly identified history proxy for a GO
request; the environment receives the original continuous destination. EVADE
and HOLD retain the old geometric controller and catalog destinations exactly.
This is an executor-swap ablation, not a newly post-trained planner.
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np

from mouse_llm.data.planner_schema import Preference
from mouse_llm.evaluation.closed_loop import PolicyDecision
from mouse_llm.hierarchical.context import SCHEMA_VERSION as LEGACY_CONTEXT_SCHEMA
from mouse_llm.hierarchical.policy import GeometricSkillController, Skill
from mouse_llm.hierarchical.verified_policy import ProposeVerifyPolicy

from .contracts import ACTION_SCHEMA, ContinuousDecision, validate_xy


class _LegacyHistoryController:
    """Return a legacy history ID while retaining the actual XY decision."""

    def __init__(self, specialist: Any, destinations: np.ndarray, *, evade_distance: float):
        self.specialist = specialist
        self.destinations = destinations
        self.legacy = GeometricSkillController(specialist, destinations, evade_distance=evade_distance)
        self.last_decision: ContinuousDecision | None = None
        self.last_history_action: int | None = None

    def clear(self) -> None:
        self.last_decision = None
        self.last_history_action = None

    def act(self, skill: Skill | str, observation: np.ndarray, *, goal_destination=None) -> PolicyDecision:
        if goal_destination is not None:
            raise ValueError("Legacy GO-only hybrid retains the original BotEvade goal interface")
        skill = Skill(skill)
        if skill == Skill.GO_TO_GOAL:
            decision = self.specialist.act(observation)
            if not isinstance(decision, ContinuousDecision):
                raise TypeError("GO replacement must return a ContinuousDecision")
            xy = validate_xy(decision.action)
            # This index is never used as the executed action. It is only what
            # the unchanged legacy context builder records for its next prompt.
            history_action = int(np.linalg.norm(self.destinations - np.asarray(xy), axis=1).argmin())
            self.last_decision = ContinuousDecision(xy, decision.valid, dict(decision.metadata))
        else:
            decision = self.legacy.act(skill, observation)
            history_action = int(decision.action)
            self.last_decision = ContinuousDecision(
                tuple(self.destinations[history_action]), decision.valid,
                dict(decision.metadata or {}),
            )
        self.last_history_action = history_action
        return PolicyDecision(action=history_action, valid=self.last_decision.valid)


class LegacyPlannerContinuousBCPolicy:
    """Reuse ProposeVerifyPolicy scheduling/context with an exact-XY GO skill.

    The caller supplies the existing frozen planner and continuous BC. This
    class never trains or alters either. In particular it does not introduce
    continuous-context prompts, remembered-threat evasion, or a new decoder.
    The outer policy intentionally exposes no ``controller`` attribute: the
    old geometric controller has no continuous threat-memory state to export.
    ``last_context`` is the original v1 PlannerContext used for that decision.
    """

    observation_mode = "environment"
    action_schema = ACTION_SCHEMA

    def __init__(
        self,
        *,
        specialist: Any,
        planner: Any,
        destinations: np.ndarray,
        preference: Preference,
        planner_horizon: int = 8,
        evade_distance: float = 0.35,
        temporal_window: int = 8,
        name: str = "legacy-minimind-continuous-bc",
    ):
        catalog = np.asarray(destinations, dtype=np.float64)
        if catalog.ndim != 2 or catalog.shape[1] != 2 or len(catalog) == 0:
            raise ValueError("Legacy history requires a nonempty [actions, 2] destination catalog")
        catalog = np.asarray([validate_xy(row) for row in catalog], dtype=np.float64)
        self.name = name
        self.specialist = specialist
        self.planner = planner
        self._hierarchy = ProposeVerifyPolicy(
            specialist=specialist, destinations=catalog, planner=planner,
            preference=Preference(preference), planner_horizon=planner_horizon,
            evade_distance=evade_distance, temporal_window=temporal_window,
            name=name,
        )
        self._controller = _LegacyHistoryController(specialist, catalog, evade_distance=evade_distance)
        # Reuse the original plan/replan/interrupt/context code verbatim. The
        # injected adapter changes the GO executor, never the planner's skill.
        self._hierarchy.controller = self._controller
        self.last_context = None

    def reset(self, seed: int) -> None:
        self._hierarchy.reset(seed)
        self._controller.clear()
        self.last_context = None

    def act(self, observation: Sequence[float]) -> ContinuousDecision:
        legacy_decision = self._hierarchy.act(np.asarray(observation).copy())
        self.last_context = self._hierarchy.last_context
        actual = self._controller.last_decision
        proxy = self._controller.last_history_action
        if actual is None or proxy is None:
            raise RuntimeError("Legacy hierarchy did not produce an executor decision")
        legacy_metadata = dict(legacy_decision.metadata or {})
        selected_skill = legacy_metadata["skill"]
        # These diagnostics do not change validity/fallback or the legacy
        # planner's returned skill. Raw text is only stored in private traces.
        last_valid = getattr(self.planner, "last_valid", None)
        metadata = {
            **actual.metadata, **legacy_metadata,
            "legacy_context_schema": LEGACY_CONTEXT_SCHEMA,
            "history_action_proxy": "nearest_catalog_id_for_history_only",
            "history_proxy_action": proxy,
            "history_proxy_used": selected_skill == Skill.GO_TO_GOAL.value,
            "history_proxy_source": "requested_xy_before_environment_feasibility_projection",
            "go_execution_source": "continuous_bc_exact_xy",
            "non_goal_execution_source": "legacy_geometric_catalog",
            "executor_swap_only": True,
            "planner_last_valid": None if last_valid is None else bool(last_valid),
            "planner_raw_response": getattr(self.planner, "last_response", None),
        }
        return ContinuousDecision(actual.action, legacy_decision.valid, metadata)
