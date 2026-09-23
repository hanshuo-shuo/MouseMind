"""GRPO reward using the existing counterfactual planner utility verbatim."""

from __future__ import annotations

from dataclasses import asdict, is_dataclass
from typing import Any, Mapping

from mouse_llm.data.planner_schema import Preference, utility_scores
from mouse_llm.hierarchical.policy import Skill


RL_PREFERENCES = (
    Preference.SURVIVAL_FIRST,
    Preference.BALANCED,
    Preference.GOAL_FIRST,
)


def _payload(outcome: Any) -> dict[str, Any]:
    result = asdict(outcome) if is_dataclass(outcome) else dict(outcome)
    result["skill"] = Skill(result["skill"]).value
    return result


def verified_reward(outcome: Any, preference: Preference | str) -> float:
    """Return exactly the utility used for SFT labels and DPO preferences."""
    preference = Preference(preference)
    if preference not in RL_PREFERENCES:
        raise ValueError("HOLD preference is excluded from verified RL reward")
    payload = _payload(outcome)
    # utility_scores is the single source of truth. It expects three branches;
    # only the branch keyed by this outcome's skill is read for these preferences.
    branches: Mapping[str, Mapping[str, Any]] = {
        skill.value: payload for skill in Skill
    }
    return utility_scores(branches, preference)[payload["skill"]]


def reward_components(outcome: Any, preference: Preference | str) -> dict[str, float]:
    """Expose the two monitored utility contributions without changing reward."""
    preference = Preference(preference)
    if preference not in RL_PREFERENCES:
        raise ValueError("HOLD preference is excluded from verified RL reward")
    row = _payload(outcome)
    captured = float(bool(row.get("capture_within_h", 0)))
    captures = float(row.get("captures", 0))
    progress = float(row.get("goal_distance_change", 0))
    goal = float(row.get("terminal_goal_event", 0))
    if preference == Preference.SURVIVAL_FIRST:
        capture_component = 1000.0 * (1.0 - captured) - 100.0 * captures
        goal_component = 10.0 * goal + progress
    elif preference == Preference.BALANCED:
        capture_component = -12.0 * captured - 4.0 * captures
        goal_component = 8.0 * goal + 4.0 * progress
    else:
        capture_component = -3.0 * captured - captures
        goal_component = 20.0 * goal + 10.0 * progress
    return {
        "capture": capture_component,
        "goal_progress": goal_component,
        "capture_within_h": captured,
        "captures": captures,
        "terminal_goal_event": goal,
    }
