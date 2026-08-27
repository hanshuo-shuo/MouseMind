"""Language, utility, and target schema for counterfactual skill planning."""

from __future__ import annotations

import enum
import hashlib
import json
from collections.abc import Mapping, Sequence
from typing import Any

from mouse_llm.hierarchical.context import PlannerContext
from mouse_llm.hierarchical.policy import Skill


PLANNER_SCHEMA_VERSION = "mousemind_skill_planner_v1"
SYSTEM_PROMPT = (
    "You are the high-level Cellworld strategy planner. Use the instruction and "
    "semantic temporal state to select exactly one skill. Return only one JSON "
    'object: {"skill":"go_to_goal"}, {"skill":"evade_predator"}, or '
    '{"skill":"hold_position"}.'
)


class Preference(str, enum.Enum):
    SURVIVAL_FIRST = "survival_first"
    BALANCED = "balanced"
    GOAL_FIRST = "goal_first"
    HOLD = "hold"


INSTRUCTION_TEMPLATES: dict[str, dict[Preference, tuple[str, ...]]] = {
    "train": {
        Preference.SURVIVAL_FIRST: (
            "Prioritize survival and avoid being captured.",
            "Take the cautious route even if progress is slower.",
            "Minimize predator risk before moving toward the goal.",
        ),
        Preference.BALANCED: (
            "Balance goal progress with predator avoidance.",
            "Make progress without taking unnecessary risk.",
            "Trade off a short path against exposure to the predator.",
        ),
        Preference.GOAL_FIRST: (
            "Reach the goal quickly while keeping capture cost finite.",
            "Favor progress toward the goal over excessive caution.",
            "Move decisively toward the target unless danger is severe.",
        ),
        Preference.HOLD: (
            "Hold the current position.",
            "Wait here instead of making progress.",
            "Remain stationary until the next planning decision.",
        ),
    },
    "validation": {
        Preference.SURVIVAL_FIRST: (
            "Survival matters more than speed.",
        ),
        Preference.BALANCED: (
            "Seek the goal with measured caution.",
        ),
        Preference.GOAL_FIRST: (
            "Prefer immediate goal progress while still penalizing capture.",
        ),
        Preference.HOLD: (
            "Do not leave this location.",
        ),
    },
    "unseen_test": {
        Preference.SURVIVAL_FIRST: (
            "Treat any encounter as unacceptable, even at the cost of delay.",
        ),
        Preference.BALANCED: (
            "Advance prudently: neither freeze nor gamble with the predator.",
        ),
        Preference.GOAL_FIRST: (
            "Bias the plan toward rapid completion, accepting bounded exposure.",
        ),
        Preference.HOLD: (
            "Stay exactly where you are for this decision window.",
        ),
    },
}


def validate_instruction_template_splits() -> None:
    ownership: dict[str, str] = {}
    for split, by_preference in INSTRUCTION_TEMPLATES.items():
        if set(by_preference) != set(Preference):
            raise ValueError(f"Instruction split {split} does not cover every preference")
        for templates in by_preference.values():
            for template in templates:
                normalized = " ".join(template.lower().split())
                previous = ownership.setdefault(normalized, split)
                if previous != split:
                    raise ValueError(
                        f"Instruction template leaks across {previous} and {split}"
                    )


def instruction_for(
    preference: Preference,
    *,
    split: str,
    stable_key: str,
) -> str:
    validate_instruction_template_splits()
    try:
        templates = INSTRUCTION_TEMPLATES[split][preference]
    except KeyError as exc:
        raise ValueError(f"Unknown instruction split {split!r}") from exc
    digest = hashlib.sha256(f"{split}:{stable_key}".encode("utf-8")).digest()
    return templates[int.from_bytes(digest[:4], "big") % len(templates)]


def skill_target(skill: Skill) -> str:
    return json.dumps({"skill": skill.value}, separators=(",", ":"))


def parse_skill(text: str) -> Skill | None:
    try:
        payload = json.loads(text.strip())
    except json.JSONDecodeError:
        return None
    if not isinstance(payload, dict) or set(payload) != {"skill"}:
        return None
    value = payload["skill"]
    if not isinstance(value, str):
        return None
    try:
        return Skill(value)
    except ValueError:
        return None


def planner_messages(
    context: PlannerContext | Mapping[str, Any],
    *,
    preference: Preference,
    instruction: str,
    target_skill: Skill | None = None,
) -> list[dict[str, str]]:
    payload = context.payload() if isinstance(context, PlannerContext) else dict(context)
    user = json.dumps(
        {
            "planner_schema": PLANNER_SCHEMA_VERSION,
            "preference": preference.value,
            "instruction": instruction,
            "context": payload,
        },
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user},
    ]
    if target_skill is not None:
        messages.append({"role": "assistant", "content": skill_target(target_skill)})
    return messages


def _outcome_value(outcome: Mapping[str, Any], name: str) -> float:
    return float(outcome.get(name, 0.0))


def utility_scores(
    branches: Mapping[str, Mapping[str, Any]], preference: Preference
) -> dict[str, float]:
    scores: dict[str, float] = {}
    for skill in Skill:
        outcome = branches[skill.value]
        captures = _outcome_value(outcome, "captures")
        captured = float(bool(_outcome_value(outcome, "capture_within_h")))
        progress = _outcome_value(outcome, "goal_distance_change")
        exposure = _outcome_value(outcome, "predator_exposure_steps")
        path = _outcome_value(outcome, "path_length")
        occlusion = _outcome_value(outcome, "near_occlusion_exposure")
        goal = _outcome_value(outcome, "terminal_goal_event")
        if preference == Preference.SURVIVAL_FIRST:
            score = (
                1000.0 * (1.0 - captured)
                - 100.0 * captures
                + 10.0 * goal
                + progress
                - 0.02 * exposure
                - 0.01 * occlusion
                - 0.005 * path
            )
        elif preference == Preference.BALANCED:
            score = (
                4.0 * progress
                + 8.0 * goal
                - 12.0 * captured
                - 4.0 * captures
                - 0.08 * exposure
                - 0.03 * occlusion
                - 0.02 * path
            )
        elif preference == Preference.GOAL_FIRST:
            score = (
                10.0 * progress
                + 20.0 * goal
                - 3.0 * captured
                - 1.0 * captures
                - 0.01 * exposure
                - 0.01 * path
            )
        else:
            score = 1.0 if skill == Skill.HOLD_POSITION else 0.0
        scores[skill.value] = float(score)
    return scores


def select_skill(
    branches: Mapping[str, Mapping[str, Any]], preference: Preference
) -> tuple[Skill, dict[str, float]]:
    scores = utility_scores(branches, preference)
    if preference == Preference.SURVIVAL_FIRST:
        def survival_key(skill: Skill) -> tuple[float, ...]:
            outcome = branches[skill.value]
            return (
                -float(bool(_outcome_value(outcome, "capture_within_h"))),
                -_outcome_value(outcome, "captures"),
                _outcome_value(outcome, "goal_distance_change"),
                -_outcome_value(outcome, "predator_exposure_steps"),
                -_outcome_value(outcome, "path_length"),
                -float(tuple(Skill).index(skill)),
            )
        chosen = max(Skill, key=survival_key)
    else:
        chosen = max(
            Skill,
            key=lambda skill: (scores[skill.value], -tuple(Skill).index(skill)),
        )
    return chosen, scores


def preference_one_hot(preference: Preference) -> list[float]:
    return [float(preference == item) for item in Preference]


def assert_no_exact_instruction_leakage(rows_by_split: Mapping[str, Sequence[str]]) -> None:
    ownership: dict[str, str] = {}
    for split, values in rows_by_split.items():
        for value in values:
            normalized = " ".join(value.lower().split())
            previous = ownership.setdefault(normalized, split)
            if previous != split:
                raise ValueError(f"Instruction appears in both {previous} and {split}")
