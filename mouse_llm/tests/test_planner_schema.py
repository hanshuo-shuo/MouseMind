import pytest

from mouse_llm.data.planner_schema import (
    INSTRUCTION_TEMPLATES,
    Preference,
    assert_no_exact_instruction_leakage,
    parse_skill,
    select_skill,
    validate_instruction_template_splits,
)
from mouse_llm.hierarchical.policy import Skill


def _outcome(capture, progress, exposure=0.0):
    return {
        "capture_within_h": capture,
        "captures": capture,
        "goal_distance_change": progress,
        "predator_exposure_steps": exposure,
        "path_length": 1.0,
        "near_occlusion_exposure": 0,
        "terminal_goal_event": 0,
    }


def test_counterfactual_labels_prioritize_survival_lexicographically():
    branches = {
        Skill.GO_TO_GOAL.value: _outcome(1, 10.0),
        Skill.EVADE_PREDATOR.value: _outcome(0, 0.2),
        Skill.HOLD_POSITION.value: _outcome(0, 0.0, 2.0),
    }
    skill, utilities = select_skill(branches, Preference.SURVIVAL_FIRST)
    assert skill == Skill.EVADE_PREDATOR
    assert utilities[skill.value] > utilities[Skill.GO_TO_GOAL.value]
    hold, _ = select_skill(branches, Preference.HOLD)
    assert hold == Skill.HOLD_POSITION


def test_instruction_splits_and_exact_skill_parser_fail_closed():
    validate_instruction_template_splits()
    rows = {
        split: [text for templates in by_preference.values() for text in templates]
        for split, by_preference in INSTRUCTION_TEMPLATES.items()
    }
    assert_no_exact_instruction_leakage(rows)
    assert parse_skill('{"skill":"evade_predator"}') == Skill.EVADE_PREDATOR
    assert parse_skill('{"skill":"evade_predator","why":"x"}') is None
    with pytest.raises(ValueError, match="both"):
        assert_no_exact_instruction_leakage({"train": ["same"], "test": ["same"]})
