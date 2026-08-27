import json

import numpy as np

from mouse_llm.hierarchical.context import PlannerContextBuilder, SCHEMA_VERSION
from mouse_llm.hierarchical.policy import Skill


def _observation(x=0.2, *, visible=False, near_wall=False):
    return np.asarray(
        [
            x,
            0.3,
            0.1,
            float(visible),
            0.7 if visible else 0.0,
            0.8 if visible else 0.0,
            -0.2 if visible else 0.0,
            float(near_wall),
            0.0,
            0.0,
            0.0,
            1.0,
            0.0,
            0.0,
            0.6,
        ],
        dtype=np.float32,
    )


def test_semantic_context_serialization_is_exact_and_machine_readable():
    builder = PlannerContextBuilder(temporal_window=2)
    context = builder.observe(_observation(near_wall=True))
    payload = json.loads(context.serialize())
    assert payload["schema_version"] == SCHEMA_VERSION
    assert payload["current"]["near_wall"] is True
    assert payload["current"]["goal_distance"] == 0.6
    assert tuple(payload["history"]["recent_prey_positions"][0]) == (0.2, 0.3)
    assert context.serialize() == context.serialize()
    assert context.numeric().shape == (27,)


def test_temporal_window_keeps_recent_actions_visibility_and_skills():
    builder = PlannerContextBuilder(temporal_window=2)
    builder.observe(_observation(0.1, visible=False))
    builder.record_decision(4, Skill.GO_TO_GOAL)
    builder.observe(_observation(0.2, visible=True))
    builder.record_decision(5, Skill.EVADE_PREDATOR)
    context = builder.observe(_observation(0.3, visible=False))
    assert np.allclose(context.recent_prey_positions, ((0.2, 0.3), (0.3, 0.3)))
    assert context.recent_predator_visibility == (1, 0)
    assert context.recent_low_level_actions == (4, 5)
    assert context.recent_high_level_skills == (
        Skill.GO_TO_GOAL.value,
        Skill.EVADE_PREDATOR.value,
    )
    assert context.time_since_predator_seen == 1
