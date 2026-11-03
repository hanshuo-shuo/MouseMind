import numpy as np
import pytest

from mouse_llm.continuous.context import CONTEXT_SCHEMA, ContinuousContextBuilder
from mouse_llm.continuous.contracts import validate_xy
from mouse_llm.data.planner_schema import Preference, planner_messages


def test_hidden_predator_retains_only_previously_observed_position():
    builder = ContinuousContextBuilder(2)
    obs = np.zeros(15)
    obs[:2] = [0.3, 0.4]
    obs[3:7] = [1, 0.8, 0.6, 1.2]
    first = builder.observe(obs)
    builder.record_decision((0.35, 0.45), "go_to_goal")
    obs[3] = 0
    obs[4:7] = [0.9, 0.9, 0.9]  # even if an upstream source forgets masking
    hidden = builder.observe(obs)
    assert hidden.current["predator_x"] is None
    assert hidden.current["last_seen_predator_x"] == 0.8
    assert hidden.current["predator_memory_age"] == 1
    assert hidden.history["recent_xy_actions"] == [[0.35, 0.45]]
    assert first.history["recent_xy_actions"] == []
    messages = planner_messages(hidden, preference=Preference.SURVIVAL_FIRST, instruction="Avoid capture")
    assert CONTEXT_SCHEMA in messages[1]["content"]
    builder.reset()
    assert builder.observe(obs).current["last_seen_predator_x"] is None


@pytest.mark.parametrize("value", [(0.1,), (0.1, 0.2, 0), (float('nan'), 0), (1.1, 0), (True, 0), ('0.2', 0)])
def test_xy_rejects_invalid_commands(value):
    with pytest.raises(ValueError):
        validate_xy(value)
