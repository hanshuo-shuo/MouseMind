from __future__ import annotations

import json

import pytest

from mouse_llm.data.schema import (
    FEATURE_NAMES,
    SEMANTIC_FEATURE_NAMES,
    conversation_observation,
    make_conversation,
    parse_user_observation,
    state_payload,
)


def test_prompt_contract_is_deterministic_and_machine_readable():
    observation = [0.1 * index for index in range(len(FEATURE_NAMES))]
    sample = make_conversation(observation, 24, action_count=295)
    assert [message["role"] for message in sample["conversations"]] == [
        "system",
        "user",
        "assistant",
    ]
    assert json.loads(sample["conversations"][-1]["content"]) == {"action": 24}
    assert "Current observation" in sample["conversations"][1]["content"]
    assert conversation_observation(sample["conversations"]) == pytest.approx(
        observation
    )
    assert parse_user_observation(sample["conversations"][1]["content"]) == (
        pytest.approx(observation)
    )


def test_schema_rejects_wrong_vector_length():
    with pytest.raises(ValueError, match="expects 10 values"):
        state_payload([0.0, 1.0])


def test_parser_rejects_prompt_schema_drift():
    with pytest.raises(ValueError, match="does not match"):
        parse_user_observation('{"obs_00":0}')


def test_recovered_semantics_cover_all_serialized_positions():
    assert len(FEATURE_NAMES) == len(SEMANTIC_FEATURE_NAMES) == 10
    assert SEMANTIC_FEATURE_NAMES[:3] == (
        "prey_x",
        "prey_y",
        "prey_direction",
    )
