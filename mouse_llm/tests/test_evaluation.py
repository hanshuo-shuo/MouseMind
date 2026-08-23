from __future__ import annotations

import numpy as np

from mouse_llm.evaluation.evaluate_policy import ActionTokenConstraint, parse_action


def test_parse_action_accepts_only_bounded_integer_json():
    assert parse_action('{"action":24}', action_count=295) == 24
    assert parse_action('answer: {"action": 294}', action_count=295) is None
    assert parse_action('{"action":295}', action_count=295) is None
    assert parse_action('{"action":true}', action_count=295) is None
    assert parse_action('{"action":24,"note":"extra"}', action_count=295) is None
    assert parse_action("action 24", action_count=295) is None


class _CharacterTokenizer:
    eos_token_id = 999

    @staticmethod
    def encode(text, add_special_tokens=False):
        assert add_special_tokens is False
        return [ord(character) for character in text]


def test_action_constraint_allows_only_json_trie_tokens():
    constraint = ActionTokenConstraint(_CharacterTokenizer(), action_count=3)
    allowed = constraint.prefix_allowed_tokens_fn(prompt_length=2)
    assert allowed(0, np.asarray([10, 11])) == [ord("{")]
    prefix = '{"action":'
    next_tokens = allowed(
        0, np.asarray([10, 11, *[ord(character) for character in prefix]])
    )
    assert next_tokens == [ord("0"), ord("1"), ord("2")]
    complete = '{"action":1}'
    assert allowed(
        0, np.asarray([10, 11, *[ord(character) for character in complete]])
    ) == [999]
