from __future__ import annotations

import numpy as np
import torch
from types import SimpleNamespace

from mouse_llm.evaluation.evaluate_policy import (
    ActionTokenConstraint,
    generate_response,
    parse_action,
)


def test_parse_action_accepts_only_bounded_integer_json():
    assert parse_action('{"action":24}', action_count=295) == 24
    assert parse_action('answer: {"action": 294}', action_count=295) is None
    assert parse_action('{"action":295}', action_count=295) is None
    assert parse_action('{"action":true}', action_count=295) is None
    assert parse_action('{"action":24,"note":"extra"}', action_count=295) is None
    assert parse_action("action 24", action_count=295) is None


class _CharacterTokenizer:
    eos_token_id = 999
    pad_token_id = 0

    @staticmethod
    def encode(text, add_special_tokens=False):
        assert add_special_tokens is False
        return [ord(character) for character in text]

    @staticmethod
    def decode(tokens, skip_special_tokens=True):
        assert skip_special_tokens is True
        return "".join(chr(int(token)) for token in tokens if int(token) != 999)


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


def test_constrained_generation_masks_logits_without_generation_mixin_support():
    class _Model:
        def __call__(self, input_ids, attention_mask=None):
            del attention_mask
            return SimpleNamespace(
                logits=torch.zeros((1, input_ids.shape[1], 1000))
            )

        def generate(self, **kwargs):
            raise AssertionError("constrained decoding must not call model.generate")

    tokenizer = _CharacterTokenizer()
    constraint = ActionTokenConstraint(tokenizer, action_count=1)
    response, _ = generate_response(
        _Model(),
        tokenizer,
        {
            "input_ids": torch.asarray([[1, 2]]),
            "attention_mask": torch.ones((1, 2), dtype=torch.long),
        },
        max_new_tokens=24,
        action_constraint=constraint,
    )
    assert response == '{"action":0}'
