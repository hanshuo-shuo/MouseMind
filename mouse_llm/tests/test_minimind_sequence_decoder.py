"""Skill decoder tests with exact small-vocabulary autoregressive probabilities.

No real tokenizer, pretrained weights, Transformers model, or simulator is used.
The existing failed closed-loop runs remain evidence about the old decoder;
these tests only establish the semantics of an explicit inference comparator.
"""

import inspect
import json
import math
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

from mouse_llm.data.planner_schema import Preference, skill_target
from mouse_llm.evaluation.evaluate_policy import generate_response
from mouse_llm.hierarchical.minimind_planner import MiniMindSkillPlanner, SkillTokenConstraint
from mouse_llm.hierarchical.policy import Skill
from mouse_llm.hierarchical.sequence_decoder import score_canonical_skill_sequences


class TinyTokenizer:
    eos_token_id = 1
    eos_token = "<EOS>"
    pad_token_id = 0
    truncation_side = "right"
    prompt_text = "PROMPT"
    prompt_tokens = [2, 3]
    skill_tokens = {
        Skill.GO_TO_GOAL.value: [4, 6],
        Skill.EVADE_PREDATOR.value: [5, 7, 8],
        Skill.HOLD_POSITION.value: [9],
    }

    def encode(self, text, add_special_tokens=False):
        del add_special_tokens  # PROMPT already represents serialized chat.
        prefix = []
        if text.startswith(self.prompt_text):
            prefix = list(self.prompt_tokens)
            text = text[len(self.prompt_text):]
        suffix = []
        if text.endswith(self.eos_token):
            text = text[:-len(self.eos_token)]
            suffix = [self.eos_token_id]
        if not text:
            return [*prefix, *suffix]
        payload = json.loads(text)
        assert set(payload) == {"skill"}
        return [*prefix, *self.skill_tokens[payload["skill"]], *suffix]

    def decode(self, tokens, skip_special_tokens=True):
        assert skip_special_tokens
        values = [int(value) for value in tokens if int(value) not in (self.pad_token_id, self.eos_token_id)]
        for skill, expected in self.skill_tokens.items():
            if values == expected:
                return skill_target(Skill(skill))
        raise AssertionError("Toy decoder received a noncanonical skill sequence")

    def apply_chat_template(self, messages, **kwargs):
        assert kwargs["add_generation_prompt"] is True
        assert messages[-1]["role"] == "user"
        return self.prompt_text

    def __call__(self, text, *, return_tensors=None, add_special_tokens=True,
                 truncation=False, max_length=None, **kwargs):
        tokens = self.encode(text, add_special_tokens=add_special_tokens)
        if truncation and max_length is not None:
            tokens = tokens[:max_length] if self.truncation_side == "right" else tokens[-max_length:]
        if return_tensors == "pt":
            return {"input_ids": torch.tensor([tokens], dtype=torch.long),
                    "attention_mask": torch.ones((1, len(tokens)), dtype=torch.long)}
        return SimpleNamespace(input_ids=tokens)


class BigramModel:
    """Each vocabulary distribution is determined by the current input token."""

    vocab_size = 16

    def __init__(self, overrides=None, *, nan_at_first_response=False):
        self.transitions = {
            2: {3: 0.001, 10: 0.999},  # Prompt likelihood must not enter scores.
            3: {4: 0.6, 5: 0.3, 9: 0.1},
            4: {6: 0.01, 10: 0.99},
            6: {1: 0.01, 10: 0.99},
            5: {7: 0.9, 10: 0.1},
            7: {8: 0.9, 10: 0.1},
            8: {1: 0.9, 10: 0.1},
            9: {1: 0.5, 10: 0.5},
            1: {0: 1e-25, 10: 1.0},  # Counting right padding would heavily bias scores.
            0: {0: 1e-25, 10: 1.0},
        }
        self.transitions.update(overrides or {})
        self.nan_at_first_response = nan_at_first_response
        self.calls = []

    def __call__(self, input_ids, attention_mask=None, **kwargs):
        self.calls.append({"input_ids": input_ids.detach().clone(),
                           "attention_mask": None if attention_mask is None else attention_mask.detach().clone()})
        logits = torch.full((*input_ids.shape, self.vocab_size), -100.0, device=input_ids.device)
        for batch in range(input_ids.shape[0]):
            for position, current in enumerate(input_ids[batch].tolist()):
                probabilities = self.transitions.get(current, {10: 1.0})
                for following, probability in probabilities.items():
                    logits[batch, position, following] = math.log(probability)
                if self.nan_at_first_response and current == 3:
                    logits[batch, position, 4] = float("nan")
        return SimpleNamespace(logits=logits)


def _inputs(tokenizer):
    return tokenizer(tokenizer.prompt_text, return_tensors="pt")


def _scores(model, tokenizer=None, **kwargs):
    tokenizer = tokenizer or TinyTokenizer()
    return score_canonical_skill_sequences(
        model, tokenizer, _inputs(tokenizer), prompt_text=tokenizer.prompt_text, **kwargs,
    )


def test_complete_sequence_winner_differs_from_greedy_first_token_winner():
    tokenizer = TinyTokenizer()
    model = BigramModel()
    greedy, _ = generate_response(
        model, tokenizer, _inputs(tokenizer), max_new_tokens=4,
        action_constraint=SkillTokenConstraint(tokenizer),
    )
    assert greedy == skill_target(Skill.GO_TO_GOAL)
    scores = _scores(model, tokenizer, max_new_tokens=4)
    assert max(scores, key=scores.get) == Skill.EVADE_PREDATOR.value
    assert scores[Skill.GO_TO_GOAL.value] == pytest.approx(math.log(0.6 * 0.01 * 0.01), abs=1e-6)
    assert scores[Skill.EVADE_PREDATOR.value] == pytest.approx(math.log(0.3 * 0.9 * 0.9 * 0.9), abs=1e-6)
    assert scores[Skill.HOLD_POSITION.value] == pytest.approx(math.log(0.1 * 0.5), abs=1e-6)


def test_different_lengths_use_shifted_response_sum_not_prompt_or_pad_likelihood():
    model = BigramModel()
    scores = _scores(model, max_new_tokens=4)
    # These exact sums catch the prediction shift, counting prompt tokens,
    # omitting EOS, and accidentally including candidate right padding.
    expected = {
        Skill.GO_TO_GOAL.value: math.log(0.6) + math.log(0.01) + math.log(0.01),
        Skill.EVADE_PREDATOR.value: math.log(0.3) + 3 * math.log(0.9),
        Skill.HOLD_POSITION.value: math.log(0.1) + math.log(0.5),
    }
    assert scores == pytest.approx(expected, abs=1e-6)
    assert any(call["attention_mask"] is not None and (call["attention_mask"] == 0).any()
               for call in model.calls), "Different response lengths must mask batch padding"
    assert any(call["input_ids"].shape[0] == 3 for call in model.calls)


def test_scores_are_not_normalized_by_candidate_token_length():
    model = BigramModel({
        3: {4: 0.4, 5: 0.3, 9: 0.3},
        4: {6: 0.5, 10: 0.5}, 6: {1: 1.0},
        5: {7: 0.8, 10: 0.2}, 7: {8: 0.625, 10: 0.375}, 8: {1: 1.0},
        9: {1: 0.1, 10: 0.9},
    })
    scores = _scores(model)
    assert scores == pytest.approx({"go_to_goal": math.log(0.2), "evade_predator": math.log(0.15),
                                    "hold_position": math.log(0.03)}, abs=1e-6)
    assert max(scores, key=scores.get) == Skill.GO_TO_GOAL.value
    lengths = {"go_to_goal": 3, "evade_predator": 4, "hold_position": 2}
    assert max(scores, key=lambda skill: scores[skill] / lengths[skill]) == Skill.EVADE_PREDATOR.value


def test_eos_probability_can_change_the_winning_skill():
    model = BigramModel({
        4: {6: 0.9, 10: 0.1}, 6: {1: 0.01, 10: 0.99},
        5: {7: 0.1, 10: 0.9}, 7: {8: 0.1, 10: 0.9}, 8: {1: 0.1, 10: 0.9},
        9: {1: 0.99, 10: 0.01},
    })
    scores = _scores(model)
    assert 0.6 * 0.9 > 0.1  # GO beats HOLD before termination probability.
    assert max(scores, key=scores.get) == Skill.HOLD_POSITION.value
    assert scores[Skill.HOLD_POSITION.value] == pytest.approx(math.log(0.1 * 0.99), abs=1e-6)


def test_padding_probability_changes_do_not_change_response_scores():
    low_pad = _scores(BigramModel())
    high_pad = _scores(BigramModel({1: {0: 0.999, 10: 0.001}, 0: {0: 0.999, 10: 0.001}}))
    assert high_pad == pytest.approx(low_pad, abs=1e-6)


def test_nonfinite_scored_logits_are_rejected():
    with pytest.raises(FloatingPointError):
        _scores(BigramModel(nan_at_first_response=True))


def test_nonfinite_padding_only_logits_are_not_scored():
    class NonfinitePaddingModel(BigramModel):
        def __call__(self, input_ids, attention_mask=None, **kwargs):
            result = super().__call__(input_ids, attention_mask=attention_mask, **kwargs)
            result.logits[attention_mask == 0] = float("nan")
            return result

    assert _scores(NonfinitePaddingModel()) == pytest.approx(_scores(BigramModel()), abs=1e-6)


def test_max_new_tokens_includes_eos_and_never_truncates_a_candidate():
    with pytest.raises(ValueError):
        _scores(BigramModel(), max_new_tokens=3)  # EVADE requires three tokens plus EOS.


def test_prefix_inconsistent_tokenization_is_rejected_before_model_scoring():
    class BadBoundaryTokenizer(TinyTokenizer):
        def encode(self, text, add_special_tokens=False):
            tokens = super().encode(text, add_special_tokens=add_special_tokens)
            if text.startswith(self.prompt_text) and text.endswith(self.eos_token):
                return [*tokens[:-1], 11, tokens[-1]]
            return tokens

    model = BigramModel()
    with pytest.raises(ValueError):
        _scores(model, BadBoundaryTokenizer())
    assert model.calls == []


def _planner(mode):
    # Bypass only heavyweight construction; exercise the real plan_context.
    planner = MiniMindSkillPlanner.__new__(MiniMindSkillPlanner)
    planner.device = torch.device("cpu")
    planner.model = BigramModel()
    planner.tokenizer = TinyTokenizer()
    planner.constraint = SkillTokenConstraint(planner.tokenizer)
    planner.max_seq_len = 64
    planner.max_new_tokens = 4
    planner.instruction_split = "train"
    planner.ablation = None
    planner.decoding_mode = mode
    planner.last_response = None
    planner.last_valid = True
    planner.last_scores = None
    return planner


def _context():
    return SimpleNamespace(
        serialize=lambda: "toy context identity",
        payload=lambda: {"schema_version": "toy-context", "current": {"predator_visible": False}, "history": {}},
    )


def test_planner_keeps_greedy_trie_as_default_and_old_greedy_behavior():
    default = inspect.signature(MiniMindSkillPlanner.__init__).parameters["decoding_mode"].default
    assert default == "greedy_trie"
    planner = _planner(default)
    decision = planner.plan_context(_context(), Preference.SURVIVAL_FIRST)
    assert decision.skill == Skill.GO_TO_GOAL
    assert planner.last_valid is True
    assert planner.last_response == skill_target(Skill.GO_TO_GOAL)
    assert planner.last_scores is None


def test_planner_sequence_mode_uses_full_sequence_scores_and_canonical_json():
    planner = _planner("sequence_logprob")
    decision = planner.plan_context(_context(), Preference.SURVIVAL_FIRST)
    assert decision.skill == Skill.EVADE_PREDATOR
    assert planner.last_valid is True
    assert planner.last_response == skill_target(Skill.EVADE_PREDATOR)
    assert set(planner.last_scores) == {skill.value for skill in Skill}
    assert max(planner.last_scores, key=planner.last_scores.get) == decision.skill.value


def test_planner_nonfinite_sequence_scores_raise_without_a_silent_skill_fallback():
    planner = _planner("sequence_logprob")
    planner.model = BigramModel(nan_at_first_response=True)
    planner.last_response = skill_target(Skill.HOLD_POSITION)
    planner.last_scores = {Skill.HOLD_POSITION.value: 0.0}
    with pytest.raises(FloatingPointError):
        planner.plan_context(_context(), Preference.SURVIVAL_FIRST)
    assert planner.last_valid is False
    assert planner.last_response is None
    assert planner.last_scores is None
