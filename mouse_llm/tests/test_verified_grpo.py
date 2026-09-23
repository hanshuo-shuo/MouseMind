"""Focused checks for verified GRPO semantics and leakage guards."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest
import torch

from mouse_llm.data.planner_schema import Preference, utility_scores
from mouse_llm.hierarchical.counterfactual import BranchOutcome
from mouse_llm.hierarchical.policy import Skill
from mouse_llm.training.train_verified_grpo import adapt_kl_coefficient, anchor_id, grpo_loss, group_advantages, load_train_anchors
from mouse_llm.verified_rl.reward import verified_reward
from mouse_llm.verified_rl.rollout import ReplayVerificationError, RolloutVerifier, _VerifiedBranchEnv
from mouse_llm.verified_rl.sampler import sample_skills, score_skill_distributions


def test_reward_matches_existing_utility():
    outcomes = {
        skill.value: BranchOutcome(
            skill=skill.value, horizon=8, steps_executed=8,
            capture_within_h=int(skill == Skill.GO_TO_GOAL),
            captures=int(skill == Skill.GO_TO_GOAL),
            goal_distance_change=float(index), predator_exposure_steps=index,
            path_length=0.1 * index, near_occlusion_exposure=index,
            terminal_goal_event=int(skill == Skill.GO_TO_GOAL),
        )
        for index, skill in enumerate(Skill)
    }
    for preference in (Preference.SURVIVAL_FIRST, Preference.BALANCED, Preference.GOAL_FIRST):
        scores = utility_scores({skill: outcome.__dict__ for skill, outcome in outcomes.items()}, preference)
        for skill, outcome in outcomes.items():
            assert verified_reward(outcome, preference) == pytest.approx(scores[skill])
    with pytest.raises(ValueError, match="HOLD"):
        verified_reward(outcomes[Skill.HOLD_POSITION.value], Preference.HOLD)


def test_group_advantage_and_clipping():
    assert group_advantages([1.0] * 6) is None
    advantages = group_advantages([1, 1, 1, 2, 2, 2])
    assert advantages is not None
    assert float(advantages.mean()) == pytest.approx(0, abs=1e-6)
    assert float(advantages.std(unbiased=False)) == pytest.approx(1, abs=1e-5)
    current = torch.log_softmax(torch.tensor([0.2, 0.1, -0.2], requires_grad=True), dim=0)
    old = current.detach()[torch.tensor([0, 1, 1, 2, 2, 2])]
    loss, kl = grpo_loss(current[torch.tensor([0, 1, 1, 2, 2, 2])], old, advantages, current, current.detach(), clip_epsilon=0.2, kl_coefficient=0.01)
    assert float(kl.detach()) == pytest.approx(0)
    loss.backward()
    assert current.grad_fn is not None


def test_adaptive_kl_grows_when_policy_drifts():
    assert adapt_kl_coefficient(0.2, observed_kl=0.2, target_kl=0.05, initial_coefficient=0.2) == pytest.approx(0.3)
    assert adapt_kl_coefficient(0.3, observed_kl=0.001, target_kl=0.05, initial_coefficient=0.2) == pytest.approx(0.2)
    assert adapt_kl_coefficient(0.2, observed_kl=0.2, target_kl=0, initial_coefficient=0.2) == pytest.approx(0.2)


def test_grpo_clipping_limits_reused_group_update():
    current = torch.tensor([0.9, 0.1]).log()
    old = torch.tensor([0.5, 0.5]).log()
    distribution = torch.tensor([0.9, 0.1, 1e-9]).log()
    loss, _ = grpo_loss(
        current, old, torch.tensor([1.0, -1.0]),
        distribution, distribution,
        clip_epsilon=0.2, kl_coefficient=0.0,
    )
    assert float(loss) == pytest.approx(-0.2)


def test_duplicate_samples_roll_out_once(monkeypatch):
    calls = []
    monkeypatch.setattr("mouse_llm.verified_rl.rollout.verify_anchor_replay", lambda *args, **kwargs: {"verified": True})

    def fake_branch(*args, **kwargs):
        skill = kwargs["skill"]
        calls.append(skill)
        return BranchOutcome(skill.value, 8, 8, 0, 0, 0.1, 0, 0.1, 0, 0)

    monkeypatch.setattr("mouse_llm.verified_rl.rollout.branch_skill", fake_branch)
    verifier = RolloutVerifier(lambda: None, None, None)
    anchor = {"seed": 10000, "prefix_actions": [1]}
    outcomes, rewards = verifier.evaluate_group(anchor, [Skill.GO_TO_GOAL, Skill.EVADE_PREDATOR, Skill.EVADE_PREDATOR, Skill.GO_TO_GOAL], "balanced")
    assert len(calls) == 2
    assert outcomes[1] is outcomes[2]
    assert rewards[1] == rewards[2]


def test_replay_retry_requires_an_exact_match(monkeypatch):
    attempts = iter(({"verified": False, "environment_max_abs_error": 2e-5}, {"verified": True}))
    monkeypatch.setattr("mouse_llm.verified_rl.rollout.verify_anchor_replay", lambda *args, **kwargs: next(attempts))
    verifier = RolloutVerifier(lambda: None, None, None)
    verifier.verify_anchor({"seed": 10000, "prefix_actions": []})
    assert verifier.last_replay_attempts == 2


def test_each_branch_checks_its_own_prefix_state():
    class FakeEnv:
        def reset(self, **kwargs):
            return [0.0], {}

        def step(self, action):
            return [0.1], 0.0, False, False, {}

        def legacy_policy_observation(self):
            return [0.1]

    anchor = {
        "prefix_actions": [1],
        "environment_observation": [0.0],
        "legacy_observation": [0.0],
    }
    env = _VerifiedBranchEnv(FakeEnv(), anchor, 1e-7)
    env.reset(seed=10000)
    with pytest.raises(ReplayVerificationError, match="Branch prefix state mismatch"):
        env.step(1)


def test_training_source_rejects_final_seed(tmp_path):
    anchor = {"seed": 30000, "step_index": 0, "prefix_actions": []}
    anchors = tmp_path / "anchors.jsonl"
    verified = tmp_path / "verified.jsonl"
    anchors.write_text(json.dumps(anchor) + "\n")
    verified.write_text(json.dumps({"anchor_id": anchor_id(anchor), "seed": 30000, "step_index": 0, "replay_verification": {"verified": True}, "branches_by_horizon": {"8": {}}}) + "\n")
    with pytest.raises(ValueError, match="outside permitted"):
        load_train_anchors(anchors, verified, __import__("mouse_llm.evaluation.contracts", fromlist=["DEFAULT_P2_CONTRACT"]).DEFAULT_P2_CONTRACT, source_pool="data_collection", horizon=8)


def test_constrained_sampling_does_not_favor_short_json():
    class Characters:
        eos_token_id = 0
        pad_token_id = 0

        def encode(self, text, add_special_tokens=True):
            return [ord(character) + 1 for character in text]

    class FlatModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.bias = torch.nn.Parameter(torch.zeros(256))

        def forward(self, ids):
            return SimpleNamespace(logits=self.bias.expand(*ids.shape, -1))

    tokenizer = Characters()
    model = FlatModel()
    raw, constrained = score_skill_distributions(model, tokenizer, "prompt")
    assert raw[0] != raw[1]  # Full sequences have different lengths.
    assert torch.allclose(constrained.exp(), torch.full((3,), 1 / 3), atol=1e-6)
    samples, probabilities = sample_skills(model, tokenizer, "prompt", group_size=6)
    assert len(samples) == 6
    assert all(probability == pytest.approx(1 / 3) for probability in probabilities.values())
