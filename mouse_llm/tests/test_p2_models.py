import numpy as np
import pytest

torch = pytest.importorskip("torch")

from mouse_llm.baselines.planner_mlp import PlannerMLP
from mouse_llm.hierarchical.minimind_planner import SkillTokenConstraint
from mouse_llm.hierarchical.policy import Skill
from mouse_llm.hierarchical.risk_critic import RiskCriticMLP
from mouse_llm.training.train_risk_critic import auprc, auroc, calibration_metrics


class _CharacterTokenizer:
    eos_token_id = 999

    @staticmethod
    def encode(text, add_special_tokens=False):
        assert add_special_tokens is False
        return [ord(character) for character in text]


def test_numeric_planner_and_risk_critic_shapes_are_compact():
    planner = PlannerMLP(73)
    critic = RiskCriticMLP(72)
    assert planner(torch.zeros(4, 73)).shape == (4, 3)
    assert critic(torch.zeros(4, 72)).shape == (4,)
    assert sum(value.numel() for value in planner.parameters()) < 100_000
    assert sum(value.numel() for value in critic.parameters()) < 100_000


def test_skill_constraint_allows_only_three_canonical_json_targets():
    constraint = SkillTokenConstraint(_CharacterTokenizer())
    allowed = constraint.prefix_allowed_tokens_fn(prompt_length=1)
    prefix = '{"skill":"'
    next_tokens = allowed(0, np.asarray([0, *map(ord, prefix)]))
    assert next_tokens == sorted({ord(skill.value[0]) for skill in Skill})


def test_risk_metrics_report_discrimination_and_calibration():
    targets = np.asarray([0, 0, 1, 1], dtype=np.float32)
    probabilities = np.asarray([0.1, 0.2, 0.8, 0.9], dtype=np.float32)
    assert auroc(targets, probabilities) == 1.0
    assert auprc(targets, probabilities) == 1.0
    metrics = calibration_metrics(targets, probabilities, bins=2)
    assert 0.0 <= metrics["ece"] <= 1.0
    assert metrics["brier_score"] < 0.05
