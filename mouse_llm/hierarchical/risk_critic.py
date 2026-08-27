"""Compact calibrated candidate-skill capture-risk verifier."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch import nn

from mouse_llm.hierarchical.context import PlannerContext
from mouse_llm.hierarchical.policy import Skill


SKILLS = tuple(Skill)


def skill_one_hot(skill: Skill) -> list[float]:
    return [float(skill == item) for item in SKILLS]


class RiskCriticMLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dims: tuple[int, ...] = (128, 64)):
        super().__init__()
        layers: list[nn.Module] = []
        previous = input_dim
        for width in hidden_dims:
            layers.extend((nn.Linear(previous, width), nn.ReLU()))
            previous = width
        layers.append(nn.Linear(previous, 1))
        self.network = nn.Sequential(*layers)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.network(features).squeeze(-1)


class RuntimeRiskCritic:
    def __init__(self, checkpoint: Path, *, device: str = "cpu"):
        payload = torch.load(checkpoint, map_location=device, weights_only=False)
        if payload.get("schema_version") != 1:
            raise ValueError(f"Unsupported risk critic checkpoint: {checkpoint}")
        self.device = torch.device(device)
        self.context_dim = int(payload["context_dim"])
        self.input_dim = self.context_dim + len(SKILLS)
        self.model = RiskCriticMLP(
            self.input_dim, tuple(int(value) for value in payload["hidden_dims"])
        ).to(self.device)
        self.model.load_state_dict(payload["model_state"])
        self.model.eval()
        self.mean = payload["feature_mean"].to(self.device)
        self.std = payload["feature_std"].to(self.device)
        self.temperature = float(payload["temperature"])
        self.recommended_threshold = float(payload["recommended_threshold"])

    def score(self, context: PlannerContext, skill: Skill) -> float:
        features = np.asarray(
            [*context.numeric().tolist(), *skill_one_hot(skill)], dtype=np.float32
        )
        if len(features) != self.input_dim:
            raise ValueError(
                f"Risk critic expects {self.input_dim} features, got {len(features)}"
            )
        tensor = torch.from_numpy(features).to(self.device)
        with torch.inference_mode():
            logit = self.model(((tensor - self.mean) / self.std).unsqueeze(0))
            probability = torch.sigmoid(logit / self.temperature)
        return float(probability.cpu())
