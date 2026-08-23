from __future__ import annotations

import json

import pytest

torch = pytest.importorskip("torch")

from mouse_llm.baselines.mlp_bc import MLPPolicy, load_supervised_jsonl
from mouse_llm.data.schema import make_conversation
from mouse_llm.evaluation.evaluate_policy import load_samples


def test_mlp_is_sub_million_parameter_baseline():
    model = MLPPolicy()
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    assert parameter_count < 1_000_000
    assert model(torch.zeros(2, 10)).shape == (2, 295)


def test_mlp_loader_uses_versioned_conversation_contract(tmp_path):
    path = tmp_path / "train.jsonl"
    samples = [
        make_conversation([float(index)] * 10, index, action_count=295)
        for index in range(3)
    ]
    path.write_text(
        "".join(json.dumps(sample) + "\n" for sample in samples), encoding="utf-8"
    )
    split = load_supervised_jsonl(path, action_count=295)
    assert split.observations.shape == (3, 10)
    assert split.actions.tolist() == [0, 1, 2]

    mlp_subset = load_supervised_jsonl(
        path, action_count=295, max_samples=2, seed=7
    )
    llm_subset = load_samples(path, max_samples=2, seed=7)
    assert mlp_subset.actions.tolist() == [
        sample.target_action for sample in llm_subset
    ]
