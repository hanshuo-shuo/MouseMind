from __future__ import annotations

import json

import numpy as np
import pytest

from mouse_llm.evaluation.analyze_mouse_alignment import (
    AlignmentSample,
    load_direct_predictions,
    render_markdown,
    summarize_alignment,
)


def _sample(sample_id: str, action: int, *, visible: bool) -> AlignmentSample:
    observation = (
        0.1,
        0.2,
        0.0,
        0.7 if visible else 0.0,
        0.8 if visible else 0.0,
        0.0,
        0.5,
        0.0,
        0.0,
        0.0,
    )
    return AlignmentSample(sample_id, observation, action)


def test_alignment_summary_separates_visibility_and_penalizes_invalid():
    samples = [
        _sample("a", 0, visible=False),
        _sample("b", 0, visible=False),
        _sample("c", 1, visible=True),
        _sample("d", 1, visible=True),
    ]
    destinations = np.asarray(((0.0, 0.0), (1.0, 0.0)), dtype=np.float64)
    summary = summarize_alignment(
        samples,
        [0, 1, 1, None],
        destinations,
        seed=7,
        bootstrap_iterations=100,
    )

    assert summary["overall"]["exact_action_agreement"]["mean"] == 0.5
    assert summary["predator_hidden"]["exact_action_agreement"]["mean"] == 0.5
    assert summary["predator_visible"]["exact_action_agreement"]["mean"] == 0.5
    assert summary["predator_visible"]["valid_output_rate"]["mean"] == 0.5
    expected_error = (0.0 + 1.0 / np.sqrt(2.0) + 0.0 + 1.0) / 4.0
    assert summary["overall"]["normalized_destination_error"]["mean"] == pytest.approx(
        expected_error
    )
    js = summary["overall"]["action_distribution_js_divergence_bits"]
    assert 0.0 < js["mean"] < 1.0


def test_direct_prediction_join_requires_exact_frozen_sample_set(tmp_path):
    samples = [_sample("a", 0, visible=False), _sample("b", 1, visible=True)]
    path = tmp_path / "predictions.jsonl"
    row = {
        "sample_id": "a",
        "base": {"sample_id": "a", "target_action": 0, "predicted_action": 0},
        "lora": {"sample_id": "a", "target_action": 0, "predicted_action": 0},
    }
    path.write_text(json.dumps(row) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="missing=1, extra=0"):
        load_direct_predictions(path, samples)


def test_markdown_renders_a_separate_alignment_table():
    metric = {"mean": 0.5, "ci_low": 0.25, "ci_high": 0.75}
    summary = {
        "overall": {
            "sample_count": 4,
            "exact_action_agreement": metric,
            "normalized_destination_error": metric,
            "action_distribution_js_divergence_bits": {
                "mean": 0.125,
                "ci_low": 0.1,
                "ci_high": 0.2,
            },
        },
        "predator_hidden": {"sample_count": 2, "exact_action_agreement": metric},
        "predator_visible": {"sample_count": 2, "exact_action_agreement": metric},
    }
    markdown = render_markdown(
        {
            "artifact": "mousemind_source_trajectory_alignment",
            "policies": {"mlp-bc": summary},
        }
    )

    assert "# Source mouse trajectory alignment" in markdown
    assert "Predator hidden (N=2)" in markdown
    assert "| MLP BC (low-level upper reference) | 50.0% (25.0–75.0%)" in markdown
    assert "biological mouse behavior" in markdown
