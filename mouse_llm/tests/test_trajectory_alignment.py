from __future__ import annotations

import numpy as np
import pytest

from mouse_llm.evaluation.analyze_trajectory_alignment import (
    FEATURES,
    analyze,
    profile_distance,
    render_markdown,
)


def test_profile_distance_is_equal_weight_mean_absolute_gap():
    source = np.zeros((2, len(FEATURES)), dtype=np.float64)
    policy = np.zeros((2, len(FEATURES)), dtype=np.float64)
    policy[:, 0] = 1.0
    assert profile_distance(source, policy) == 1.0 / len(FEATURES)


def test_analysis_uses_paired_policy_bootstrap_for_comparisons():
    source = np.zeros((4, len(FEATURES)), dtype=np.float64)
    proposed = np.full((3, len(FEATURES)), 0.1, dtype=np.float64)
    reference = np.full((3, len(FEATURES)), 0.4, dtype=np.float64)
    policies, comparisons = analyze(
        source,
        {
            "minimind-learned": proposed,
            "direct-minimind-lora": reference,
        },
        seed=9,
        bootstrap_iterations=100,
    )
    assert policies["minimind-learned"]["alignment_distance"][
        "mean"
    ] == pytest.approx(0.1)
    delta = comparisons["minimind-learned_minus_direct-minimind-lora"]
    assert delta["mean"] == pytest.approx(-0.3)
    assert delta["ci_high"] < 0.0


def test_markdown_positions_minimind_and_upper_reference_honestly():
    def summary(role: str, mean: float):
        return {
            "role": role,
            "alignment_distance": {
                "mean": mean,
                "ci_low": mean - 0.01,
                "ci_high": mean + 0.01,
            },
        }

    markdown = render_markdown(
        {
            "artifact": "mousemind_trajectory_profile_alignment",
            "source": {"held_out_episode_count": 500},
            "policies": {
                "minimind-learned": summary("proposed MiniMind hierarchy", 0.3),
                "numeric-learned": summary(
                    "non-language upper reference", 0.2
                ),
            },
        }
    )
    assert "**MiniMind hierarchy (full)**" in markdown
    assert "non-language upper reference" in markdown
    assert "than evidence that MiniMind is the unconstrained overall winner" in markdown
