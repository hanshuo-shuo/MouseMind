from __future__ import annotations

from mouse_llm.evaluation.mine_failures import build_manifest


def test_failure_manifest_selects_most_severe_per_mode():
    rows = [
        {
            "policy": "mlp-bc",
            "seed": "10",
            "failure_mode": "capture_near_occlusion",
            "captures": "2",
            "episode_return": "-2",
            "first_predator_visible_step": "5",
            "first_capture_step": "9",
            "oscillation_score": "0",
            "recent_path_length": "1",
            "goal_distance_end": "0.5",
            "goal_distance_min": "0.4",
        },
        {
            "policy": "mlp-bc",
            "seed": "11",
            "failure_mode": "capture_near_occlusion",
            "captures": "5",
            "episode_return": "-5",
            "first_predator_visible_step": "4",
            "first_capture_step": "6",
            "oscillation_score": "0",
            "recent_path_length": "1",
            "goal_distance_end": "0.5",
            "goal_distance_min": "0.4",
        },
    ]
    source = {
        "metadata": {"research_evidence": True, "seed_sha256": "paired"}
    }
    manifest = build_manifest(
        rows, policy="mlp-bc", per_mode=1, source_metrics=source
    )
    assert manifest["selected_episode_count"] == 1
    assert manifest["replay_queue"][0]["seed"] == 11
    assert manifest["research_evidence"] is True
