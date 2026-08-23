from __future__ import annotations

import csv
import json

from mouse_llm.evaluation.merge_closed_loop import merge_runs


def _write_run(path, *, policy, value):
    path.mkdir()
    metadata = {
        "environment": "BotEvadeEnv",
        "world": "21_05",
        "episode_count": 2,
        "seed_start": 10,
        "seed_sha256": "same",
        "max_steps": 3,
        "control_budget_seconds": 0.25,
        "research_evidence": True,
    }
    report = {
        "experiment": "mousemind_seeded_closed_loop",
        "metadata": metadata,
        "policies": {policy: {"episode_count": 2}},
        "failure_taxonomy": {"definitions_version": 1, "definitions": {}},
        "system": {},
    }
    (path / "closed_loop_metrics.json").write_text(
        json.dumps(report), encoding="utf-8"
    )
    rows = []
    for seed in (10, 11):
        rows.append(
            {
                "policy": policy,
                "seed": seed,
                "episode_return": value,
                "success": value,
                "captured": 0,
                "captures": 0,
                "survived": 1,
                "steps": 3,
                "path_length": value,
                "path_efficiency": value,
                "valid_action_rate": 1,
                "deadline_miss_rate": 0,
            }
        )
    with (path / "closed_loop_episodes.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def test_merge_reconstructs_paired_policy_delta(tmp_path):
    random_dir = tmp_path / "random-run"
    mlp_dir = tmp_path / "mlp-run"
    _write_run(random_dir, policy="random", value=0)
    _write_run(mlp_dir, policy="mlp-bc", value=1)
    report, rows = merge_runs(
        [random_dir, mlp_dir], reference_policy="random", seed=42
    )
    assert set(report["policies"]) == {"random", "mlp-bc"}
    paired = report["paired_comparisons"]["mlp-bc_minus_random"]
    assert paired["success_rate"]["mean"] == 1.0
    assert len(rows) == 4
