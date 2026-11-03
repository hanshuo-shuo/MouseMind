from __future__ import annotations

import json
import math

import pytest

from mouse_llm.continuous.contracts import ACTION_SCHEMA
from mouse_llm.continuous.promotion import (
    DEFAULT_CONTRACT,
    EVALUATION_SCHEMA,
    SUMMARY_FIELDS,
    _sha256,
    evaluate_promotion,
    main,
    write_report,
)


def episode(index, *, policy, clean_count, success_count=200, captured_count=1):
    success = int(index < success_count)
    captures = 0 if index < clean_count else captured_count
    return {
        "policy": policy, "seed": 52000 + index, "success": success,
        "clean_success": int(success and captures == 0), "captures": captures,
        "captured": int(captures > 0), "survived": int(captures == 0),
        "steps": 30, "episode_return": float(success - captures),
        "path_length": 0.8, "valid_action_rate": 1.0,
        "latency_mean_seconds": 0.001, "deadline_miss_rate": 0.0,
        "destination_rejection_rate": 0.0,
        "requested_invalid_rate": 0.2, "projection_rate": 0.2,
        "mean_projection_distance": 0.01,
    }


def save_fixture(directory, *, policy, clean_count, success_count=200, captured_count=1, reverse=False):
    # These constructed files test gate behavior only, not research performance.
    rows = [episode(index, policy=policy, clean_count=clean_count,
                    success_count=success_count, captured_count=captured_count) for index in range(200)]
    if reverse:
        rows.reverse()
    summary = {
        "schema_version": EVALUATION_SCHEMA, "action_schema": ACTION_SCHEMA,
        "context_schema": "mousemind_continuous_context_v1", "research_evidence": True,
        "policy": policy, "seed_pool": "final_id_test", "episodes": 200,
        "seeds": [row["seed"] for row in rows],
        "metrics": {name: math.fsum(row[field] for row in rows) / len(rows) for name, field in SUMMARY_FIELDS.items()},
        "config": {
            "world": "21_05", "max_steps": 300, "time_step": 0.25,
            "predator_speed_ratio": 0.15, "max_line_of_sight_distance": 1.0,
            "navigation_contract": "graph_route_exact_xy_terminal_project_v2",
            "reset_contract": "fresh_navigation_episode_reset_v1",
            "destination_mode": "project", "projection_epsilon": 1e-5,
            "reuse_environment": False,
            "complete_seed_pool": True, "contract_sha256": _sha256(DEFAULT_CONTRACT),
        },
    }
    directory.mkdir()
    rewrite(directory, summary, rows)
    return summary, rows


def rewrite(directory, summary, rows):
    (directory / "metrics.json").write_text(json.dumps(summary))
    (directory / "episodes.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))


@pytest.fixture
def pair(tmp_path):
    reference, candidate = tmp_path / "reference", tmp_path / "candidate"
    ref_data = save_fixture(reference, policy="reference-policy", clean_count=40)
    cand_data = save_fixture(candidate, policy="candidate-policy", clean_count=80, reverse=True)
    return reference, candidate, ref_data, cand_data


def test_superior_complete_pair_is_eligible_not_automatically_promoted(pair):
    reference, candidate, _, _ = pair
    report = evaluate_promotion(reference, candidate, bootstrap_replicates=500, bootstrap_seed=11)
    assert report["status"] == "eligible_for_manual_review"
    assert all(report["criteria"].values())
    assert report["candidate_minus_reference"]["clean_success_rate"]["delta"] == pytest.approx(0.2)
    assert report["candidate_minus_reference"]["success_rate"]["ci_low"] == 0.0
    assert report["candidate_minus_reference"]["captures_per_episode"]["ci_high"] < 0.0
    assert report["automatic_commit"] is report["automatic_push"] is report["automatic_publication"] is False
    assert report["inference_limits"]["training_randomness_covered"] is False
    assert report == evaluate_promotion(reference, candidate, bootstrap_replicates=500, bootstrap_seed=11)


@pytest.mark.parametrize("clean_count,success_count,captured_count,failed", [
    (40, 200, 1, "clean_success_superiority"),
    (80, 120, 1, "task_success_noninferiority"),
    (80, 200, 2, "captures_nonincrease"),
])
def test_each_scientific_criterion_is_required(tmp_path, clean_count, success_count, captured_count, failed):
    reference, candidate = tmp_path / "reference", tmp_path / "candidate"
    save_fixture(reference, policy="reference", clean_count=40)
    save_fixture(candidate, policy="candidate", clean_count=clean_count,
                 success_count=success_count, captured_count=captured_count)
    report = evaluate_promotion(reference, candidate, bootstrap_replicates=300)
    assert report["status"] == "not_promoted"
    assert failed in report["failed_criteria"]


@pytest.mark.parametrize("mutation", ["not_research", "synthetic", "mock", "action", "schema"])
def test_mock_or_incompatible_evidence_is_rejected(pair, mutation):
    reference, candidate, _, (summary, rows) = pair
    if mutation == "not_research":
        summary["research_evidence"] = False
    elif mutation in {"synthetic", "mock"}:
        summary[mutation] = True
    elif mutation == "action":
        summary["action_schema"] = "old_discrete_schema"
    else:
        summary["schema_version"] = "old_report_schema"
    rewrite(candidate, summary, rows)
    with pytest.raises(ValueError):
        evaluate_promotion(reference, candidate, bootstrap_replicates=10)


@pytest.mark.parametrize("mutation", ["missing_row", "duplicate_row", "old_100", "summary_seed_duplicate"])
def test_seed_coverage_must_match_exact_new_final_pool(pair, mutation):
    reference, candidate, _, (summary, rows) = pair
    if mutation == "missing_row":
        rows.pop()
    elif mutation == "duplicate_row":
        rows.append(rows[0].copy())
    elif mutation == "old_100":
        rows = rows[:100]
        for index, row in enumerate(rows):
            row["seed"] = 30000 + index
        summary["episodes"] = 100
        summary["seeds"] = [row["seed"] for row in rows]
    else:
        summary["seeds"][0] = summary["seeds"][1]
    rewrite(candidate, summary, rows)
    with pytest.raises(ValueError):
        evaluate_promotion(reference, candidate, bootstrap_replicates=10)


@pytest.mark.parametrize("field,value", [
    ("predator_speed_ratio", 0.2), ("time_step", 0.5),
    ("max_line_of_sight_distance", 0.7),
    ("navigation_contract", "different_navigation_v2"),
    ("reset_contract", "different_reset_v2"),
    ("destination_mode", "reject"), ("projection_epsilon", 1e-4),
    ("reuse_environment", True),
    ("contract_sha256", "old-contract-hash"),
])
def test_environment_and_contract_must_be_identical_and_registered(pair, field, value):
    reference, candidate, _, (summary, rows) = pair
    summary["config"][field] = value
    rewrite(candidate, summary, rows)
    with pytest.raises(ValueError):
        evaluate_promotion(reference, candidate, bootstrap_replicates=10)


@pytest.mark.parametrize("mutation", ["stale_metric", "nan_metric", "infinite_row", "wrong_clean", "negative_captures", "wrong_policy", "negative_projection", "invalid_projection_rate", "stale_projection_metric"])
def test_stale_inconsistent_or_nonfinite_reports_are_rejected(pair, mutation):
    reference, candidate, _, (summary, rows) = pair
    if mutation == "stale_metric":
        summary["metrics"]["clean_success_rate"] = 0.99
    elif mutation == "nan_metric":
        summary["metrics"]["mean_return"] = float("nan")
    elif mutation == "infinite_row":
        rows[0]["path_length"] = float("inf")
    elif mutation == "wrong_clean":
        rows[0]["clean_success"] = 1
    elif mutation == "negative_captures":
        rows[0]["captures"] = -1
    elif mutation == "negative_projection":
        rows[0]["mean_projection_distance"] = -0.1
    elif mutation == "invalid_projection_rate":
        rows[0]["projection_rate"] = 1.1
    elif mutation == "stale_projection_metric":
        summary["metrics"]["requested_invalid_rate"] = 0.0
    else:
        rows[0]["policy"] = "other-checkpoint"
    rewrite(candidate, summary, rows)
    with pytest.raises(ValueError):
        evaluate_promotion(reference, candidate, bootstrap_replicates=10)


def test_old_reject_telemetry_cannot_enter_new_project_gate(pair):
    reference, candidate, _, (summary, rows) = pair
    for name in ("requested_invalid_rate", "projection_rate", "mean_projection_distance"):
        summary["metrics"].pop(name)
        for row in rows:
            row.pop(name)
    rewrite(candidate, summary, rows)
    with pytest.raises(ValueError):
        evaluate_promotion(reference, candidate, bootstrap_replicates=10)


def test_missing_projection_configuration_is_rejected(pair):
    reference, candidate, _, (summary, rows) = pair
    summary["config"].pop("destination_mode")
    rewrite(candidate, summary, rows)
    with pytest.raises(ValueError, match="missing"):
        evaluate_promotion(reference, candidate, bootstrap_replicates=10)


def test_explicit_reject_contract_requires_both_reports_to_use_reject_v2(pair, tmp_path):
    reference, candidate, reference_data, candidate_data = pair
    contract = json.loads(DEFAULT_CONTRACT.read_text())
    contract.update(destination_mode="reject", navigation_contract="graph_route_exact_xy_terminal_reject_v2")
    path = tmp_path / "reject-contract.json"
    path.write_text(json.dumps(contract))
    for directory, (summary, rows) in ((reference, reference_data), (candidate, candidate_data)):
        summary["config"].update(destination_mode="reject", navigation_contract="graph_route_exact_xy_terminal_reject_v2", contract_sha256=_sha256(path))
        for row in rows:
            row.update(projection_rate=0.0, mean_projection_distance=0.0,
                       destination_rejection_rate=row["requested_invalid_rate"])
        for name, field in SUMMARY_FIELDS.items():
            summary["metrics"][name] = math.fsum(row[field] for row in rows) / len(rows)
        rewrite(directory, summary, rows)
    report = evaluate_promotion(reference, candidate, contract_path=path, bootstrap_replicates=100)
    assert report["environment"]["destination_mode"] == "reject"
    assert report["status"] == "eligible_for_manual_review"


def test_cli_defaults_to_readonly_aggregate_output(pair, monkeypatch, capsys):
    reference, candidate, _, _ = pair
    before = {str(path): path.read_bytes() for directory in (reference, candidate) for path in directory.iterdir()}
    monkeypatch.setattr("sys.argv", ["promotion", "--reference-dir", str(reference), "--candidate-dir", str(candidate), "--bootstrap-replicates", "20"])
    main()
    report = json.loads(capsys.readouterr().out)
    assert report["paired_episodes"] == 200
    after = {str(path): path.read_bytes() for directory in (reference, candidate) for path in directory.iterdir()}
    assert before == after


def test_explicit_private_output_cannot_overwrite_evidence(pair, tmp_path):
    reference, candidate, _, _ = pair
    report = evaluate_promotion(reference, candidate, bootstrap_replicates=20)
    with pytest.raises(ValueError, match="overwrite"):
        write_report(candidate / "metrics.json", report, input_dirs=(reference, candidate))
    path = tmp_path / "private" / "promotion.json"
    write_report(path, report, input_dirs=(reference, candidate))
    assert json.loads(path.read_text()) == report
    assert path.stat().st_mode & 0o777 == 0o600
    with pytest.raises(FileExistsError):
        write_report(path, report, input_dirs=(reference, candidate))
