"""Constructed artifact fixtures validate merging, not real policy performance."""

from __future__ import annotations

import json

import pytest

from mouse_llm.continuous.evaluation import DEFAULT_CONTRACT, summarize
from mouse_llm.continuous.merge_evaluations import merge_evaluations
from mouse_llm.continuous.promotion import evaluate_promotion


def config():
    return {
        "world": "21_05", "max_steps": 300, "time_step": 0.25,
        "predator_speed_ratio": 0.15, "max_line_of_sight_distance": 1.0,
        "navigation_contract": "graph_route_exact_xy_terminal_project_v2",
        "reset_contract": "fresh_navigation_episode_reset_v1",
        "destination_mode": "project", "projection_epsilon": 1e-5,
        "complete_seed_pool": False, "reuse_environment": False,
        "planner_horizon": 8, "memory_steps": 8, "evade_distance": 0.35,
        "preference": "survival_first", "instruction_split": "train",
        "specialist_sha256": "a" * 64, "teacher_sha256": None,
        "skill_lora_sha256": None, "base_weight_sha256": None,
        "planner_manifest_sha256": None, "planner_max_seq_len": 1024,
        "legacy_planner_diagnostic": False, "catalog_constrained_reference": False,
    }


def row(seed, *, start, clean_count, policy="continuous-bc"):
    captures = int(seed - start >= clean_count)
    return {
        "policy": policy, "seed": seed, "success": 1,
        "clean_success": int(captures == 0), "captures": captures,
        "captured": captures, "survived": int(captures == 0),
        "steps": 20, "episode_return": 1.0 - captures, "path_length": 0.9,
        "valid_action_rate": 1.0, "latency_mean_seconds": 0.003,
        "deadline_miss_rate": 0.0, "destination_rejection_rate": 0.0,
        "requested_invalid_rate": 0.2, "projection_rate": 0.2,
        "mean_projection_distance": 0.01,
        "skill_counts": {"go_to_goal": 20},
    }


def rewrite(directory, summary, rows):
    (directory / "metrics.json").write_text(json.dumps(summary))
    (directory / "episodes.jsonl").write_text("".join(json.dumps(item) + "\n" for item in rows))


def make_shards(root, *, pool="development", clean_count=40, research=True, synthetic=False):
    root.mkdir()
    start, count = (51000, 80) if pool == "development" else (52000, 200)
    shards = []
    for index in range(4):
        directory = root / f"part-{index}"
        directory.mkdir()
        rows = [row(seed, start=start, clean_count=clean_count) for seed in range(start + index * count // 4, start + (index + 1) * count // 4)]
        if index % 2:
            rows.reverse()
        summary = summarize(rows, research_evidence=research, seed_pool="diagnostic", config=config())
        if synthetic:
            summary["synthetic"] = True
        rewrite(directory, summary, rows)
        shards.append((directory, summary, rows))
    return shards


def directories(shards):
    return [entry[0] for entry in shards]


def recompute(summary, rows):
    return summarize(rows, research_evidence=summary["research_evidence"], seed_pool=summary["seed_pool"], config=summary["config"])


@pytest.mark.parametrize("pool,count,start", [("development", 80, 51000), ("final_id_test", 200, 52000)])
def test_complete_merge_preserves_config_and_writes_runtime_artifacts(tmp_path, pool, count, start):
    shards = make_shards(tmp_path / "shards", pool=pool)
    before = {str(path): path.read_bytes() for directory in directories(shards) for path in directory.iterdir()}
    output = tmp_path / "merged"
    summary = merge_evaluations(list(reversed(directories(shards))), seed_pool=pool, output_dir=output)
    assert summary["seed_pool"] == pool
    assert summary["episodes"] == count
    assert summary["seeds"] == list(range(start, start + count))
    assert summary["research_evidence"] is True
    assert summary["config"]["complete_seed_pool"] is True
    assert len(summary["config"]["contract_sha256"]) == 64
    for key, value in config().items():
        if key != "complete_seed_pool":
            assert summary["config"][key] == value
    assert summary["merge_provenance"]["shard_count"] == 4
    assert all(len(item["metrics_sha256"]) == len(item["episodes_sha256"]) == 64 for item in summary["merge_provenance"]["sources"])
    assert {path.name for path in output.iterdir()} == {"metrics.json", "episodes.jsonl", "episodes.csv", "merge_provenance.json"}
    assert len((output / "episodes.jsonl").read_text().splitlines()) == count
    assert len((output / "episodes.csv").read_text().splitlines()) == count + 1
    assert output.stat().st_mode & 0o777 == 0o700
    assert all(path.stat().st_mode & 0o777 == 0o600 for path in output.iterdir())
    after = {str(path): path.read_bytes() for directory in directories(shards) for path in directory.iterdir()}
    assert before == after


def test_complete_final_merges_are_accepted_by_promotion_gate(tmp_path):
    reference = make_shards(tmp_path / "reference-parts", pool="final_id_test", clean_count=40)
    candidate = make_shards(tmp_path / "candidate-parts", pool="final_id_test", clean_count=80)
    reference_dir, candidate_dir = tmp_path / "reference", tmp_path / "candidate"
    merge_evaluations(directories(reference), seed_pool="final_id_test", output_dir=reference_dir)
    merge_evaluations(directories(candidate), seed_pool="final_id_test", output_dir=candidate_dir)
    result = evaluate_promotion(reference_dir, candidate_dir, bootstrap_replicates=100)
    assert result["status"] == "eligible_for_manual_review"
    assert result["candidate_minus_reference"]["clean_success_rate"]["delta"] == pytest.approx(0.2)


def test_mock_only_merge_stays_nonresearch_and_is_rejected_for_promotion(tmp_path):
    shards = make_shards(tmp_path / "shards", pool="final_id_test", research=False, synthetic=True)
    output = tmp_path / "merged"
    summary = merge_evaluations(directories(shards), seed_pool="final_id_test", output_dir=output)
    assert summary["research_evidence"] is False
    assert summary["synthetic"] is True
    with pytest.raises(ValueError, match="mock|Synthetic|research"):
        evaluate_promotion(output, output, bootstrap_replicates=10)


@pytest.mark.parametrize("mutation", ["missing", "duplicate_across", "duplicate_within", "unexpected", "summary_missing"])
def test_missing_duplicate_or_unexpected_seeds_fail_before_output(tmp_path, mutation):
    shards = make_shards(tmp_path / "shards")
    directory, summary, rows = shards[-1]
    if mutation == "missing":
        rows.pop()
        summary = recompute(summary, rows)
    elif mutation == "duplicate_across":
        rows[0]["seed"] = 51000
        summary = recompute(summary, rows)
    elif mutation == "duplicate_within":
        rows.append(rows[0].copy())
    elif mutation == "unexpected":
        rows[0]["seed"] = 52000
        summary = recompute(summary, rows)
    else:
        summary["seeds"].pop()
    rewrite(directory, summary, rows)
    output = tmp_path / "merged"
    with pytest.raises(ValueError):
        merge_evaluations(directories(shards), seed_pool="development", output_dir=output)
    assert not output.exists()


@pytest.mark.parametrize("key,value", [
    ("reuse_environment", True), ("planner_horizon", 4),
    ("specialist_sha256", "b" * 64), ("base_weight_sha256", "b" * 64),
    ("planner_manifest_sha256", "b" * 64), ("planner_max_seq_len", 512),
    ("predator_speed_ratio", 0.2), ("max_line_of_sight_distance", 0.7),
    ("navigation_contract", "wrong_navigation_v2"), ("reset_contract", "wrong_reset_v2"),
    ("destination_mode", "reject"), ("projection_epsilon", 1e-4),
])
def test_all_configuration_and_checkpoint_identity_must_match(tmp_path, key, value):
    shards = make_shards(tmp_path / "shards")
    directory, summary, rows = shards[-1]
    summary["config"][key] = value
    rewrite(directory, summary, rows)
    with pytest.raises(ValueError):
        merge_evaluations(directories(shards), seed_pool="development", output_dir=tmp_path / "merged")


@pytest.mark.parametrize("mutation", ["policy", "research", "mock_flag", "schema", "context", "action", "stale_metric", "nan", "wrong_count", "wrong_clean", "negative_projection", "invalid_projection_rate", "stale_projection_metric"])
def test_mixed_or_inconsistent_shard_evidence_is_rejected(tmp_path, mutation):
    shards = make_shards(tmp_path / "shards")
    directory, summary, rows = shards[-1]
    if mutation == "policy":
        for item in rows:
            item["policy"] = "different-policy"
        summary = recompute(summary, rows)
    elif mutation == "research":
        summary["research_evidence"] = False
    elif mutation == "mock_flag":
        summary["mock"] = True
    elif mutation == "schema":
        summary["schema_version"] = "wrong"
    elif mutation == "context":
        summary["context_schema"] = "wrong"
    elif mutation == "action":
        summary["action_schema"] = "discrete"
    elif mutation == "stale_metric":
        summary["metrics"]["mean_return"] += 0.1
    elif mutation == "nan":
        rows[0]["path_length"] = float("nan")
    elif mutation == "wrong_count":
        summary["episodes"] += 1
    elif mutation == "negative_projection":
        rows[0]["mean_projection_distance"] = -0.1
        summary = recompute(summary, rows)
    elif mutation == "invalid_projection_rate":
        rows[0]["projection_rate"] = 1.1
        summary = recompute(summary, rows)
    elif mutation == "stale_projection_metric":
        summary["metrics"]["mean_projection_distance"] = 0.0
    else:
        rows[0]["clean_success"] = 1
        summary = recompute(summary, rows)
    rewrite(directory, summary, rows)
    with pytest.raises(ValueError):
        merge_evaluations(directories(shards), seed_pool="development", output_dir=tmp_path / "merged")


def test_old_reject_artifacts_cannot_be_merged_into_project_contract(tmp_path):
    shards = make_shards(tmp_path / "shards")
    directory, summary, rows = shards[-1]
    summary["config"].pop("destination_mode")
    summary["config"].pop("projection_epsilon")
    for name in ("requested_invalid_rate", "projection_rate", "mean_projection_distance"):
        summary["metrics"].pop(name)
        for item in rows:
            item.pop(name)
    rewrite(directory, summary, rows)
    with pytest.raises(ValueError):
        merge_evaluations(directories(shards), seed_pool="development", output_dir=tmp_path / "merged")


def test_explicit_reject_v2_contract_can_merge_consistent_reject_shards(tmp_path):
    shards = make_shards(tmp_path / "shards")
    contract = json.loads(DEFAULT_CONTRACT.read_text())
    contract.update(destination_mode="reject", navigation_contract="graph_route_exact_xy_terminal_reject_v2")
    path = tmp_path / "reject-contract.json"
    path.write_text(json.dumps(contract))
    for directory, summary, rows in shards:
        summary["config"].update(destination_mode="reject", navigation_contract="graph_route_exact_xy_terminal_reject_v2")
        for item in rows:
            item.update(projection_rate=0.0, mean_projection_distance=0.0,
                        destination_rejection_rate=item["requested_invalid_rate"])
        rewrite(directory, recompute(summary, rows), rows)
    result = merge_evaluations(directories(shards), seed_pool="development",
                               contract_path=path, output_dir=tmp_path / "merged")
    assert result["config"]["destination_mode"] == "reject"
    assert result["metrics"]["projection_rate"] == 0.0


def test_development_subset_label_is_supported(tmp_path):
    shards = make_shards(tmp_path / "shards")
    directory, summary, rows = shards[0]
    summary["seed_pool"] = "development_subset"
    rewrite(directory, summary, rows)
    result = merge_evaluations(directories(shards), seed_pool="development", output_dir=tmp_path / "merged")
    assert result["episodes"] == 80


def test_output_cannot_overwrite_nonempty_directory_or_enter_git(tmp_path):
    shards = make_shards(tmp_path / "shards")
    with pytest.raises(ValueError, match="empty"):
        merge_evaluations(directories(shards), seed_pool="development", output_dir=shards[0][0])
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / ".git").write_text("gitdir: /some/metadata")
    with pytest.raises(ValueError, match="Git"):
        merge_evaluations(directories(shards), seed_pool="development", output_dir=repo / "merged")
