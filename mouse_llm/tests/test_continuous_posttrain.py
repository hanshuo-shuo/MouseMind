"""Training-input and checkpoint-lineage validation without model imports."""

import builtins
import json

import pytest

pytest.importorskip("numpy")

from mouse_llm.continuous import posttrain
from mouse_llm.continuous.context import CONTEXT_SCHEMA
from mouse_llm.continuous.contracts import ACTION_SCHEMA
from mouse_llm.continuous.posttraining import DATASET_SCHEMA, build_rows, seed_split, sha256
from mouse_llm.hierarchical.policy import Skill


def _record(seed):
    return {
        "seed": seed, "anchor_id": f"synthetic-{seed}",
        "context": {
            "schema_version": CONTEXT_SCHEMA, "action_schema": ACTION_SCHEMA,
            "temporal_window": 8,
            "current": {"prey_x": seed / 100000.0, "prey_y": 0.5, "predator_visible": False},
            "history": {"recent_xy_actions": [], "recent_high_level_skills": []},
        },
        "branches": {
            Skill.GO_TO_GOAL.value: {"captures": 0, "success": 1, "goal_progress": 0.4},
            Skill.EVADE_PREDATOR.value: {"captures": 0, "success": 0, "goal_progress": 0.1},
            Skill.HOLD_POSITION.value: {"captures": 1, "success": 0, "goal_progress": 0.0},
        },
    }


def _write_json(path, value):
    path.write_text(json.dumps(value, sort_keys=True) + "\n")


@pytest.fixture
def valid_dataset(tmp_path):
    directory = tmp_path / "data"
    directory.mkdir()
    specialist = tmp_path / "specialist.pt"
    specialist.write_bytes(b"synthetic specialist for hash validation only")
    manifest = {
        "schema_version": DATASET_SCHEMA,
        "action_schema": ACTION_SCHEMA,
        "context_schema": CONTEXT_SCHEMA,
        "specialist_sha256": sha256(specialist),
        "horizon": 32, "candidate_steps": 8, "memory_steps": 8,
        "evade_distance": 0.35, "minimum_margin": 0.01,
        "continuation": "public_visibility_rule",
        "environment_config": {
            "world": "21_05", "max_steps": 300, "time_step": 0.25,
            "predator_speed_ratio": 0.15, "max_line_of_sight_distance": 1.0,
            "navigation_contract": "graph_route_exact_xy_terminal_v1",
            "reset_contract": "fresh_navigation_episode_reset_v1",
        },
        "splits": {},
    }
    for split in ("train", "validation"):
        seed = next(seed for seed in range(50000, 50200) if seed_split(seed) == split)
        rows = build_rows(_record(seed), minimum_margin=manifest["minimum_margin"])
        path = directory / f"planner_{split}.jsonl"
        path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows))
        manifest["splits"][split] = {"sha256": sha256(path), "rows": len(rows)}
    _write_json(directory / "manifest.json", manifest)
    return directory, specialist


def _mutate_manifest(dataset, mutate):
    directory, _ = dataset
    path = directory / "manifest.json"
    manifest = json.loads(path.read_text())
    mutate(manifest)
    _write_json(path, manifest)


def _mutate_row(dataset, mutate, *, split="train", update_hash=True):
    directory, _ = dataset
    path = directory / f"planner_{split}.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    mutate(rows[0])
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows))
    if update_hash:
        _mutate_manifest(dataset, lambda manifest: manifest["splits"][split].update(sha256=sha256(path)))


def _mutate_context(row, key, value):
    payload = json.loads(row["prompt"][1]["content"])
    payload["context"][key] = value
    row["prompt"][1]["content"] = json.dumps(payload, sort_keys=True)


def test_valid_continuous_dataset_matches_hashes_schemas_and_seed_ownership(valid_dataset):
    directory, specialist = valid_dataset
    manifest = posttrain.validate_dataset(directory, specialist)
    assert manifest["action_schema"] == ACTION_SCHEMA
    assert manifest["context_schema"] == CONTEXT_SCHEMA
    assert manifest["specialist_sha256"] == sha256(specialist)
    assert manifest["splits"]["train"]["rows"] > 0
    assert manifest["splits"]["validation"]["rows"] > 0


@pytest.mark.parametrize("seed", [30000, 51000, 52000, 50200])
def test_dataset_rejects_historical_development_final_and_out_of_pool_seeds(valid_dataset, seed):
    _mutate_row(valid_dataset, lambda row: row.update(seed=seed))
    with pytest.raises(ValueError, match="collection seed"):
        posttrain.validate_dataset(*valid_dataset)


def test_dataset_rejects_validation_owned_seed_in_training_file(valid_dataset):
    seed = next(seed for seed in range(50000, 50200) if seed_split(seed) == "validation")
    _mutate_row(valid_dataset, lambda row: row.update(seed=seed))
    with pytest.raises(ValueError, match="collection seed"):
        posttrain.validate_dataset(*valid_dataset)


def test_dataset_rejects_changed_executor(valid_dataset):
    _, specialist = valid_dataset
    specialist.write_bytes(b"a different executor needs different counterfactual labels")
    with pytest.raises(ValueError, match="executor changed"):
        posttrain.validate_dataset(*valid_dataset)


@pytest.mark.parametrize("key", ["schema_version", "action_schema", "context_schema"])
def test_manifest_rejects_legacy_schema(valid_dataset, key):
    _mutate_manifest(valid_dataset, lambda manifest: manifest.update({key: "legacy-v1"}))
    with pytest.raises(ValueError, match="continuous planner dataset"):
        posttrain.validate_dataset(*valid_dataset)


@pytest.mark.parametrize("key,value", [
    ("schema_version", "legacy_preferences"),
    ("split", "validation"),
    ("action_schema", "discrete-action-v1"),
    ("context_schema", "mousemind_planner_context_v1"),
])
def test_rows_reject_schema_or_split_mismatch_even_with_fresh_file_hash(valid_dataset, key, value):
    _mutate_row(valid_dataset, lambda row: row.update({key: value}))
    with pytest.raises(ValueError):
        posttrain.validate_dataset(*valid_dataset)


@pytest.mark.parametrize("key,value", [
    ("schema_version", "mousemind_planner_context_v1"),
    ("action_schema", "discrete-action-v1"),
])
def test_prompt_rejects_legacy_context_or_action_schema(valid_dataset, key, value):
    _mutate_row(valid_dataset, lambda row: _mutate_context(row, key, value))
    with pytest.raises(ValueError):
        posttrain.validate_dataset(*valid_dataset)


def test_mutated_dataset_bytes_fail_before_training(valid_dataset):
    _mutate_row(valid_dataset, lambda row: row.update(anchor_id="mutated"), update_hash=False)
    with pytest.raises(ValueError, match="manifest"):
        posttrain.validate_dataset(*valid_dataset)


def test_manifest_row_count_must_match_actual_nonempty_file(valid_dataset):
    _mutate_manifest(valid_dataset, lambda manifest: manifest["splits"]["validation"].update(rows=12345))
    with pytest.raises(ValueError, match="sample count"):
        posttrain.validate_dataset(*valid_dataset)


@pytest.mark.parametrize("key,value", [
    ("chosen", "not json"),
    ("rejected", '{"skill":"invented-skill"}'),
    ("chosen", '{"skill":"go_to_goal","extra":1}'),
])
def test_invalid_preference_responses_are_rejected(valid_dataset, key, value):
    _mutate_row(valid_dataset, lambda row: row.update({key: value}))
    with pytest.raises(ValueError, match="preference"):
        posttrain.validate_dataset(*valid_dataset)


@pytest.mark.parametrize("different_formatting", [False, True])
def test_chosen_and_rejected_cannot_be_the_same_skill(valid_dataset, different_formatting):
    def mutate(row):
        row["rejected"] = (
            json.dumps(json.loads(row["chosen"]), indent=2)
            if different_formatting else row["chosen"]
        )
    _mutate_row(valid_dataset, mutate)
    with pytest.raises(ValueError, match="preference"):
        posttrain.validate_dataset(*valid_dataset)


def test_valid_skills_with_inverted_preference_order_are_rejected(valid_dataset):
    def mutate(row):
        row["chosen"], row["rejected"] = row["rejected"], row["chosen"]
        row["target_skill"] = json.loads(row["chosen"])["skill"]
        row["conversations"][-1]["content"] = row["chosen"]
    _mutate_row(valid_dataset, mutate)
    with pytest.raises(ValueError):
        posttrain.validate_dataset(*valid_dataset)


def test_target_skill_must_agree_with_chosen_response(valid_dataset):
    _mutate_row(valid_dataset, lambda row: row.update(target_skill=Skill.HOLD_POSITION.value))
    with pytest.raises(ValueError):
        posttrain.validate_dataset(*valid_dataset)


def test_instruction_templates_cannot_cross_training_validation_partition(valid_dataset):
    _mutate_row(valid_dataset, lambda row: row.update(instruction_split="unseen_test"))
    with pytest.raises(ValueError, match="Instruction split"):
        posttrain.validate_dataset(*valid_dataset)


def test_utility_scores_must_be_finite(valid_dataset):
    _mutate_row(valid_dataset, lambda row: row["utilities"].update(go_to_goal=float("nan")))
    with pytest.raises(ValueError, match="utility"):
        posttrain.validate_dataset(*valid_dataset)


def _cli_args(dataset, tmp_path, *, stage="sft"):
    directory, specialist = dataset
    base = tmp_path / "base.pth"
    base.write_bytes(b"synthetic base hash only")
    return [
        "posttrain", stage, "--data-dir", str(directory),
        "--specialist-checkpoint", str(specialist), "--base-weight", str(base),
        "--output-dir", str(tmp_path / "training-output"), "--validate-only",
    ], base


def _forbid_transformers(monkeypatch):
    original = builtins.__import__
    def guarded(name, *args, **kwargs):
        if name == "transformers" or name.startswith("transformers."):
            pytest.fail("validate-only must not import or load a transformer")
        return original(name, *args, **kwargs)
    monkeypatch.setattr(builtins, "__import__", guarded)


def test_sft_validate_only_does_not_load_models_or_create_training_output(valid_dataset, tmp_path, monkeypatch, capsys):
    args, _ = _cli_args(valid_dataset, tmp_path)
    monkeypatch.setattr("sys.argv", args)
    _forbid_transformers(monkeypatch)
    posttrain.main()
    output = json.loads(capsys.readouterr().out)
    assert output["validated"] is True and output["stage"] == "sft"
    assert not (tmp_path / "training-output").exists()


def test_dpo_validate_only_requires_sft_initialization(valid_dataset, tmp_path, monkeypatch):
    args, _ = _cli_args(valid_dataset, tmp_path, stage="dpo")
    monkeypatch.setattr("sys.argv", args)
    _forbid_transformers(monkeypatch)
    with pytest.raises(SystemExit, match="2"):
        posttrain.main()


@pytest.mark.parametrize("wrong_key", [None, "skill_lora_sha256", "base_weight_sha256", "specialist_sha256", "context_schema", "action_schema"])
def test_dpo_validate_only_checks_frozen_sft_lineage(valid_dataset, tmp_path, monkeypatch, wrong_key):
    args, base = _cli_args(valid_dataset, tmp_path, stage="dpo")
    lora = tmp_path / "sft.pth"
    lora.write_bytes(b"synthetic sft hash only")
    manifest = {
        "schema_version": "mousemind_continuous_planner_checkpoint_v1", "stage": "sft",
        "skill_lora_sha256": sha256(lora), "base_weight_sha256": sha256(base),
        "specialist_sha256": sha256(valid_dataset[1]),
        "context_schema": CONTEXT_SCHEMA, "action_schema": ACTION_SCHEMA,
    }
    if wrong_key is not None:
        manifest[wrong_key] = "wrong"
    path = tmp_path / "sft-manifest.json"
    _write_json(path, manifest)
    monkeypatch.setattr("sys.argv", [*args, "--sft-lora-weight", str(lora), "--sft-manifest", str(path)])
    _forbid_transformers(monkeypatch)
    if wrong_key is None:
        posttrain.main()
        assert not (tmp_path / "training-output").exists()
    else:
        with pytest.raises(ValueError, match=wrong_key):
            posttrain.main()


@pytest.mark.parametrize("flag,value", [("--epochs", "0"), ("--beta", "0"), ("--max-seq-len", "63")])
def test_validate_only_rejects_invalid_training_parameters(valid_dataset, tmp_path, monkeypatch, flag, value):
    args, _ = _cli_args(valid_dataset, tmp_path)
    monkeypatch.setattr("sys.argv", [*args, flag, value])
    with pytest.raises(SystemExit, match="2"):
        posttrain.main()
