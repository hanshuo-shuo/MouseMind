"""CLI-only hybrid wiring checks; all model loading and rollouts are stubbed."""

from __future__ import annotations

import json
import sys
from types import ModuleType

import pytest

from mouse_llm.continuous import evaluation


@pytest.fixture
def hybrid_cli(tmp_path, monkeypatch):
    calls = {"continuous_bc": [], "legacy_mlp": [], "planner": [], "hybrid": []}
    specialist_module = ModuleType("mouse_llm.continuous.specialist")

    class ContinuousBC:
        def __init__(self, path, *, device):
            self.name = "fixture-continuous-bc"
            calls["continuous_bc"].append((path, device))

    specialist_module.ContinuousCheckpointPolicy = ContinuousBC
    planner_module = ModuleType("mouse_llm.hierarchical.minimind_planner")

    class Planner:
        def __init__(self, **kwargs):
            calls["planner"].append(kwargs)
            self.decoding_mode = kwargs["decoding_mode"]
            self.max_seq_len = kwargs.get("max_seq_len", 1024)

    planner_module.MiniMindSkillPlanner = Planner
    hybrid_module = ModuleType("mouse_llm.continuous.legacy_hybrid")

    class Hybrid:
        def __init__(self, **kwargs):
            calls["hybrid"].append(kwargs)
            self.name = kwargs["name"]

    hybrid_module.LegacyPlannerContinuousBCPolicy = Hybrid
    environment_module = ModuleType("mouse_llm.continuous.environment")
    environment_module.RESET_CONTRACT = "fresh_navigation_episode_reset_v1"
    environment_module.ContinuousBotEvadeEnv = lambda **kwargs: pytest.fail("CLI test must not construct a simulator")
    navigation_module = ModuleType("mouse_llm.continuous.navigation")
    navigation_module.navigation_contract = lambda mode: f"graph_route_exact_xy_terminal_{mode}_v2"
    for module in (specialist_module, planner_module, hybrid_module, environment_module, navigation_module):
        monkeypatch.setitem(sys.modules, module.__name__, module)

    from mouse_llm.evaluation import closed_loop

    class LegacyMLP:
        def __init__(self, path, **kwargs):
            self.name = "fixture-legacy-mlp"
            calls["legacy_mlp"].append((path, kwargs))

    monkeypatch.setattr(closed_loop, "MLPCheckpointPolicy", LegacyMLP)

    def no_rollout(env_factory, policy, *, seeds, **kwargs):
        calls["evaluated_policy"] = policy
        return [{
            "policy": policy.name, "seed": seeds[0], "success": 1,
            "clean_success": 1, "captured": 0, "captures": 0, "survived": 1,
            "steps": 1, "episode_return": 1.0, "path_length": 0.0,
            "valid_action_rate": 1.0, "latency_mean_seconds": 0.001,
            "deadline_miss_rate": 0.0, "destination_rejection_rate": 0.0,
            "requested_invalid_rate": 0.0, "projection_rate": 0.0,
            "mean_projection_distance": 0.0, "skill_counts": {},
        }]

    monkeypatch.setattr(evaluation, "run_policy", no_rollout)
    files = {name: tmp_path / filename for name, filename in (
        ("specialist", "continuous_bc.pt"), ("base", "base.pth"),
        ("lora", "historical_dpo.pth"), ("legacy_mlp", "legacy_mlp.pt"),
    )}
    for name, path in files.items():
        path.write_bytes(("fixture-" + name).encode())
    catalog = tmp_path / "catalog.json"
    catalog.write_text(json.dumps({"actions": [
        {"action": 0, "destination": [0.2, 0.3]},
        {"action": 1, "destination": [0.7, 0.8]},
    ]}))
    output = tmp_path / "evaluation"
    argv = ["evaluation", "--policy", "legacy-minimind-continuous-bc",
            "--specialist-checkpoint", str(files["specialist"]),
            "--base-weight", str(files["base"]), "--skill-lora-weight", str(files["lora"]),
            "--tokenizer", str(tmp_path / "tokenizer"), "--action-catalog", str(catalog),
            "--seed-start", "51000", "--episodes", "1", "--output-dir", str(output)]
    return {"calls": calls, "files": files, "catalog": catalog, "output": output, "argv": argv}


def test_new_policy_wires_continuous_go_without_loading_legacy_mlp(hybrid_cli, monkeypatch, capsys):
    item = hybrid_cli
    before = {path: path.read_bytes() for path in item["files"].values()}
    monkeypatch.setattr(sys, "argv", item["argv"])
    evaluation.main()
    summary = json.loads(capsys.readouterr().out)
    calls, config = item["calls"], summary["config"]
    assert len(calls["continuous_bc"]) == len(calls["hybrid"]) == len(calls["planner"]) == 1
    assert calls["legacy_mlp"] == []
    assert calls["planner"][0]["lora_weight"] == item["files"]["lora"]
    assert calls["planner"][0]["max_seq_len"] == 1024
    assert calls["planner"][0]["decoding_mode"] == "greedy_trie"
    assert calls["hybrid"][0]["temporal_window"] == 8
    assert calls["hybrid"][0]["planner_horizon"] == 8
    assert calls["hybrid"][0]["evade_distance"] == 0.35
    assert summary["policy"] == "legacy-minimind-continuous-bc"
    assert summary["context_schema"] == evaluation.CONTEXT_SCHEMA
    assert config["planner_context_schema"] == "mousemind_planner_context_v1"
    assert config["executed_action_schema"] == evaluation.ACTION_SCHEMA
    assert config["history_action_proxy"] == "nearest_catalog_id_for_history_only"
    assert config["only_go_replaced"] is True
    assert config["go_execution_source"] == "continuous_bc_exact_xy"
    assert config["non_goal_execution_source"] == "legacy_geometric_catalog"
    assert config["legacy_mlp_checkpoint_used"] is False
    assert config["planner_manifest_sha256"] is None
    assert config["specialist_sha256"] == evaluation._sha256(item["files"]["specialist"])
    assert config["base_weight_sha256"] == evaluation._sha256(item["files"]["base"])
    assert config["skill_lora_sha256"] == evaluation._sha256(item["files"]["lora"])
    assert config["action_catalog_sha256"] == evaluation._sha256(item["catalog"])
    assert config["planner_decoding"] == "greedy_trie"
    assert config["planner_decoding_ablation"] is False
    assert "No planner training" in config["go_replacement_note"]
    assert {path: path.read_bytes() for path in before} == before


def test_new_policy_does_not_consume_an_unused_legacy_mlp_argument(hybrid_cli, monkeypatch, capsys):
    item = hybrid_cli
    monkeypatch.setattr(sys, "argv", [*item["argv"], "--legacy-mlp-checkpoint", str(item["output"].parent / "does-not-exist.pt")])
    evaluation.main()
    report = json.loads(capsys.readouterr().out)
    assert item["calls"]["legacy_mlp"] == []
    assert report["config"]["legacy_mlp_checkpoint_used"] is False


def test_common_experiment_parameters_remain_configurable(hybrid_cli, monkeypatch, capsys):
    item = hybrid_cli
    monkeypatch.setattr(sys, "argv", [*item["argv"], "--preference", "goal_first",
                                      "--planner-horizon", "4", "--evade-distance", "0.2",
                                      "--instruction-split", "validation"])
    evaluation.main()
    report = json.loads(capsys.readouterr().out)
    assert item["calls"]["hybrid"][0]["preference"].value == "goal_first"
    assert item["calls"]["hybrid"][0]["planner_horizon"] == 4
    assert item["calls"]["hybrid"][0]["evade_distance"] == 0.2
    assert item["calls"]["planner"][0]["instruction_split"] == "validation"
    assert report["config"]["planner_context_schema"] == "mousemind_planner_context_v1"
    assert report["config"]["planner_max_seq_len"] == 1024


@pytest.mark.parametrize("extra", [
    ["--allow-legacy-planner"],
    ["--planner-decoding", "sequence_logprob"],
    ["--planner-manifest", "/nonexistent/new-planner-manifest.json"],
])
def test_new_policy_rejects_context_decoder_and_new_checkpoint_manifest_changes(hybrid_cli, monkeypatch, extra):
    item = hybrid_cli
    monkeypatch.setattr(sys, "argv", [*item["argv"], *extra])
    with pytest.raises(ValueError):
        evaluation.main()
    assert not item["output"].exists()
    assert item["calls"]["planner"] == item["calls"]["hybrid"] == []


@pytest.mark.parametrize("flag", ["--specialist-checkpoint", "--base-weight", "--skill-lora-weight"])
def test_new_policy_requires_its_actual_three_checkpoints(hybrid_cli, monkeypatch, flag):
    item = hybrid_cli
    argv = list(item["argv"])
    index = argv.index(flag)
    del argv[index:index + 2]
    monkeypatch.setattr(sys, "argv", argv)
    with pytest.raises(ValueError, match="requires"):
        evaluation.main()
    assert not item["output"].exists()


def test_existing_legacy_policy_still_uses_its_original_discrete_specialist(hybrid_cli, monkeypatch, capsys):
    item = hybrid_cli
    argv = ["legacy-minimind" if value == "legacy-minimind-continuous-bc" else value for value in item["argv"]]
    argv.extend(["--legacy-mlp-checkpoint", str(item["files"]["legacy_mlp"])])
    monkeypatch.setattr(sys, "argv", argv)
    evaluation.main()
    report = json.loads(capsys.readouterr().out)
    assert len(item["calls"]["legacy_mlp"]) == 1
    assert item["calls"]["continuous_bc"] == item["calls"]["hybrid"] == []
    assert report["policy"] == "legacy-minimind-catalog-reference"
    assert report["config"]["specialist_sha256"] == evaluation._sha256(item["files"]["legacy_mlp"])
    assert report["config"]["only_go_replaced"] is False
    assert report["config"]["catalog_constrained_reference"] is True
    assert report["config"]["planner_decoding"] == "greedy_trie"
