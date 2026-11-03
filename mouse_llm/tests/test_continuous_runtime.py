"""Behavioral checks for XY execution and capture-free completion."""

import json
import sys
from types import ModuleType

import numpy as np
import pytest

from mouse_llm.continuous.contracts import ContinuousDecision
from mouse_llm.continuous.evaluation import run_episode, run_policy, summarize
from mouse_llm.continuous.policy import (
    ContinuousHierarchicalPolicy, ContinuousSkillController,
    DirectContinuousPolicy, DiscreteDestinationAdapter,
)
from mouse_llm.hierarchical.policy import Skill


def observation(*, x=0.5, y=0.5, predator=None, puffed=False):
    values = np.zeros(15, dtype=np.float32)
    values[:2] = x, y
    values[14] = 0.4
    values[10] = int(puffed)
    if predator is not None:
        values[3] = 1
        values[4:6] = predator
    return values


class Specialist:
    name = "fixed-test-specialist"

    def reset(self, seed):
        self.seed = seed

    def act(self, values):
        assert len(values) == 10
        return ContinuousDecision((0.237, 0.811), metadata={"specialist": "test"})


class EpisodeEnv:
    max_step = 3

    def __init__(self, *, capture_step=None, succeeds=True):
        self.capture_step, self.succeeds = capture_step, succeeds
        self.closed = False

    def reset(self, *, seed):
        self.seed, self.index, self.captures = seed, 0, 0
        return observation(), {}

    def step(self, action):
        assert len(action) == 2
        self.index += 1
        puffed = self.index == self.capture_step
        self.captures += int(puffed)
        done = self.index == 3
        return observation(x=action[0], y=action[1], puffed=puffed), 1.0, done and self.succeeds, done and not self.succeeds, {
            "captures": self.captures, "is_success": done and self.succeeds,
            "predator_x": 0.999, "predator_y": 0.999,
        }

    def close(self):
        self.closed = True


def test_xy_wrapper_preserves_native_destination_without_catalog_projection():
    gym = pytest.importorskip("gymnasium")
    from mouse_llm.continuous.environment import ContinuousBotEvadeEnv

    class Native(gym.Env):
        action_space = gym.spaces.Box(0.0, 1.0, shape=(3,), dtype=np.float32)
        observation_space = gym.spaces.Box(-np.inf, np.inf, shape=(15,), dtype=np.float32)
        frame_stack_k = 1

        def step(self, action):
            self.received = action
            return observation(), 7.0, False, False, {"native": True}

    native = Native()
    env = ContinuousBotEvadeEnv(native, destination_mode="reject")
    assert env.action_space.shape == (2,)
    result = env.step((0.23713, 0.81127))
    np.testing.assert_array_equal(native.received, [0.23713, 0.81127, 0.0])
    assert result[1:4] == (7.0, False, False)
    assert result[4]["native"] is True
    assert result[4]["invalid_destination"] is False
    with pytest.raises(ValueError):
        env.step([0.1, 0.2, 0.3])
    with pytest.raises(ValueError):
        env.step([-0.1, 0.2])


def test_hidden_predator_is_not_origin_and_memory_expires():
    controller = ContinuousSkillController(Specialist(), evade_distance=0.2, memory_steps=1)
    controller.reset(7)
    no_memory = controller.act(Skill.EVADE_PREDATOR, observation())
    assert no_memory.action == (0.237, 0.811)
    assert no_memory.metadata["fallback_reason"] == "no_unexpired_public_predator_memory"
    seen = controller.act(Skill.GO_TO_GOAL, observation(predator=(0.8, 0.5)))
    assert seen.metadata["predator_memory_age"] == 0
    remembered = controller.act(Skill.EVADE_PREDATOR, observation())
    assert remembered.action == pytest.approx((0.3, 0.5))
    assert remembered.metadata["predator_source"] == "public_memory"
    expired = controller.act(Skill.EVADE_PREDATOR, observation())
    assert expired.action == (0.237, 0.811)
    controller.reset(8)
    assert controller.memory_state()["last_seen_predator_xy"] is None


def test_controller_memory_restore_supports_branch_continuation():
    first = ContinuousSkillController(Specialist(), memory_steps=3)
    first.reset(1)
    first.observe(observation(predator=(0.8, 0.5)))
    first.observe(observation())
    restored = ContinuousSkillController(Specialist(), memory_steps=3)
    restored.reset(1)
    restored.restore_memory(first.memory_state())
    assert restored.act(Skill.EVADE_PREDATOR, observation()).action == first.act(Skill.EVADE_PREDATOR, observation()).action


def test_hierarchical_history_stores_xy_without_integer_aliases():
    policy = ContinuousHierarchicalPolicy(specialist=Specialist())
    policy.reset(1)
    policy.act(observation())
    policy.act(observation())
    history = policy.last_context.payload()["history"]
    assert history["recent_xy_actions"] == [[0.237, 0.811]]
    assert "recent_low_level_actions" not in history


@pytest.mark.parametrize("captured,succeeds,clean", [(None, True, 1), (2, True, 0), (None, False, 0)])
def test_clean_success_is_joint_completion_and_zero_capture(captured, succeeds, clean):
    trace = []
    row = run_episode(EpisodeEnv(capture_step=captured, succeeds=succeeds),
                      DirectContinuousPolicy(Specialist()), seed=9, trace_hook=trace.append)
    assert row["clean_success"] == clean
    assert row["success"] == int(succeeds)
    assert row["time_to_first_capture"] == captured
    assert len(trace) == 3
    assert trace[0]["prefix_xy_actions"] == []
    assert trace[1]["prefix_xy_actions"] == [[0.237, 0.811]]
    assert trace[0]["legacy_observation"][3:5] == [0.0, 0.0]
    assert trace[1]["context"]["history"]["recent_high_level_skills"] == ["unlabeled"]
    assert "predator_x" not in trace[0]  # evaluator-only info never becomes policy state
    report = summarize([row])
    assert report["research_evidence"] is False


def test_run_policy_uses_fresh_environments_and_closes_them():
    instances = []
    def factory():
        env = EpisodeEnv()
        instances.append(env)
        return env
    rows = run_policy(factory, DirectContinuousPolicy(Specialist()), seeds=[3, 4])
    assert [row["seed"] for row in rows] == [3, 4]
    assert len(instances) == 2 and all(env.closed for env in instances)
    with pytest.raises(ValueError):
        run_policy(factory, DirectContinuousPolicy(Specialist()), seeds=[3, 3])


def test_run_policy_reuses_one_environment_but_resets_each_seed():
    instances, reset_seeds, closes = [], [], []
    class TrackedEnvironment(EpisodeEnv):
        def reset(self, *, seed):
            reset_seeds.append(seed)
            return super().reset(seed=seed)
        def close(self):
            closes.append(self)
            super().close()
    def factory():
        env = TrackedEnvironment()
        instances.append(env)
        return env
    rows = run_policy(factory, DirectContinuousPolicy(Specialist()), seeds=[3, 4], reuse_environment=True)
    assert [row["seed"] for row in rows] == [3, 4]
    assert reset_seeds == [3, 4]
    assert len(instances) == 1 and closes == instances
    assert all(row["steps"] == 3 and row["clean_success"] == 1 for row in rows)


def test_reused_evaluation_closes_its_environment_on_policy_error():
    instances = []
    class BrokenPolicy(DirectContinuousPolicy):
        def act(self, observation):
            raise RuntimeError("policy failed")
    def factory():
        env = EpisodeEnv()
        instances.append(env)
        return env
    with pytest.raises(RuntimeError, match="policy failed"):
        run_policy(factory, BrokenPolicy(Specialist()), seeds=[3, 4], reuse_environment=True)
    assert len(instances) == 1 and instances[0].closed


def test_discrete_reference_has_explicit_catalog_identity():
    class Discrete:
        def reset(self, seed):
            pass
        def act(self, obs):
            from mouse_llm.evaluation.closed_loop import PolicyDecision
            return PolicyDecision(1)
    adapter = DiscreteDestinationAdapter(Discrete(), [[0.1, 0.2], [0.7, 0.9]])
    decision = adapter.act(observation())
    assert decision.action == (0.7, 0.9)
    assert decision.metadata["catalog_constrained_reference"] is True


def test_legacy_hierarchy_reference_keeps_its_public_environment_observation():
    class LegacyHierarchy:
        observation_mode = "environment"
        def reset(self, seed):
            pass
        def act(self, obs):
            from mouse_llm.evaluation.closed_loop import PolicyDecision
            assert len(obs) == 15
            assert obs[3] == 1
            return PolicyDecision(0, metadata={"skill": Skill.EVADE_PREDATOR.value})
    adapter = DiscreteDestinationAdapter(LegacyHierarchy(), [[0.1, 0.2]])
    decision = adapter.act(observation(predator=(0.8, 0.5)))
    assert decision.action == (0.1, 0.2)
    assert decision.metadata["skill"] == Skill.EVADE_PREDATOR.value


def test_exact_navigation_preserves_different_destinations_in_same_cell():
    pytest.importorskip("shapely")
    from types import SimpleNamespace
    from shapely.geometry import Polygon
    from mouse_llm.continuous.navigation import ExactXYNavigation

    base = SimpleNamespace(points=SimpleNamespace(point_list=[(0.5, 0.5)]))
    base.get_path = lambda src, dst: [(0.5, 0.5)]
    nav = ExactXYNavigation(base, arena=Polygon([(0, 0), (1, 0), (1, 1), (0, 1)]), occlusions=[])
    assert nav.get_path((0.4, 0.5), (0.43, 0.51)) == [(0.43, 0.51)]
    assert nav.get_path((0.4, 0.5), (0.44, 0.51)) == [(0.44, 0.51)]
    assert nav.get_path((0.4, 0.5), (0.4, 0.5)) == []
    assert nav.last_invalid_reason is None


def test_exact_navigation_rejects_obstacles_and_keeps_predator_router():
    pytest.importorskip("shapely")
    from types import SimpleNamespace
    from shapely.geometry import Polygon
    from mouse_llm.continuous.navigation import install_exact_xy_navigation

    base = SimpleNamespace(points=SimpleNamespace(point_list=[(0.3, 0.5)]))
    base.get_path = lambda src, dst: [(0.3, 0.5)]
    model = SimpleNamespace(
        prey=SimpleNamespace(navigation=base), predator=SimpleNamespace(navigation=base),
        arena=Polygon([(0.1, 0.1), (0.9, 0.1), (0.9, 0.9), (0.1, 0.9)]),
        occlusions=[Polygon([(0.6, 0.4), (0.8, 0.4), (0.8, 0.6), (0.6, 0.6)])],
    )
    nav = install_exact_xy_navigation(model)
    assert model.predator.navigation is base
    assert model.prey.navigation is nav
    assert nav.get_path((0.3, 0.5), (0.7, 0.5)) == []
    assert nav.last_invalid_reason == "destination_outside_free_space"
    assert nav.get_path((0.3, 0.5), (0.01, 0.5)) == []
    assert nav.last_invalid_reason == "destination_outside_free_space"


def test_exact_navigation_uses_clear_graph_route_before_terminal_xy():
    pytest.importorskip("shapely")
    from types import SimpleNamespace
    from shapely.geometry import Polygon, LineString
    from mouse_llm.continuous.navigation import ExactXYNavigation

    wall = Polygon([(0.45, 0.1), (0.55, 0.1), (0.55, 0.7), (0.45, 0.7)])
    base = SimpleNamespace(points=SimpleNamespace(point_list=[(0.3, 0.8), (0.7, 0.8), (0.7, 0.5)]))
    base.get_path = lambda src, dst: [(0.3, 0.8), (0.7, 0.8), (0.7, 0.5)]
    nav = ExactXYNavigation(base, arena=Polygon([(0, 0), (1, 0), (1, 1), (0, 1)]), occlusions=[wall])
    path = nav.get_path((0.3, 0.5), (0.71, 0.52))
    assert path[-1] == (0.71, 0.52)
    assert not LineString([(0.3, 0.5), *path]).intersects(wall)


def test_wrapper_rejects_wall_target_and_clears_stale_navigation_on_reset():
    gym = pytest.importorskip("gymnasium")
    pytest.importorskip("shapely")
    from shapely.geometry import Polygon
    from types import SimpleNamespace
    from mouse_llm.continuous.environment import ContinuousBotEvadeEnv, RESET_CONTRACT

    class Agent:
        def __init__(self, navigation):
            self.navigation = navigation
            self.state = SimpleNamespace(location=(0.3, 0.5))
            self.dynamics = SimpleNamespace(forward_speed=0.0, turn_speed=0.0)
            self.new_destination = self.destination = None
            self.path = []
            self.destination_wait = self.navigation_plan_update_wait = 0
            self.active_navigation = True
        def stop_navigation(self):
            self.dynamics.forward_speed = self.dynamics.turn_speed = 0.0

    class Data:
        def reset(self):
            self.puffed, self.puff_count, self.goal_achieved = False, 0, False

    class Native(gym.Env):
        action_space = gym.spaces.Box(0.0, 1.0, shape=(3,), dtype=np.float32)
        observation_space = gym.spaces.Box(-np.inf, np.inf, shape=(15,), dtype=np.float32)
        frame_stack_k = 1
        def __init__(self):
            router = SimpleNamespace(points=SimpleNamespace(point_list=[(0.3, 0.5)]))
            router.get_path = lambda src, dst: [(0.3, 0.5)]
            self.model = SimpleNamespace(
                prey=Agent(router), predator=Agent(router), time=0.0, paused=False,
                arena=Polygon([(0, 0), (1, 0), (1, 1), (0, 1)]),
                occlusions=[Polygon([(0.6, 0.4), (0.8, 0.4), (0.8, 0.6), (0.6, 0.6)])],
                prey_data=Data(),
            )
        def reset(self, *, seed=None, options=None):
            self.reset_seed = seed
            return observation(x=0.3), {}
        def step(self, action):
            self.received = action
            return observation(x=0.3), 0.0, False, False, {}

    native = Native()
    env = ContinuousBotEvadeEnv(native, destination_mode="reject")
    original_predator_router = native.model.predator.navigation
    env.reset(seed=17)
    native.model.prey.destination = (0.8, 0.5)
    native.model.prey.path = [(0.8, 0.5)]
    result = env.step((0.7, 0.5))
    assert result[-1]["invalid_destination"] is True
    np.testing.assert_array_equal(native.received, [0.3, 0.5, 0.0])
    assert native.model.prey.path == []
    native.model.time, native.model.paused = 5.0, True
    for agent in (native.model.prey, native.model.predator):
        agent.new_destination, agent.destination = (0.8, 0.5), (0.8, 0.5)
        agent.path = [(0.8, 0.5)]
        agent.destination_wait, agent.navigation_plan_update_wait = 7, 6
        agent.dynamics.forward_speed, agent.dynamics.turn_speed = 0.5, 3.0
    native.model.prey_data.puffed = True
    native.model.prey_data.puff_count = 8
    obs, info = env.reset(seed=17)
    fresh = ContinuousBotEvadeEnv(Native(), destination_mode="reject")
    fresh_obs, _ = fresh.reset(seed=17)
    np.testing.assert_array_equal(obs, fresh_obs)
    assert native.model.time == 0 and native.model.paused is False
    assert native.model.prey_data.puffed is False and native.model.prey_data.puff_count == 0
    for agent in (native.model.prey, native.model.predator):
        assert agent.new_destination is None and agent.destination is None and agent.path == []
        assert agent.destination_wait == 0 and agent.navigation_plan_update_wait == 0
        assert agent.dynamics.forward_speed == 0 and agent.dynamics.turn_speed == 0
    assert native.model.predator.navigation is original_predator_router
    assert native.model.prey.navigation is env.exact_navigation
    assert info["reset_contract"] == RESET_CONTRACT


def test_evaluation_cli_keeps_historical_greedy_decoding_default():
    from mouse_llm.continuous.evaluation import build_parser

    args = build_parser().parse_args(["--output-dir", "/tmp/unused-evaluation"])
    assert args.planner_decoding == "greedy_trie"
    with pytest.raises(SystemExit):
        build_parser().parse_args(["--output-dir", "/tmp/unused-evaluation",
                                   "--planner-decoding", "unknown-decoder"])


@pytest.mark.parametrize("policy", ["legacy-minimind", "continuous-bc", "rule", "discrete-reference", "sac-teacher"])
def test_evaluation_cli_rejects_decoder_ablation_for_other_policies_before_io(tmp_path, monkeypatch, policy):
    from mouse_llm.continuous import evaluation

    output = tmp_path / "evaluation"
    monkeypatch.setattr(sys, "argv", ["evaluation", "--policy", policy,
                                      "--planner-decoding", "sequence_logprob",
                                      "--output-dir", str(output)])
    monkeypatch.setattr(evaluation, "_private_output", lambda path: pytest.fail("Invalid decoder/policy combination must fail before writing output"))
    with pytest.raises(ValueError, match="decoding"):
        evaluation.main()
    assert not output.exists()


@pytest.fixture
def minimind_cli_fixture(tmp_path, monkeypatch):
    """Exercise CLI wiring with stub model/env constructors and no rollout."""
    from mouse_llm.continuous import evaluation

    captured = {}
    specialist_module = ModuleType("mouse_llm.continuous.specialist")

    class MockSpecialist:
        def __init__(self, checkpoint, *, device):
            self.name = "mock-specialist"

    specialist_module.ContinuousCheckpointPolicy = MockSpecialist
    planner_module = ModuleType("mouse_llm.hierarchical.minimind_planner")

    class MockPlanner:
        def __init__(self, **kwargs):
            captured["planner_arguments"] = kwargs
            self.decoding_mode = kwargs["decoding_mode"]

    planner_module.MiniMindSkillPlanner = MockPlanner
    environment_module = ModuleType("mouse_llm.continuous.environment")
    environment_module.RESET_CONTRACT = "fresh_navigation_episode_reset_v1"
    environment_module.ContinuousBotEvadeEnv = lambda **kwargs: pytest.fail("CLI wiring test must not create a real environment")
    navigation_module = ModuleType("mouse_llm.continuous.navigation")
    navigation_module.navigation_contract = lambda mode: f"graph_route_exact_xy_terminal_{mode}_v2"
    for module in (specialist_module, planner_module, environment_module, navigation_module):
        monkeypatch.setitem(sys.modules, module.__name__, module)

    def no_rollout(env_factory, policy, *, seeds, **kwargs):
        captured["policy_name"] = policy.name
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
    specialist, base, lora = (tmp_path / name for name in ("specialist.pt", "base.pth", "skill_lora.pth"))
    for path in (specialist, base, lora):
        path.write_bytes(("test-fixture-" + path.name).encode())
    manifest = {
        "schema_version": "mousemind_continuous_planner_checkpoint_v1",
        "action_schema": evaluation.ACTION_SCHEMA,
        "context_schema": evaluation.CONTEXT_SCHEMA,
        "skill_lora_sha256": evaluation._sha256(lora),
        "specialist_sha256": evaluation._sha256(specialist),
        "base_weight_sha256": evaluation._sha256(base),
        "max_seq_len": 1536,
        "execution_contract": {"planner_horizon": 8, "memory_steps": 8, "evade_distance": 0.35},
        "training_environment": {"navigation_contract": "graph_route_exact_xy_terminal_project_v2",
                                 "reset_contract": "fresh_navigation_episode_reset_v1",
                                 "destination_mode": "project", "projection_epsilon": 1e-5},
    }
    manifest_path = tmp_path / "planner_manifest.json"
    manifest_path.write_text(json.dumps(manifest))
    output = tmp_path / "evaluation"
    argv = ["evaluation", "--policy", "minimind", "--specialist-checkpoint", str(specialist),
            "--base-weight", str(base), "--skill-lora-weight", str(lora),
            "--planner-manifest", str(manifest_path), "--tokenizer", str(tmp_path / "tokenizer"),
            "--seed-start", "51000", "--episodes", "1", "--device", "cpu",
            "--output-dir", str(output)]
    return {"argv": argv, "output": output, "manifest": manifest, "manifest_path": manifest_path,
            "captured": captured, "weights": (specialist, base, lora)}


@pytest.mark.parametrize("mode", ["greedy_trie", "sequence_logprob"])
def test_evaluation_cli_passes_and_records_inference_decoder_without_changing_weights(minimind_cli_fixture, monkeypatch, capsys, mode):
    from mouse_llm.continuous import evaluation

    fixture = minimind_cli_fixture
    before = {path: path.read_bytes() for path in (*fixture["weights"], fixture["manifest_path"])}
    extra = [] if mode == "greedy_trie" else ["--planner-decoding", mode]
    monkeypatch.setattr(sys, "argv", [*fixture["argv"], *extra])
    evaluation.main()
    report = json.loads(capsys.readouterr().out)
    assert fixture["captured"]["planner_arguments"]["decoding_mode"] == mode
    assert fixture["captured"]["planner_arguments"]["max_seq_len"] == 1536
    assert report["config"]["planner_decoding"] == mode
    assert report["config"]["planner_decoding_ablation"] is (mode != "greedy_trie")
    assert report["config"]["planner_manifest_sha256"] == evaluation._sha256(fixture["manifest_path"])
    assert report["config"]["skill_lora_sha256"] == fixture["manifest"]["skill_lora_sha256"]
    expected_name = "continuous-minimind" + ("-sequence-logprob-ablation" if mode != "greedy_trie" else "")
    assert report["policy"] == expected_name
    assert "inference" in report["config"]["planner_decoding_note"].lower()
    assert "not a training or weight improvement" in report["config"]["planner_decoding_note"]
    assert {path: path.read_bytes() for path in before} == before
    assert "planner_decoding" not in json.loads(fixture["manifest_path"].read_text())


@pytest.mark.parametrize("broken_contract", ["lora_hash", "execution", "environment"])
def test_decoder_ablation_does_not_bypass_checkpoint_or_environment_validation(minimind_cli_fixture, monkeypatch, broken_contract):
    from mouse_llm.continuous import evaluation

    fixture = minimind_cli_fixture
    manifest = fixture["manifest"]
    if broken_contract == "lora_hash":
        manifest["skill_lora_sha256"] = "0" * 64
    elif broken_contract == "execution":
        manifest["execution_contract"]["planner_horizon"] = 4
    else:
        manifest["training_environment"]["projection_epsilon"] = 0.01
    fixture["manifest_path"].write_text(json.dumps(manifest))
    monkeypatch.setattr(sys, "argv", [*fixture["argv"], "--planner-decoding", "sequence_logprob"])
    with pytest.raises(ValueError):
        evaluation.main()
    assert "planner_arguments" not in fixture["captured"]
