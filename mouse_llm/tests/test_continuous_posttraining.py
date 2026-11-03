"""Counterfactual-training contracts, using deterministic toy dynamics only."""

import copy
import json
import random
import sys
from types import ModuleType, SimpleNamespace

import pytest

np = pytest.importorskip("numpy")
pytest.importorskip("torch")

from mouse_llm.continuous import posttraining
from mouse_llm.continuous.context import CONTEXT_SCHEMA, ContinuousContextBuilder
from mouse_llm.continuous.contracts import ACTION_SCHEMA, ContinuousDecision, validate_xy
from mouse_llm.continuous.posttraining import build_rows, evaluate_branch, outcome_utility
from mouse_llm.continuous.replay import replay_fingerprint
from mouse_llm.data.planner_schema import Preference
from mouse_llm.hierarchical.policy import Skill


class ToySpecialist:
    def __init__(self):
        self.calls = []

    def reset(self, seed):
        self.seed = seed
        self.calls = []

    def act(self, observation):
        assert np.asarray(observation).shape == (10,)
        self.calls.append(np.asarray(observation).copy())
        return ContinuousDecision((0.73, 0.26))


class ToyEnvironment:
    """Public 10D observation plus hidden physical state and seeded RNG."""

    max_step = 30

    def __init__(self, *, visible_steps=(), hidden_predator=(0.8, 0.5), capture_step=None):
        self.visible_steps = tuple(visible_steps)
        self.hidden_predator = tuple(hidden_predator)
        self.capture_step = capture_step
        self.last_step = 0.0
        self.episode_count = 0
        self.closed = False

    @property
    def unwrapped(self):
        return self

    def reset(self, *, seed):
        self.seed, self.step_index = seed, 0
        self.episode_count += 1
        self.actions = []
        self.puffed = False
        self.model = SimpleNamespace(
            time=0.0,
            last_step=0.0,
            running=True,
            rng=random.Random(seed),
            prey_data=SimpleNamespace(puff_count=0, goal_achieved=False),
            prey=SimpleNamespace(
                state=SimpleNamespace(location=(0.5, 0.5), direction=0.0),
                dynamics=SimpleNamespace(forward_speed=0.0, turn_speed=0.0),
                destination=None, new_destination=None, path=[],
                destination_wait=0, navigation_plan_update_wait=0,
                active_navigation=True, last_step=0.0,
            ),
            predator=SimpleNamespace(
                state=SimpleNamespace(location=self.hidden_predator, direction=0.5),
                dynamics=SimpleNamespace(forward_speed=0.1, turn_speed=0.0),
                destination=(0.5, 0.5), path=[(0.5, 0.5)],
            ),
        )
        return self.legacy_policy_observation(), {"captures": 0}

    def legacy_policy_observation(self):
        x, y = self.model.prey.state.location
        visible = self.step_index in self.visible_steps
        predator = self.model.predator.state.location if visible else (0.0, 0.0)
        return np.asarray([
            x, y, 0.0, *predator, 0.5 if visible else 0.0,
            1.0 - x, float(self.puffed), 1.0, 0.0,
        ], dtype=np.float64)

    def step(self, action):
        xy = validate_xy(action)
        self.actions.append(xy)
        self.step_index += 1
        self.model.time += 0.25
        self.model.rng.random()
        self.model.prey.state.location = xy
        self.model.prey.destination = xy
        self.model.prey.path = [xy]
        self.puffed = self.step_index == self.capture_step
        self.model.prey_data.puff_count += int(self.puffed)
        truncated = self.step_index >= self.max_step
        return self.legacy_policy_observation(), 0.0, False, truncated, {
            "captures": self.model.prey_data.puff_count, "is_success": False,
        }

    def close(self):
        self.closed = True


def _anchor(*, seed=50000, prefix=(), factory=ToyEnvironment):
    env = factory()
    builder = ContinuousContextBuilder()
    observation, _ = env.reset(seed=seed)
    for action in prefix:
        builder.observe(observation)
        builder.record_decision(action, Skill.GO_TO_GOAL)
        observation, _, _, _, _ = env.step(action)
    return {
        "seed": seed,
        "anchor_id": f"toy-{seed}-{len(prefix)}",
        "action_schema": ACTION_SCHEMA,
        "prefix_xy_actions": [list(action) for action in prefix],
        "environment_observation": observation.tolist(),
        "legacy_observation": env.legacy_policy_observation().tolist(),
        "replay_fingerprint": replay_fingerprint(env),
        "environment_config": {
            "world": "21_05", "max_steps": 300, "time_step": 0.25,
            "predator_speed_ratio": 0.15, "max_line_of_sight_distance": 1.0,
            "navigation_contract": "graph_route_exact_xy_terminal_v1",
            "reset_contract": "fresh_navigation_episode_reset_v1",
        },
        "context": builder.observe(observation).payload(),
    }


def _record():
    return {
        "seed": 50000, "anchor_id": "toy-record", "context": _anchor()["context"],
        "branches": {
            Skill.GO_TO_GOAL.value: {"captures": 0, "success": 1, "goal_progress": 0.4},
            Skill.EVADE_PREDATOR.value: {"captures": 0, "success": 0, "goal_progress": 0.1},
            Skill.HOLD_POSITION.value: {"captures": 1, "success": 0, "goal_progress": 0.0},
        },
    }


def test_chosen_sft_and_rejected_dpo_share_the_exact_scalar_utility():
    record = _record()
    rows = build_rows(record, minimum_margin=0.01)
    assert {row["preference"] for row in rows} == {preference.value for preference in Preference}
    for row in rows:
        preference = Preference(row["preference"])
        expected = {
            skill.value: outcome_utility(record["branches"][skill.value], preference, skill)
            for skill in Skill
        }
        chosen = json.loads(row["chosen"])["skill"]
        rejected = json.loads(row["rejected"])["skill"]
        assert expected[chosen] == max(expected.values())
        assert expected[rejected] == min(expected.values())
        assert row["utilities"] == expected
        assert row["target_skill"] == chosen
        assert row["conversations"][-1] == {"role": "assistant", "content": row["chosen"]}
        assert row["margin"] == pytest.approx(expected[chosen] - expected[rejected])


@pytest.mark.parametrize("second_progress", [0.3, 0.2995])
def test_best_runner_up_tie_or_near_tie_is_filtered_despite_large_worst_gap(second_progress):
    record = _record()
    record["branches"][Skill.GO_TO_GOAL.value] = {"captures": 0, "success": 0, "goal_progress": 0.3}
    record["branches"][Skill.EVADE_PREDATOR.value]["goal_progress"] = second_progress
    rows = build_rows(record, minimum_margin=0.01)
    # HOLD is an explicit instruction with its own unambiguous target.
    assert rows and {row["preference"] for row in rows} == {Preference.HOLD.value}


@pytest.mark.parametrize("key,value", [
    ("schema_version", "mousemind_planner_context_v1"),
    ("action_schema", "discrete_295"),
])
def test_legacy_planner_context_is_rejected(key, value):
    record = _record()
    record["context"][key] = value
    with pytest.raises(ValueError, match="discrete planner contexts"):
        build_rows(record, minimum_margin=0.01)


def test_prompt_keeps_new_context_schema_and_excludes_replay_state():
    record = _record()
    record["replay_fingerprint"] = "privileged-state-hash"
    record["simulator_state"] = {"hidden_predator": [0.2, 0.3]}
    rows = build_rows(record, minimum_margin=0.01)
    for row in rows:
        payload = json.loads(row["prompt"][1]["content"])
        assert payload["context"]["schema_version"] == CONTEXT_SCHEMA
        assert payload["context"]["action_schema"] == ACTION_SCHEMA
        assert "recent_xy_actions" in payload["context"]["history"]
        serialized = json.dumps(row["prompt"])
        assert "privileged-state-hash" not in serialized
        assert "simulator_state" not in serialized


def test_repeated_seeded_branches_match_outcomes_and_fingerprints():
    anchor = _anchor(prefix=((0.43, 0.61),))
    first = evaluate_branch(ToyEnvironment, anchor, ToySpecialist(), Skill.GO_TO_GOAL, horizon=5, candidate_steps=2)
    second = evaluate_branch(ToyEnvironment, anchor, ToySpecialist(), Skill.GO_TO_GOAL, horizon=5, candidate_steps=2)
    assert first == second
    assert first["steps"] == 5
    assert len(first["final_replay_fingerprint"]) == 64


def test_replay_rejects_changed_hidden_physical_state_even_when_observations_match():
    anchor = _anchor()
    other = ToyEnvironment(hidden_predator=(0.7, 0.6))
    observation, _ = other.reset(seed=anchor["seed"])
    assert observation.tolist() == anchor["environment_observation"]
    instances = []

    def factory():
        env = ToyEnvironment(hidden_predator=(0.7, 0.6))
        instances.append(env)
        return env

    with pytest.raises(ValueError, match="replay verification"):
        evaluate_branch(factory, anchor, ToySpecialist(), Skill.GO_TO_GOAL, horizon=3, candidate_steps=1)
    assert instances[0].closed


def test_replay_rejects_missing_fingerprint_and_observation_mismatch():
    anchor = _anchor()
    missing = copy.deepcopy(anchor)
    del missing["replay_fingerprint"]
    with pytest.raises(ValueError, match="fingerprint"):
        evaluate_branch(ToyEnvironment, missing, ToySpecialist(), Skill.GO_TO_GOAL, horizon=3, candidate_steps=1)
    altered = copy.deepcopy(anchor)
    altered["environment_observation"][0] += 0.1
    with pytest.raises(ValueError, match="replay verification"):
        evaluate_branch(ToyEnvironment, altered, ToySpecialist(), Skill.GO_TO_GOAL, horizon=3, candidate_steps=1)


def test_candidate_runs_k_steps_then_common_public_goal_continuation():
    instances = []

    def factory():
        env = ToyEnvironment()
        instances.append(env)
        return env

    specialist = ToySpecialist()
    outcome = evaluate_branch(factory, _anchor(), specialist, Skill.HOLD_POSITION, horizon=5, candidate_steps=2)
    assert outcome["steps"] == 5
    assert instances[0].actions == [(0.5, 0.5)] * 2 + [(0.73, 0.26)] * 3
    assert len(specialist.calls) == 3
    assert instances[0].closed


def test_public_continuation_changes_to_evasion_when_predator_becomes_visible():
    instances = []

    def factory():
        env = ToyEnvironment(visible_steps=(2, 3, 4))
        instances.append(env)
        return env

    specialist = ToySpecialist()
    anchor = _anchor(factory=factory)
    instances.clear()
    evaluate_branch(factory, anchor, specialist, Skill.HOLD_POSITION, horizon=4, candidate_steps=2)
    assert instances[0].actions[:2] == [(0.5, 0.5)] * 2
    assert instances[0].actions[2] == pytest.approx((0.15, 0.5))
    assert specialist.calls == []


@pytest.mark.parametrize("candidate_steps,horizon", [(0, 4), (5, 4), (1, 0)])
def test_branch_rejects_invalid_candidate_window(candidate_steps, horizon):
    with pytest.raises(ValueError, match="candidate_steps"):
        evaluate_branch(ToyEnvironment, _anchor(), ToySpecialist(), Skill.HOLD_POSITION,
                        horizon=horizon, candidate_steps=candidate_steps)


def test_fingerprint_ignores_declared_wall_clock_and_lifetime_telemetry():
    env = ToyEnvironment()
    env.reset(seed=7)
    expected = replay_fingerprint(env)
    env.last_step = 1793660000.1
    env.model.last_step = 1793660002.2
    env.model.prey.last_step = 1793660004.3
    env.episode_count += 10
    assert replay_fingerprint(env) == expected


@pytest.mark.parametrize("mutation", ["position", "dynamics", "navigation", "capture", "rng"])
def test_fingerprint_depends_on_physical_control_and_rng_state(mutation):
    env = ToyEnvironment()
    env.reset(seed=7)
    expected = replay_fingerprint(env)
    if mutation == "position":
        env.model.predator.state.location = (0.75, 0.45)
    elif mutation == "dynamics":
        env.model.prey.dynamics.forward_speed = 0.12
    elif mutation == "navigation":
        env.model.prey.path.append((0.7, 0.6))
    elif mutation == "capture":
        env.model.prey_data.puff_count += 1
    else:
        env.model.rng.random()
    assert replay_fingerprint(env) != expected


@pytest.mark.parametrize("bad_pool", ["development", "final_id_test"])
def test_main_rejects_development_and_final_seeds_before_any_branch(tmp_path, monkeypatch, bad_pool):
    # Mock both dynamic imports: this test must never create a real simulator
    # or load a real model checkpoint.
    environment_module = ModuleType("mouse_llm.continuous.environment")
    environment_module.ContinuousBotEvadeEnv = lambda **kwargs: pytest.fail("Simulator must not be constructed")
    specialist_module = ModuleType("mouse_llm.continuous.specialist")
    specialist_module.ContinuousCheckpointPolicy = lambda *args, **kwargs: ToySpecialist()
    monkeypatch.setitem(sys.modules, environment_module.__name__, environment_module)
    monkeypatch.setitem(sys.modules, specialist_module.__name__, specialist_module)
    contract = json.loads(posttraining.EXPERIMENT.read_text())
    train_seed = contract["seed_pools"]["data_collection"]["start"]
    forbidden_seed = contract["seed_pools"][bad_pool]["start"]
    trace = tmp_path / "trace.jsonl"
    trace.write_text("".join(json.dumps(_anchor(seed=seed)) + "\n" for seed in (train_seed, forbidden_seed)))
    checkpoint = tmp_path / "unused.pt"
    checkpoint.write_bytes(b"synthetic-placeholder")
    monkeypatch.setattr("sys.argv", [
        "posttraining", "--trace", str(trace), "--specialist-checkpoint", str(checkpoint),
        "--output-dir", str(tmp_path / "output"), "--max-anchors", "1",
    ])
    monkeypatch.setattr(posttraining, "evaluate_branch", lambda *args, **kwargs: pytest.fail("No branch before seed audit"))
    with pytest.raises(ValueError, match="data_collection"):
        posttraining.main()


def test_seed_split_assigns_every_anchor_of_a_seed_to_one_partition():
    first, second = _record(), _record()
    second["anchor_id"] = "another-anchor-same-seed"
    left = {row["split"] for row in build_rows(first, minimum_margin=0.01)}
    right = {row["split"] for row in build_rows(second, minimum_margin=0.01)}
    assert left == right
    assert left in ({"train"}, {"validation"}, {"seen_test", "unseen_test"})
