"""Feasible XY projection and honest accounting, using static/toy geometry."""

import math
import random
from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip("gymnasium")
pytest.importorskip("shapely")
import gymnasium as gym
from shapely.geometry import Point, Polygon

from mouse_llm.continuous.contracts import ContinuousDecision
from mouse_llm.continuous.environment import ContinuousBotEvadeEnv
from mouse_llm.continuous.evaluation import run_episode, summarize
from mouse_llm.continuous.navigation import ExactXYNavigation, navigation_contract
from mouse_llm.continuous.policy import DirectContinuousPolicy
from mouse_llm.continuous.posttraining import evaluate_branch
from mouse_llm.hierarchical.policy import Skill


def router(points):
    base = SimpleNamespace(points=SimpleNamespace(point_list=points))
    base.get_path = lambda src, dst: [min(points, key=lambda point: math.dist(point, dst))]
    return base


def navigation(*, arena=None, obstacles=(), points=((0.3, 0.5),), mode="project", epsilon=1e-5):
    return ExactXYNavigation(router(points),
        arena=arena or Polygon([(0.1, 0.1), (0.9, 0.1), (0.9, 0.9), (0.1, 0.9)]),
        occlusions=list(obstacles), destination_mode=mode, projection_epsilon=epsilon)


def test_valid_exact_destination_is_unchanged_even_near_inset_boundary():
    nav = navigation()
    for requested in ((0.33331, 0.51119), (0.100001, 0.5), (0.3, 0.5)):
        result = nav.resolve_destination((0.3, 0.5), requested)
        assert result["executed_xy"] == list(requested)
        assert result["requested_invalid_destination"] is False
        assert result["was_projected"] is False
        assert result["projection_distance"] == 0.0


@pytest.mark.parametrize("target", [(0.0, 0.0), (0.61, 0.5)])
def test_invalid_arena_or_wall_target_gets_continuous_free_space_projection(target):
    wall = Polygon([(0.6, 0.4), (0.8, 0.4), (0.8, 0.6), (0.6, 0.6)])
    nav = navigation(obstacles=[wall])
    result = nav.resolve_destination((0.3, 0.5), target)
    assert result["requested_xy"] == list(target)
    assert result["requested_invalid_destination"] is True
    assert result["original_invalid_reason"] == "destination_outside_free_space"
    assert result["was_projected"] is True and result["invalid_destination"] is False
    projected = tuple(result["executed_xy"])
    assert nav.free_space.contains(Point(projected))
    assert projected not in nav.locations
    assert result["projection_distance"] == pytest.approx(math.dist(target, projected))
    assert nav.get_path((0.3, 0.5), projected)[-1] == projected


def test_projection_never_jumps_to_a_disconnected_free_space_component():
    wall = Polygon([(0.45, 0.0), (0.55, 0.0), (0.55, 1.0), (0.45, 1.0)])
    nav = navigation(arena=Polygon([(0, 0), (1, 0), (1, 1), (0, 1)]),
                     obstacles=[wall], points=((0.2, 0.5), (0.8, 0.5)))
    result = nav.resolve_destination((0.2, 0.5), (0.8, 0.5))
    assert result["requested_invalid_destination"] is True
    assert result["was_projected"] is True
    assert result["executed_xy"][0] == pytest.approx(0.45 - nav.projection_epsilon)
    assert result["executed_xy"][0] < 0.45
    assert result["invalid_destination"] is False


def test_empty_inset_falls_back_to_explicit_hold_without_claiming_projection():
    nav = navigation(arena=Polygon([(0.1, 0.1), (0.10001, 0.1), (0.10001, 0.9), (0.1, 0.9)]),
                     points=((0.100005, 0.5),))
    result = nav.resolve_destination((0.100005, 0.5), (0.8, 0.5))
    assert result["executed_xy"] == [0.100005, 0.5]
    assert result["requested_invalid_destination"] is True and result["invalid_destination"] is True
    assert result["projection_attempted"] is True and result["was_projected"] is False
    assert result["projection_distance"] == 0.0
    assert result["invalid_destination_reason"] == "current_free_space_component_has_empty_inset"


def test_reject_mode_remains_available_and_has_a_distinct_contract():
    nav = navigation(mode="reject")
    result = nav.resolve_destination((0.3, 0.5), (0.0, 0.0))
    assert result["executed_xy"] == [0.3, 0.5]
    assert result["invalid_destination"] is True
    assert result["projection_attempted"] is False and result["was_projected"] is False
    assert nav.contract == navigation_contract("reject")
    assert nav.contract != navigation_contract("project")


class Agent:
    def __init__(self, base):
        self.navigation = base
        self.state = SimpleNamespace(location=(0.3, 0.5), direction=0.0)
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


class ToyNative(gym.Env):
    action_space = gym.spaces.Box(0.0, 1.0, shape=(3,), dtype=np.float32)
    observation_space = gym.spaces.Box(-np.inf, np.inf, shape=(10,), dtype=np.float32)
    frame_stack_k = 1
    max_step = 4

    def __init__(self):
        base = router(((0.3, 0.5),))
        self.model = SimpleNamespace(
            prey=Agent(base), time=0.0, paused=False,
            arena=Polygon([(0.1, 0.1), (0.9, 0.1), (0.9, 0.9), (0.1, 0.9)]),
            occlusions=[Polygon([(0.6, 0.4), (0.8, 0.4), (0.8, 0.6), (0.6, 0.6)])],
            prey_data=Data(), rng=random.Random(0),
        )
        self.closed = False

    def legacy_policy_observation(self):
        xy = self.model.prey.state.location
        return np.asarray([*xy, 0, 0, 0, 0, 1.0-xy[0], 0, 1, 0], dtype=np.float32)

    def reset(self, *, seed=None, options=None):
        self.index = 0
        self.model.prey.state.location = (0.3, 0.5)
        self.model.rng.seed(seed)
        self.model.prey_data.reset()
        return self.legacy_policy_observation(), {"captures": 0}

    def step(self, action):
        self.index += 1
        self.model.time += 0.25
        self.model.prey.state.location = tuple(action[:2])
        self.model.rng.random()
        return self.legacy_policy_observation(), 0.0, False, self.index >= self.max_step, {
            "captures": 0, "is_success": False,
        }

    def close(self):
        self.closed = True


class WallTargetPolicy:
    def reset(self, seed):
        pass
    def act(self, observation):
        return ContinuousDecision((0.61, 0.5))


def test_projection_statistics_and_raw_requested_prefix_replay_are_preserved():
    def factory():
        return ContinuousBotEvadeEnv(ToyNative())
    trace = []
    env = factory()
    try:
        row = run_episode(env, DirectContinuousPolicy(WallTargetPolicy()), seed=50000, trace_hook=trace.append)
    finally:
        env.close()
    assert row["requested_invalid_rate"] == 1.0
    assert row["projection_rate"] == 1.0
    assert row["destination_rejection_rate"] == 0.0
    assert row["mean_projection_distance"] > 0
    assert trace[1]["prefix_xy_actions"] == [[0.61, 0.5]]
    assert trace[0]["requested_xy"] == [0.61, 0.5]
    assert trace[0]["executed_xy"] != trace[0]["requested_xy"]
    assert trace[0]["action_xy"] == trace[0]["requested_xy"]
    summary = summarize([row])
    for key in ("requested_invalid_rate", "projection_rate", "mean_projection_distance"):
        assert summary["metrics"][key] == row[key]
    first = evaluate_branch(factory, trace[1], WallTargetPolicy(), Skill.GO_TO_GOAL,
                            horizon=2, candidate_steps=1)
    second = evaluate_branch(factory, trace[1], WallTargetPolicy(), Skill.GO_TO_GOAL,
                             horizon=2, candidate_steps=1)
    assert first == second
