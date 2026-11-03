"""Short cached-simulator contract checks, never policy performance evidence."""

import os
from pathlib import Path

import numpy as np
import pytest


def _environment():
    cache = os.environ.get("CELLWORLD_CACHE")
    if not cache or not (Path(cache) / "paths/hexagonal.21_05.astar.robot").is_file():
        pytest.skip("Existing offline Cellworld cache required; no network downloads")
    pytest.importorskip("cellworld")
    from mouse_llm.continuous.environment import ContinuousBotEvadeEnv
    return ContinuousBotEvadeEnv(max_step=4)


def test_real_xy_endpoint_invalid_target_and_seeded_reset():
    from mouse_llm.continuous.replay import replay_fingerprint
    env = _environment()
    try:
        first, _ = env.reset(seed=50000)
        fingerprint = replay_fingerprint(env)
        prey = tuple(env.model.prey.state.location)
        # Select a small free displacement that is not a catalog destination.
        destination = (prey[0] + 0.001, prey[1] + 0.002)
        route = env.model.prey.navigation.get_path(prey, destination)
        assert route[-1] == destination
        assert not any(np.allclose(destination, point, atol=1e-10, rtol=0)
                       for point in env.model.prey.navigation.locations)
        env.step(destination)
        _, _, _, _, info = env.step((0.0, 0.0))
        assert info["invalid_destination"] is True
        second, _ = env.reset(seed=50000)
        assert np.array_equal(first, second)
        assert replay_fingerprint(env) == fingerprint
        assert env.action_space.shape == (2,)
    finally:
        env.close()


def test_real_private_trace_replays_with_continuous_skill_memory():
    from mouse_llm.continuous.contracts import ContinuousDecision
    from mouse_llm.continuous.evaluation import run_episode
    from mouse_llm.continuous.policy import ContinuousHierarchicalPolicy
    from mouse_llm.continuous.posttraining import evaluate_branch
    from mouse_llm.hierarchical.policy import Skill

    class EngineeringSpecialist:
        def reset(self, seed):
            pass

        def act(self, observation):
            return ContinuousDecision((0.5, 0.5))

    env = _environment()
    specialist = EngineeringSpecialist()
    trace = []
    try:
        run_episode(env, ContinuousHierarchicalPolicy(specialist=specialist), seed=50001,
                    trace_hook=trace.append)
    finally:
        env.close()
    assert len(trace) >= 2
    first = evaluate_branch(_environment, trace[1], specialist, Skill.GO_TO_GOAL,
                            horizon=2, candidate_steps=1)
    second = evaluate_branch(_environment, trace[1], specialist, Skill.GO_TO_GOAL,
                             horizon=2, candidate_steps=1)
    assert first == second
