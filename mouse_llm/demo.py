"""Dependency-light public smoke demo for the closed-loop evaluator.

This is intentionally a synthetic engineering check, not Cellworld research
evidence. It lets reviewers exercise the seeded/paired reporting path without
private trajectories, model weights, or a downloaded Cellworld cache.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from mouse_llm.evaluation.closed_loop import (
    PolicyDecision,
    RandomPolicy,
    _write_outputs,
    build_report,
    run_policy,
)
from mouse_llm.evaluation.evaluate_policy import load_action_catalog


class SyntheticArena:
    """Small deterministic arena with the same 295 destination actions."""

    def __init__(self, destinations: np.ndarray, *, max_steps: int = 40):
        self.destinations = destinations
        self.max_steps = max_steps

    def reset(self, seed=None):
        self.rng = np.random.default_rng(seed)
        self.prey = np.asarray([0.03, 0.50], dtype=np.float64)
        self.predator = np.asarray(
            [0.62, float(self.rng.uniform(0.15, 0.85))], dtype=np.float64
        )
        self.steps = 0
        self.captures = 0
        return self._observation(), {}

    def goal_distance(self, observation):
        return float(np.linalg.norm(np.asarray([1.0, 0.5]) - observation[:2]))

    def _observation(self):
        distance = np.linalg.norm(self.predator - self.prey)
        predator_visible = float(distance < 0.45)
        return np.asarray(
            [
                self.prey[0],
                self.prey[1],
                0.0,
                predator_visible,
                self.predator[0] if predator_visible else 0.0,
                self.predator[1] if predator_visible else 0.0,
                0.0,
                float(self.prey[0] < 0.08 or self.prey[0] > 0.92),
                0.0,
                float(self.steps if predator_visible else -1),
            ],
            dtype=np.float32,
        )

    @staticmethod
    def _advance(origin: np.ndarray, destination: np.ndarray, distance: float):
        delta = destination - origin
        norm = float(np.linalg.norm(delta))
        if norm == 0:
            return origin
        return origin + delta / norm * min(norm, distance)

    def step(self, action):
        if not 0 <= int(action) < len(self.destinations):
            raise ValueError("action outside synthetic arena catalog")
        self.steps += 1
        self.prey = self._advance(self.prey, self.destinations[int(action)], 0.08)
        self.predator = self._advance(self.predator, self.prey, 0.012)
        if np.linalg.norm(self.predator - self.prey) < 0.035:
            self.captures += 1
        success = self.goal_distance(self._observation()) < 0.06
        truncated = self.steps >= self.max_steps and not success
        reward = -0.01 + float(success) - float(self.captures > 0)
        info = {}
        if success or truncated:
            info = {
                "captures": self.captures,
                "is_success": int(success),
                "survived": int(self.captures == 0),
            }
        return self._observation(), reward, success, truncated, info

    def close(self):
        pass


class GoalSpecialist:
    name = "synthetic-goal-specialist"

    def __init__(self, destinations: np.ndarray):
        goal = np.asarray([1.0, 0.5])
        self.action = int(np.linalg.norm(destinations - goal, axis=1).argmin())

    def reset(self, seed: int):
        del seed

    def act(self, observation):
        del observation
        return PolicyDecision(self.action)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the public MouseMind smoke demo")
    parser.add_argument(
        "--output-dir", type=Path, default=Path("/tmp/mousemind-public-demo")
    )
    parser.add_argument("--episodes", type=int, default=12)
    parser.add_argument("--seed", type=int, default=7000)
    parser.add_argument(
        "--action-catalog",
        type=Path,
        default=Path(__file__).parent
        / "envs/mice/assets/action_catalog_21_05.json",
    )
    args = parser.parse_args()
    if args.episodes <= 0:
        raise ValueError("episodes must be positive")
    destinations = load_action_catalog(args.action_catalog)

    def env_factory():
        return SyntheticArena(destinations)

    seeds = list(range(args.seed, args.seed + args.episodes))
    policies = [
        RandomPolicy(action_count=len(destinations)),
        GoalSpecialist(destinations),
    ]
    results = {
        policy.name: run_policy(
            env_factory,
            policy,
            seeds=seeds,
            control_budget_seconds=0.25,
        )
        for policy in policies
    }
    metadata = {
        "synthetic": True,
        "research_evidence": False,
        "purpose": "public evaluator smoke test only",
        "episode_count": args.episodes,
        "seed_start": args.seed,
    }
    report = build_report(
        results,
        seed=args.seed,
        reference_policy="random",
        metadata=metadata,
    )
    _write_outputs(args.output_dir, report, results)
    print(json.dumps(report, indent=2, sort_keys=True))
    print(f"REPORT={args.output_dir / 'closed_loop_metrics.json'}")


if __name__ == "__main__":
    main()
