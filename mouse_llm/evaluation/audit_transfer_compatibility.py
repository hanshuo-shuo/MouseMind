"""Fail-closed public compatibility audit for BotEvade-to-Oasis transfer."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from mouse_llm.evaluation.contracts import (
    DEFAULT_TRANSFER_CONTRACT,
    load_transfer_contract,
    transfer_contract_seeds,
)
from mouse_llm.evaluation.evaluate_policy import load_action_catalog


def coordinate_sha256(values: Sequence[Sequence[float]]) -> str:
    coordinates = np.asarray(values, dtype="<f8")
    if coordinates.ndim != 2 or coordinates.shape[1] != 2:
        raise ValueError("Action coordinates must have shape [actions, 2]")
    if not np.isfinite(coordinates).all():
        raise ValueError("Action coordinates must be finite")
    return hashlib.sha256(coordinates.tobytes()).hexdigest()


def _environment_coordinates(env: Any) -> np.ndarray:
    return np.asarray(
        [[float(point[0]), float(point[1])] for point in env.action_list],
        dtype=np.float64,
    )


def _snapshot_oasis(env: Any, seed: int) -> dict[str, Any]:
    observation, _ = env.reset(seed=seed)
    return {
        "observation": np.asarray(observation).copy(),
        "legacy": env.legacy_policy_observation().copy(),
        "transfer": env.transfer_policy_observation().copy(),
        "prey": tuple(float(value) for value in env.model.prey.state.location),
        "predator": tuple(
            float(value) for value in env.model.predator.state.location
        ),
        "goal_order": tuple(env.sampled_goal_order),
    }


def _snapshots_equal(left: dict[str, Any], right: dict[str, Any]) -> bool:
    return all(
        np.array_equal(left[name], right[name])
        for name in ("observation", "legacy", "transfer")
    ) and all(
        left[name] == right[name]
        for name in ("prey", "predator", "goal_order")
    )


def _verify_terminal_contract(
    contract: dict[str, Any], contract_path: Path
) -> dict[str, Any]:
    from mouse_llm.envs.mice import OasisEnv

    env = OasisEnv(
        world_name=contract["target"]["world"],
        goal_locations=[
            tuple(values) for values in contract["target"]["goal_locations"]
        ],
        goal_threshold=contract["target"]["goal_threshold"],
        use_lppos=False,
        use_predator=False,
        frame_stack_k=1,
        max_step=contract["max_steps"],
        time_step=contract["time_step"],
    )
    try:
        env.reset(
            seed=transfer_contract_seeds(
                "compatibility_audit", contract_path
            )[0]
        )
        terminal_info: dict[str, Any] = {}
        steps = 0
        while True:
            goal = np.asarray(env.model.goal_location, dtype=np.float64)
            destinations = _environment_coordinates(env)
            action = int(np.linalg.norm(destinations - goal, axis=1).argmin())
            _, _, terminated, truncated, terminal_info = env.step(action)
            steps += 1
            if terminated or truncated:
                break
        expected_goals = int(contract["target"]["sampled_goal_count"])
        expected_objectives = int(contract["target"]["objective_count"])
        checks = {
            "terminated_without_truncation": bool(terminated and not truncated),
            "task_success": int(terminal_info.get("is_success", 0)) == 1,
            "all_sampled_goals_completed": int(
                terminal_info.get("goals_completed", -1)
            )
            == expected_goals,
            "return_completed": int(terminal_info.get("return_completed", 0)) == 1,
            "all_objectives_completed": int(
                terminal_info.get("objectives_completed", -1)
            )
            == expected_objectives,
            "capture_free": int(terminal_info.get("captures", -1)) == 0,
            "survival_independent_of_termination": int(
                terminal_info.get("survived", 0)
            )
            == 1,
        }
        if not all(checks.values()):
            raise ValueError(f"Oasis terminal contract failed: {checks}")
        return {"verified": True, "steps": steps, "checks": checks}
    finally:
        env.close()


def run_audit(
    *, contract_path: Path, action_catalog_path: Path
) -> dict[str, Any]:
    contract = load_transfer_contract(contract_path)
    from mouse_llm.envs.mice import BotEvadeEnv, OasisEnv

    common = {
        "use_lppos": False,
        "use_predator": True,
        "frame_stack_k": 1,
        "max_step": 2,
        "time_step": contract["time_step"],
    }
    source = BotEvadeEnv(world_name=contract["source"]["world"], **common)
    target = OasisEnv(
        world_name=contract["target"]["world"],
        goal_locations=[
            tuple(values) for values in contract["target"]["goal_locations"]
        ],
        goal_threshold=contract["target"]["goal_threshold"],
        **common,
    )
    try:
        catalog = load_action_catalog(action_catalog_path)
        source_actions = _environment_coordinates(source)
        target_actions = _environment_coordinates(target)
        expected = contract["action_contract"]
        action_checks = {
            "declared_count": len(catalog) == int(expected["action_count"]),
            "source_count": len(source_actions) == int(expected["action_count"]),
            "target_count": len(target_actions) == int(expected["action_count"]),
            "catalog_matches_source": np.array_equal(catalog, source_actions),
            "source_matches_target": np.array_equal(source_actions, target_actions),
            "coordinate_sha256": coordinate_sha256(catalog)
            == expected["coordinate_sha256"],
        }
        if not all(action_checks.values()):
            raise ValueError(f"Action compatibility failed: {action_checks}")

        deterministic_seeds = []
        goal_orders = []
        for seed in transfer_contract_seeds(
            "compatibility_audit", contract_path
        ):
            first = _snapshot_oasis(target, seed)
            second = _snapshot_oasis(target, seed)
            deterministic = _snapshots_equal(first, second)
            deterministic_seeds.append(int(seed))
            goal_orders.append(first["goal_order"])
            if not deterministic:
                raise ValueError(f"Oasis reset is not deterministic for seed {seed}")
        if len(set(goal_orders)) < 2:
            raise ValueError("Compatibility seeds do not exercise distinct goal orders")

        target.reset(seed=deterministic_seeds[0])
        legacy = target.legacy_policy_observation()
        transfer = target.transfer_policy_observation()
        compatible_goals = np.asarray(
            [
                *contract["target"]["goal_locations"],
                target.model.start_location,
            ],
            dtype=np.float64,
        )
        goal_action_errors = np.asarray(
            [
                np.linalg.norm(target_actions - goal, axis=1).min()
                for goal in compatible_goals
            ]
        )
        default_projection = contract["target"]["discrete_goal_projection"]
        default_goal_error = float(
            np.linalg.norm(
                target_actions - np.asarray(default_projection["default_location"]),
                axis=1,
            ).min()
        )
        observation_checks = {
            "legacy_shape_10": legacy.shape == (10,),
            "transfer_shape_18": transfer.shape == (18,),
            "finite": bool(np.isfinite(legacy).all() and np.isfinite(transfer).all()),
            "legacy_prefix_matches_transfer": bool(
                np.allclose(legacy[:2], transfer[:2])
                and math.isclose(float(legacy[6]), float(transfer[14]))
            ),
            "all_frozen_goals_reachable": float(goal_action_errors.max())
            <= float(contract["target"]["goal_threshold"]),
            "default_goal_mismatch_reproduced": math.isclose(
                default_goal_error,
                float(default_projection["projection_distance"]),
                abs_tol=1e-12,
            ),
        }
        if not all(observation_checks.values()):
            raise ValueError(
                f"Observation compatibility failed: {observation_checks}"
            )
    finally:
        source.close()
        target.close()

    terminal = _verify_terminal_contract(contract, contract_path)
    return {
        "schema_version": 1,
        "artifact": "cross_task_transfer_compatibility_audit",
        "contract_name": contract["contract_name"],
        "contract_sha256": hashlib.sha256(contract_path.read_bytes()).hexdigest(),
        "action_contract": {
            "verified": True,
            "coordinate_sha256": coordinate_sha256(catalog),
            "checks": action_checks,
        },
        "observation_contract": {
            "verified": True,
            "checks": observation_checks,
            "literal_low_level_transfer": contract["observation_contract"][
                "literal_low_level_transfer"
            ],
            "planner_isolation_transfer": contract["observation_contract"][
                "planner_isolation_transfer"
            ],
            "maximum_frozen_goal_action_error": float(goal_action_errors.max()),
            "default_goal_action_error": default_goal_error,
        },
        "reset_contract": {
            "verified": True,
            "seed_count": len(deterministic_seeds),
            "unique_goal_order_count": len(set(goal_orders)),
        },
        "terminal_contract": terminal,
        "private_data_used": False,
        "research_evidence": True,
        "research_evidence_blockers": [],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, default=DEFAULT_TRANSFER_CONTRACT)
    parser.add_argument(
        "--action-catalog",
        type=Path,
        default=Path("mouse_llm/envs/mice/assets/action_catalog_21_05.json"),
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = run_audit(
        contract_path=args.contract, action_catalog_path=args.action_catalog
    )
    args.output.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(temporary, args.output)
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
