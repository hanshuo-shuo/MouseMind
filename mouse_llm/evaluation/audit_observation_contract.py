"""Verify the recovered 2025 observation contract against source, data, and replay."""

from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import json
import math
import os
import subprocess
from pathlib import Path
from typing import Any

import numpy as np


CONTRACT_PATH = (
    Path(__file__).resolve().parents[1] / "data/legacy_observation_contract.json"
)


def _sha256_bytes(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def _source_fields(content: str) -> tuple[str, ...]:
    tree = ast.parse(content)
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == "BotEvadeObservation":
            for statement in node.body:
                if not isinstance(statement, ast.Assign):
                    continue
                if any(
                    isinstance(target, ast.Name) and target.id == "fields"
                    for target in statement.targets
                ):
                    value = ast.literal_eval(statement.value)
                    return tuple(str(item) for item in value)
    raise ValueError("BotEvadeObservation.fields was not found in legacy source")


def audit_legacy_source(source: Path, contract: dict[str, Any]) -> dict[str, Any]:
    content = source.read_bytes()
    source_hash = _sha256_bytes(content)
    expected_hash = contract["source"]["sha256"]
    expected_fields = tuple(field["name"] for field in contract["fields"])
    parsed_fields = _source_fields(content.decode("utf-8"))
    commit = contract["source"]["commit"]
    committed_content = subprocess.run(
        ["git", "show", f"{commit}:{contract['source']['path']}"],
        cwd=source.parent,
        check=True,
        capture_output=True,
    ).stdout
    committed_hash = _sha256_bytes(committed_content)
    checks = {
        "working_copy_sha256_matches": source_hash == expected_hash,
        "committed_blob_sha256_matches": committed_hash == expected_hash,
        "field_order_matches": parsed_fields == expected_fields,
    }
    return {
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "source_sha256": source_hash,
        "committed_blob_sha256": committed_hash,
        "commit": commit,
        "field_order": list(parsed_fields),
    }


def _parse_vector(raw: str, *, row: int, column: str) -> np.ndarray:
    try:
        values = np.asarray(json.loads(raw), dtype=np.float64)
    except (json.JSONDecodeError, TypeError, ValueError) as exc:
        raise ValueError(f"row {row}: invalid {column}") from exc
    if values.shape != (10,) or not np.isfinite(values).all():
        raise ValueError(f"row {row}: {column} must contain ten finite values")
    return values


def audit_dataset_distribution(path: Path) -> dict[str, Any]:
    digest = hashlib.sha256()
    with path.open("rb") as binary_handle:
        for chunk in iter(lambda: binary_handle.read(1024 * 1024), b""):
            digest.update(chunk)
    minima = np.full(10, np.inf, dtype=np.float64)
    maxima = np.full(10, -np.inf, dtype=np.float64)
    zeros = np.zeros(10, dtype=np.int64)
    row_count = 0
    hidden_count = 0
    visible_count = 0
    hidden_sentinel_mismatches = 0
    binary_mismatches = {"puffed": 0, "finished": 0}
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row_number, row in enumerate(reader, start=2):
            values = _parse_vector(row["obs"], row=row_number, column="obs")
            minima = np.minimum(minima, values)
            maxima = np.maximum(maxima, values)
            zeros += values == 0
            predator_hidden = values[3] == 0 and values[4] == 0
            if predator_hidden:
                hidden_count += 1
                hidden_sentinel_mismatches += int(values[5] != 0)
            else:
                visible_count += 1
                hidden_sentinel_mismatches += int(values[3] == 0 or values[4] == 0)
            binary_mismatches["puffed"] += int(values[7] not in (0.0, 1.0))
            binary_mismatches["finished"] += int(values[9] not in (0.0, 1.0))
            row_count += 1
    if row_count == 0:
        raise ValueError("legacy dataset is empty")
    tolerance = 1e-4
    checks = {
        "ten_finite_fields": True,
        "prey_position_unit_arena": bool(
            np.all(minima[:2] >= -tolerance) and np.all(maxima[:2] <= 1 + tolerance)
        ),
        "signed_angle_range": bool(
            np.all(minima[[2, 5]] >= -math.pi - tolerance)
            and np.all(maxima[[2, 5]] <= math.pi + tolerance)
        ),
        "predator_hidden_zero_sentinel": hidden_sentinel_mismatches == 0,
        "visible_and_hidden_coverage": hidden_count > 0 and visible_count > 0,
        "goal_distance_range": bool(
            minima[6] >= -tolerance and maxima[6] <= math.sqrt(2) + tolerance
        ),
        "puffed_binary": binary_mismatches["puffed"] == 0,
        "cooldown_range": bool(
            minima[8] >= -tolerance and maxima[8] <= 1.0 + tolerance
        ),
        "finished_binary": binary_mismatches["finished"] == 0,
    }
    return {
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "row_count": row_count,
        "source_sha256": digest.hexdigest(),
        "predator_visibility": {
            "hidden_rows": hidden_count,
            "visible_rows": visible_count,
        },
        "field_aggregates": [
            {
                "index": index,
                "min": float(minima[index]),
                "max": float(maxima[index]),
                "zero_rate": float(zeros[index] / row_count),
            }
            for index in range(10)
        ],
    }


def reference_legacy_observation(model: Any) -> np.ndarray:
    predator_visible = bool(model.use_predator and model.prey_data.predator_visible)
    return np.asarray(
        [
            model.prey.state.location[0],
            model.prey.state.location[1],
            math.radians(model.prey.state.direction),
            model.predator.state.location[0] if predator_visible else 0.0,
            model.predator.state.location[1] if predator_visible else 0.0,
            math.radians(model.predator.state.direction) if predator_visible else 0.0,
            model.prey_data.prey_goal_distance,
            model.prey_data.puffed,
            model.puff_cool_down,
            not model.running,
        ],
        dtype=np.float32,
    )


def audit_state_replay(
    *,
    world: str,
    cases_per_visibility: int,
    seed: int,
) -> dict[str, Any]:
    from mouse_llm.envs.mice import BotEvadeEnv, LEGACY_POLICY_FIELDS
    from mouse_llm.envs.mice._vendor.cellworld_game import AgentState

    env = BotEvadeEnv(
        world_name=world,
        use_lppos=False,
        use_predator=True,
        frame_stack_k=1,
        max_step=20,
    )
    rng = np.random.default_rng(seed)
    field_errors: dict[str, list[float]] = {
        field: [] for field in LEGACY_POLICY_FIELDS
    }
    coverage = {"visible": 0, "hidden": 0}
    angle_boundary_cases: set[int] = set()
    try:
        locations = list(env.loader.open_locations)
        candidates: list[tuple[tuple[float, float], tuple[float, float], int, int]] = []
        directions = (-180, -90, 0, 90, 180)
        for _ in range(min(5000, len(locations) ** 2)):
            prey_location = locations[int(rng.integers(0, len(locations)))]
            predator_location = locations[int(rng.integers(0, len(locations)))]
            prey_direction = int(directions[int(rng.integers(0, len(directions)))])
            predator_direction = int(directions[int(rng.integers(0, len(directions)))])
            candidates.append(
                (
                    prey_location,
                    predator_location,
                    prey_direction,
                    predator_direction,
                )
            )
        for case_index, (
            prey_location,
            predator_location,
            prey_direction,
            predator_direction,
        ) in enumerate(candidates):
            if min(coverage.values()) >= cases_per_visibility:
                break
            env.reset(seed=seed + case_index)
            env.model.set_agents_state(
                {
                    "prey": AgentState(prey_location, prey_direction),
                    "predator": AgentState(predator_location, predator_direction),
                }
            )
            env.model.prey_data.reset()
            env.model.puff_cool_down = 0
            env.model.__update_state__(delta_t=0)
            env.__update_observation__()
            visibility = (
                "visible" if env.model.prey_data.predator_visible else "hidden"
            )
            if coverage[visibility] >= cases_per_visibility:
                continue
            actual = env.legacy_policy_observation()
            expected = reference_legacy_observation(env.model)
            for field_index, field in enumerate(LEGACY_POLICY_FIELDS):
                field_errors[field].append(
                    abs(float(actual[field_index]) - float(expected[field_index]))
                )
            coverage[visibility] += 1
            if prey_direction in (-180, 180):
                angle_boundary_cases.add(prey_direction)
        field_results = {
            field: {
                "status": "PASS" if errors and max(errors) <= 1e-7 else "FAIL",
                "max_abs_error": max(errors) if errors else None,
                "case_count": len(errors),
            }
            for field, errors in field_errors.items()
        }
        checks = {
            "all_fields_exact": all(
                result["status"] == "PASS" for result in field_results.values()
            ),
            "visible_state_coverage": coverage["visible"] >= cases_per_visibility,
            "hidden_state_coverage": coverage["hidden"] >= cases_per_visibility,
            "signed_pi_boundary_coverage": angle_boundary_cases == {-180, 180},
        }
        return {
            "status": "PASS" if all(checks.values()) else "FAIL",
            "checks": checks,
            "coverage": coverage,
            "fields": field_results,
        }
    finally:
        env.close()


def _atomic_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit the legacy 10D observation contract")
    parser.add_argument("--legacy-source", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--world", default="21_05")
    parser.add_argument("--cases-per-visibility", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.cases_per_visibility <= 0:
        raise ValueError("cases-per-visibility must be positive")
    contract = json.loads(CONTRACT_PATH.read_text(encoding="utf-8"))
    source = audit_legacy_source(args.legacy_source.resolve(), contract)
    dataset = audit_dataset_distribution(args.dataset.resolve())
    dataset_hash_matches = (
        dataset["source_sha256"] == contract["audited_dataset"]["sha256"]
    )
    dataset_rows_match = dataset["row_count"] == contract["audited_dataset"]["rows"]
    dataset["checks"]["audited_dataset_sha256_matches"] = dataset_hash_matches
    dataset["checks"]["audited_dataset_row_count_matches"] = dataset_rows_match
    dataset["status"] = (
        "PASS" if all(dataset["checks"].values()) else "FAIL"
    )
    replay = audit_state_replay(
        world=args.world,
        cases_per_visibility=args.cases_per_visibility,
        seed=args.seed,
    )
    verified = all(section["status"] == "PASS" for section in (source, dataset, replay))
    report = {
        "schema_version": 1,
        "audit": "legacy_mouse_observation_contract",
        "contract": contract,
        "source_integrity": source,
        "dataset_distribution": dataset,
        "state_replay": replay,
        "verified": verified,
        "research_evidence": verified,
    }
    _atomic_json(args.output, report)
    print(json.dumps(report, indent=2, sort_keys=True))
    if not verified:
        raise SystemExit("Observation contract audit failed")


if __name__ == "__main__":
    main()
