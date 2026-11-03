"""Fail-closed paired evaluation gate; eligibility never publishes a model.

The final pool is a one-time acceptance set, not a hyperparameter selection
set. Bootstrap intervals condition on the two trained checkpoints and resample
paired environment seeds; they do not cover training-run variability.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from dataclasses import dataclass
from numbers import Real
from pathlib import Path
from typing import Any

from .contracts import ACTION_SCHEMA

PROMOTION_SCHEMA = "mousemind_continuous_promotion_v1"
EVALUATION_SCHEMA = "mousemind_continuous_evaluation_v1"
EXPERIMENT_SCHEMA = "mousemind_continuous_experiment_v1"
DEFAULT_CONTRACT = Path(__file__).with_name("experiment.json")
DEFAULT_BOOTSTRAP_REPLICATES = 10000
DEFAULT_BOOTSTRAP_SEED = 42
ENVIRONMENT_FIELDS = (
    "world", "max_steps", "time_step", "predator_speed_ratio",
    "max_line_of_sight_distance", "navigation_contract", "reset_contract",
    "destination_mode", "projection_epsilon", "reuse_environment",
)
SUMMARY_FIELDS = {
    "success_rate": "success", "clean_success_rate": "clean_success",
    "capture_rate": "captured", "captures_per_episode": "captures",
    "mean_steps": "steps", "mean_return": "episode_return",
    "mean_path_length": "path_length", "valid_action_rate": "valid_action_rate",
    "mean_latency_seconds": "latency_mean_seconds", "deadline_miss_rate": "deadline_miss_rate",
    "destination_rejection_rate": "destination_rejection_rate",
    "requested_invalid_rate": "requested_invalid_rate",
    "projection_rate": "projection_rate",
    "mean_projection_distance": "mean_projection_distance",
}
DELTA_FIELDS = ("clean_success_rate", "success_rate", "captures_per_episode")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("Duplicate JSON object keys are not allowed")
        result[key] = value
    return result


def _reject_constant(value):
    raise ValueError("Non-finite JSON numeric constants are not allowed")


def _finite_tree(value: Any) -> None:
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError("Evidence contains a non-finite numeric value")
    if isinstance(value, dict):
        for item in value.values():
            _finite_tree(item)
    elif isinstance(value, list):
        for item in value:
            _finite_tree(item)


def _json(text: str) -> dict[str, Any]:
    value = json.loads(text, parse_constant=_reject_constant, object_pairs_hook=_object)
    if not isinstance(value, dict):
        raise ValueError("Evidence JSON must be an object")
    _finite_tree(value)
    return value


def _integer(value: Any, name: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


def _number(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value):
        raise ValueError(f"{name} must be finite and numeric")
    return float(value)


def _contract(path: Path) -> tuple[dict[str, Any], tuple[int, ...], str]:
    contract = _json(path.read_text(encoding="utf-8"))
    if contract.get("schema_version") != EXPERIMENT_SCHEMA or contract.get("action_schema") != ACTION_SCHEMA:
        raise ValueError("Wrong continuous experiment/action contract")
    destination_mode = contract.get("destination_mode")
    if destination_mode not in ("project", "reject"):
        raise ValueError("Experiment must explicitly declare destination_mode project or reject")
    if _number(contract.get("projection_epsilon"), "projection epsilon") <= 0:
        raise ValueError("Experiment projection_epsilon must be positive")
    if contract.get("navigation_contract") != f"graph_route_exact_xy_terminal_{destination_mode}_v2":
        raise ValueError("Experiment navigation contract does not match its destination mode")
    if contract.get("reset_contract") != "fresh_navigation_episode_reset_v1":
        raise ValueError("Experiment must declare the verified reset contract")
    pools = contract.get("seed_pools", {})
    final = pools.get("final_id_test", {})
    start = _integer(final.get("start"), "final seed start")
    stop = _integer(final.get("stop_exclusive"), "final seed stop")
    count = _integer(final.get("count"), "final seed count", minimum=1)
    if count != 200 or stop - start != count:
        raise ValueError("Promotion requires the registered complete 200-seed final pool")
    seeds = tuple(range(start, stop))
    # Historical v1 final seeds must not become new confirmatory evidence.
    if set(seeds).intersection(range(30000, 30100)):
        raise ValueError("Historical final seeds are not fresh continuous final evidence")
    for name in ("data_collection", "development"):
        pool = pools.get(name, {})
        low = _integer(pool.get("start"), f"{name} seed start")
        high = _integer(pool.get("stop_exclusive"), f"{name} seed stop")
        if set(seeds).intersection(range(low, high)):
            raise ValueError("Final seed pool overlaps collection or development")
    selection = contract.get("selection", {})
    if selection.get("final_pool_is_never_training_or_selection_data") is not True or selection.get("publish_automatically") is not False:
        raise ValueError("Experiment must prohibit final-pool selection and automatic publication")
    margin = _number(selection.get("task_success_noninferiority_margin"), "task margin")
    if margin != 0.02:
        raise ValueError("This gate requires the predeclared 0.02 task noninferiority margin")
    return contract, seeds, _sha256(path)


def _reject_mock(metadata: dict[str, Any]) -> None:
    for obj in (metadata, metadata.get("config", {})):
        if not isinstance(obj, dict):
            raise ValueError("Evidence config must be an object")
        for marker in ("synthetic", "mock", "is_mock", "mock_run", "injected_test_dependencies"):
            if obj.get(marker) not in (None, False):
                raise ValueError("Synthetic or mock evidence cannot enter promotion")
        for key in ("environment_type", "backend", "evidence_type"):
            if str(obj.get(key, "")).lower() in {"mock", "synthetic"}:
                raise ValueError("Synthetic or mock evidence cannot enter promotion")
    policy = str(metadata.get("policy", "")).lower().replace("_", "-")
    if any(token in {"mock", "synthetic"} for token in policy.split("-")):
        raise ValueError("A mock/synthetic policy report cannot enter promotion")


@dataclass(frozen=True)
class EvaluationEvidence:
    directory: Path
    summary: dict[str, Any]
    episodes: dict[int, dict[str, Any]]
    environment: dict[str, Any]
    provenance: dict[str, Any]


def load_evidence(directory: Path, *, contract: dict[str, Any],
                  seeds: tuple[int, ...], contract_sha256: str) -> EvaluationEvidence:
    directory = Path(directory).expanduser().resolve()
    metrics_path, episodes_path = directory / "metrics.json", directory / "episodes.jsonl"
    summary = _json(metrics_path.read_text(encoding="utf-8"))
    _reject_mock(summary)
    if summary.get("research_evidence") is not True:
        raise ValueError("Only research_evidence=true runtime evaluations are accepted")
    if summary.get("schema_version") != EVALUATION_SCHEMA or summary.get("action_schema") != ACTION_SCHEMA:
        raise ValueError("Evaluation schema/action contract mismatch")
    if summary.get("context_schema") != contract.get("context_schema"):
        raise ValueError("Evaluation context schema mismatch")
    if summary.get("seed_pool") != "final_id_test":
        raise ValueError("Promotion requires final_id_test evidence")
    if _integer(summary.get("episodes"), "summary episode count") != len(seeds):
        raise ValueError("Summary must contain the complete 200 final episodes")
    summary_seeds = summary.get("seeds")
    if not isinstance(summary_seeds, list):
        raise ValueError("Summary must contain its explicit seed list")
    for seed in summary_seeds:
        _integer(seed, "summary seed")
    if len(summary_seeds) != len(seeds) or set(summary_seeds) != set(seeds):
        raise ValueError("Summary seeds must exactly cover the registered final pool without duplicates")
    policy = summary.get("policy")
    if not isinstance(policy, str) or not policy:
        raise ValueError("Summary policy must be a nonempty string")
    config = summary.get("config", {})
    if config.get("complete_seed_pool") is not True or config.get("contract_sha256") != contract_sha256:
        raise ValueError("Evaluation must use the complete matching experiment contract")
    if any(name not in config for name in ENVIRONMENT_FIELDS):
        raise ValueError("Evaluation is missing required environment/navigation/reset configuration")
    environment = {name: config[name] for name in ENVIRONMENT_FIELDS}
    expected = {"world": contract["world"], "max_steps": contract["max_steps"],
                "time_step": contract["time_step"],
                "predator_speed_ratio": contract["predator_prey_forward_speed_ratio"],
                "max_line_of_sight_distance": contract.get("max_line_of_sight_distance", 1.0),
                "destination_mode": contract["destination_mode"],
                "projection_epsilon": contract["projection_epsilon"],
                "navigation_contract": contract["navigation_contract"],
                "reset_contract": contract["reset_contract"]}
    for name, value in expected.items():
        if environment[name] != value or isinstance(environment[name], bool):
            raise ValueError(f"Final-ID environment differs from the experiment contract: {name}")
    if not isinstance(environment["reuse_environment"], bool):
        raise ValueError("reuse_environment must be explicitly boolean")
    for name in ("navigation_contract", "reset_contract"):
        value = environment[name]
        if not isinstance(value, str) or not value or any(marker in value.lower() for marker in ("unverified", "mock", "synthetic")):
            raise ValueError(f"Invalid or unverified {name}")
    episodes = {}
    with episodes_path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = _json(line)
            _reject_mock(row)
            seed = _integer(row.get("seed"), "episode seed")
            if seed in episodes:
                raise ValueError("Duplicate episode seed")
            if row.get("policy") != policy:
                raise ValueError("Episode policy disagrees with its summary")
            for field in ("success", "clean_success", "captured", "survived"):
                if _integer(row.get(field), field) not in (0, 1):
                    raise ValueError(f"{field} must be 0 or 1")
            captures = _integer(row.get("captures"), "captures")
            steps = _integer(row.get("steps"), "steps", minimum=1)
            if steps > environment["max_steps"]:
                raise ValueError("Episode exceeds the configured step budget")
            if row["clean_success"] != int(row["success"] == 1 and captures == 0) or row["captured"] != int(captures > 0) or row["survived"] != int(captures == 0):
                raise ValueError("Episode clean-success/capture accounting is inconsistent")
            for field in SUMMARY_FIELDS.values():
                number = _number(row.get(field), f"episode {field}")
                if field.endswith("_rate") and not 0.0 <= number <= 1.0:
                    raise ValueError("Episode rates must be in [0, 1]")
                if field in ("path_length", "latency_mean_seconds", "mean_projection_distance") and number < 0:
                    raise ValueError("Episode path/latency/projection-distance metrics must be nonnegative")
            episodes[seed] = row
    if len(episodes) != len(seeds) or set(episodes) != set(seeds):
        raise ValueError("Episode rows must exactly cover all 200 registered final seeds")
    metrics = summary.get("metrics")
    if not isinstance(metrics, dict) or set(metrics) != set(SUMMARY_FIELDS):
        raise ValueError("Summary metrics do not match the current runtime metric schema")
    for name, field in SUMMARY_FIELDS.items():
        reported = _number(metrics[name], f"summary {name}")
        computed = math.fsum(float(episodes[seed][field]) for seed in seeds) / len(seeds)
        if not math.isclose(reported, computed, rel_tol=1e-12, abs_tol=1e-12):
            raise ValueError(f"Summary metric disagrees with episode rows: {name}")
    return EvaluationEvidence(directory, summary, episodes, environment, {
        "policy": policy, "metrics_sha256": _sha256(metrics_path),
        "episodes_sha256": _sha256(episodes_path), "episodes": len(episodes),
        "checkpoint_hashes": {name: config.get(name) for name in ("specialist_sha256", "teacher_sha256", "skill_lora_sha256")},
    })


def evaluate_promotion(reference_dir: Path, candidate_dir: Path, *,
                       contract_path: Path = DEFAULT_CONTRACT,
                       bootstrap_replicates: int = DEFAULT_BOOTSTRAP_REPLICATES,
                       bootstrap_seed: int = DEFAULT_BOOTSTRAP_SEED) -> dict[str, Any]:
    """Return aggregate eligibility after validating paired, complete evidence."""
    import numpy as np

    _integer(bootstrap_replicates, "bootstrap_replicates", minimum=1)
    _integer(bootstrap_seed, "bootstrap_seed")
    contract_path = Path(contract_path).resolve()
    contract, seeds, contract_sha256 = _contract(contract_path)
    reference = load_evidence(reference_dir, contract=contract, seeds=seeds, contract_sha256=contract_sha256)
    candidate = load_evidence(candidate_dir, contract=contract, seeds=seeds, contract_sha256=contract_sha256)
    if reference.environment != candidate.environment:
        raise ValueError("Reference and candidate environment/navigation/reset configurations differ")
    ref = np.asarray([[reference.episodes[seed][SUMMARY_FIELDS[name]] for name in DELTA_FIELDS] for seed in seeds], dtype=np.float64)
    cand = np.asarray([[candidate.episodes[seed][SUMMARY_FIELDS[name]] for name in DELTA_FIELDS] for seed in seeds], dtype=np.float64)
    differences = cand - ref
    rng = np.random.default_rng(bootstrap_seed)
    draws = np.empty((bootstrap_replicates, len(DELTA_FIELDS)), dtype=np.float64)
    for offset in range(0, bootstrap_replicates, 512):
        count = min(512, bootstrap_replicates - offset)
        indices = rng.integers(0, len(seeds), size=(count, len(seeds)))
        draws[offset:offset + count] = differences[indices].mean(axis=1)
    bounds = np.quantile(draws, (0.025, 0.975), axis=0)
    deltas = {
        name: {"reference_mean": float(ref[:, index].mean()),
               "candidate_mean": float(cand[:, index].mean()),
               "delta": float(differences[:, index].mean()),
               "ci_low": float(bounds[0, index]), "ci_high": float(bounds[1, index]),
               "confidence_level": 0.95,
               "units": "captures_per_episode" if name == "captures_per_episode" else "rate_difference"}
        for index, name in enumerate(DELTA_FIELDS)
    }
    criteria = {
        "clean_success_superiority": deltas["clean_success_rate"]["ci_low"] > 0.0,
        "task_success_noninferiority": deltas["success_rate"]["ci_low"] >= -0.02,
        "captures_nonincrease": deltas["captures_per_episode"]["ci_high"] <= 0.0,
    }
    status = "eligible_for_manual_review" if all(criteria.values()) else "not_promoted"
    return {
        "schema_version": PROMOTION_SCHEMA, "action_schema": ACTION_SCHEMA,
        "status": status, "candidate_minus_reference": deltas, "criteria": criteria,
        "failed_criteria": [name for name, passed in criteria.items() if not passed],
        "thresholds": {"clean_success_ci_low_gt": 0.0, "task_success_ci_low_ge": -0.02, "captures_ci_high_le": 0.0},
        "seed_pool": "final_id_test", "paired_episodes": len(seeds),
        "seed_sha256": hashlib.sha256(json.dumps(seeds, separators=(",", ":")).encode()).hexdigest(),
        "contract_sha256": contract_sha256, "environment": reference.environment,
        "reference": reference.provenance, "candidate": candidate.provenance,
        "bootstrap": {"method": "paired_percentile", "replicates": bootstrap_replicates,
                      "seed": bootstrap_seed, "resampling_unit": "paired_environment_episode_seed",
                      "confidence_level": 0.95, "shared_resamples_across_metrics": True},
        "inference_limits": {
            "conditional_on_evaluated_checkpoints": True,
            "training_randomness_covered": False,
            "multiple_comparisons_corrected": False,
            "checkpoint_selection_history_verified": False,
            "notice": "Select checkpoints on development before opening the final pool. Do not use these results for final-pool tuning or candidate selection. Repeated final-pool comparisons remain exploratory.",
        },
        "automatic_commit": False, "automatic_push": False, "automatic_publication": False,
        "manual_review_required": True,
    }


def write_report(path: Path, report: dict[str, Any], *, input_dirs: tuple[Path, Path]) -> None:
    path = Path(path).expanduser().resolve()
    protected = {Path(directory).resolve() / filename for directory in input_dirs for filename in ("metrics.json", "episodes.jsonl", "episodes.csv", "trace_private.jsonl")}
    if path in protected:
        raise ValueError("Promotion output must not overwrite evaluation evidence")
    if any((parent / ".git").exists() for parent in (path.parent, *path.parent.parents)):
        raise ValueError("Write private promotion reports outside Git")
    content = json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    with os.fdopen(os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600), "w", encoding="utf-8") as handle:
        handle.write(content)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-dir", type=Path, required=True)
    parser.add_argument("--candidate-dir", type=Path, required=True)
    parser.add_argument("--contract", type=Path, default=DEFAULT_CONTRACT)
    parser.add_argument("--bootstrap-replicates", type=int, default=DEFAULT_BOOTSTRAP_REPLICATES)
    parser.add_argument("--bootstrap-seed", type=int, default=DEFAULT_BOOTSTRAP_SEED)
    parser.add_argument("--output", type=Path)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    report = evaluate_promotion(args.reference_dir, args.candidate_dir, contract_path=args.contract,
                                bootstrap_replicates=args.bootstrap_replicates, bootstrap_seed=args.bootstrap_seed)
    if args.output is not None:
        write_report(args.output, report, input_dirs=(args.reference_dir, args.candidate_dir))
    print(json.dumps(report, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
