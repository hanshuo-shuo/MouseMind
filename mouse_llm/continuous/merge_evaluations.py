"""Merge complete, compatible continuous evaluation shards without rerunning them.

All policy and execution configuration must agree. Only full seed-pool coverage
changes complete_seed_pool/contract_sha256. Mock-only merges remain non-research
evidence, and research and non-research shards can never be mixed.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from pathlib import Path
from typing import Any, Sequence

from .contracts import ACTION_SCHEMA
from .evaluation import CONTEXT_SCHEMA, DEFAULT_CONTRACT, EVALUATION_SCHEMA, summarize
from .promotion import SUMMARY_FIELDS, _contract, _integer, _json, _number, _reject_mock, _sha256


MERGE_SCHEMA = "mousemind_continuous_evaluation_merge_v1"
SEED_POOLS = ("development", "final_id_test")
COVERAGE_FIELDS = {"complete_seed_pool", "contract_sha256"}
CHECKPOINT_FIELDS = ("specialist_sha256", "teacher_sha256", "skill_lora_sha256",
                     "base_weight_sha256", "planner_manifest_sha256")
ORIGIN_FIELDS = ("synthetic", "mock", "is_mock", "mock_run", "injected_test_dependencies")


def _canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _target_contract(path: Path, seed_pool: str):
    if seed_pool not in SEED_POOLS:
        raise ValueError("Only development or final_id_test ID pools can be merged")
    contract, _, contract_sha256 = _contract(path)
    if contract.get("context_schema") != CONTEXT_SCHEMA:
        raise ValueError("Wrong continuous context contract")
    pool = contract["seed_pools"][seed_pool]
    start = _integer(pool.get("start"), "seed-pool start")
    stop = _integer(pool.get("stop_exclusive"), "seed-pool stop")
    count = _integer(pool.get("count"), "seed-pool count", minimum=1)
    if count != (80 if seed_pool == "development" else 200) or stop - start != count:
        raise ValueError("The selected ID contract must define the complete 80/200-seed pool")
    expected_environment = {
        "world": contract["world"], "max_steps": contract["max_steps"],
        "time_step": contract["time_step"],
        "predator_speed_ratio": contract["predator_prey_forward_speed_ratio"],
        "max_line_of_sight_distance": contract.get("max_line_of_sight_distance", 1.0),
        "navigation_contract": contract["navigation_contract"],
        "reset_contract": contract["reset_contract"],
        "destination_mode": contract["destination_mode"],
        "projection_epsilon": contract["projection_epsilon"],
    }
    return tuple(range(start, stop)), expected_environment, contract_sha256


def _validate_config(config: Any, *, expected_environment: dict[str, Any],
                     contract_sha256: str, research_evidence: bool) -> dict[str, Any]:
    if not isinstance(config, dict):
        raise ValueError("Shard configuration must be a JSON object")
    if not isinstance(config.get("complete_seed_pool"), bool):
        raise ValueError("Shard must explicitly declare complete_seed_pool")
    if not isinstance(config.get("reuse_environment"), bool):
        raise ValueError("Shard must explicitly declare boolean reuse_environment")
    if "contract_sha256" in config and config["contract_sha256"] != contract_sha256:
        raise ValueError("Shard experiment contract hash does not match the selected contract")
    for name, expected in expected_environment.items():
        if name not in config or isinstance(config[name], bool) or config[name] != expected:
            raise ValueError(f"Shard ID environment/navigation/reset contract mismatch: {name}")
    present_hash = False
    for name in CHECKPOINT_FIELDS:
        value = config.get(name)
        if value is None:
            continue
        if not isinstance(value, str) or len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
            raise ValueError(f"Invalid checkpoint hash: {name}")
        present_hash = True
    if research_evidence and not present_hash:
        raise ValueError("Research shards must identify their evaluated checkpoint by hash")
    # Every other field is part of identity, including optional execution flags
    # such as reuse_environment and all checkpoint/tokenizer/planner settings.
    return {name: value for name, value in config.items() if name not in COVERAGE_FIELDS}


def _validate_episode(row: dict[str, Any], *, policy: str, max_steps: int,
                      research_evidence: bool) -> int:
    if research_evidence:
        _reject_mock(row)
    seed = _integer(row.get("seed"), "episode seed")
    if row.get("policy") != policy:
        raise ValueError("Episode policy disagrees with the shard summary")
    if "action_schema" in row and row["action_schema"] != ACTION_SCHEMA:
        raise ValueError("Episode action schema mismatch")
    for field in ("success", "clean_success", "captured", "survived"):
        if _integer(row.get(field), field) not in (0, 1):
            raise ValueError(f"{field} must be 0 or 1")
    captures = _integer(row.get("captures"), "captures")
    if not 1 <= _integer(row.get("steps"), "steps", minimum=1) <= max_steps:
        raise ValueError("Episode exceeds the configured step budget")
    if (row["clean_success"] != int(row["success"] == 1 and captures == 0)
            or row["captured"] != int(captures > 0)
            or row["survived"] != int(captures == 0)):
        raise ValueError("Episode clean-success/capture accounting is inconsistent")
    for field in SUMMARY_FIELDS.values():
        value = _number(row.get(field), f"episode {field}")
        if field.endswith("_rate") and not 0 <= value <= 1:
            raise ValueError("Episode rates must be in [0, 1]")
        if field in ("path_length", "latency_mean_seconds", "mean_projection_distance") and value < 0:
            raise ValueError("Path, latency, and projection-distance metrics must be nonnegative")
    return seed


def _read_shard(directory: Path, *, target_seeds: tuple[int, ...], seed_pool: str,
                expected_environment: dict[str, Any], contract_sha256: str):
    metrics_path, episodes_path = directory / "metrics.json", directory / "episodes.jsonl"
    summary = _json(metrics_path.read_text(encoding="utf-8"))
    if (summary.get("schema_version") != EVALUATION_SCHEMA
            or summary.get("action_schema") != ACTION_SCHEMA
            or summary.get("context_schema") != CONTEXT_SCHEMA):
        raise ValueError("Shard evaluation/action/context schema mismatch")
    research = summary.get("research_evidence")
    if not isinstance(research, bool):
        raise ValueError("research_evidence must be an explicit boolean")
    if research:
        _reject_mock(summary)
    source_pool = summary.get("seed_pool")
    allowed_pools = {"diagnostic", seed_pool}
    if seed_pool == "development":
        allowed_pools.add("development_subset")
    if source_pool not in allowed_pools:
        raise ValueError("Shard source pool does not belong to the requested evaluation pool")
    config = summary.get("config")
    identity_config = _validate_config(config, expected_environment=expected_environment,
                                       contract_sha256=contract_sha256, research_evidence=research)
    if config["complete_seed_pool"] != (source_pool == seed_pool):
        raise ValueError("Shard seed-pool label and completeness flag disagree")
    policy = summary.get("policy")
    if not isinstance(policy, str) or not policy:
        raise ValueError("Shard policy must be a nonempty string")
    rows, local_seeds = [], set()
    row_fields = None
    with episodes_path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = _json(line)
            seed = _validate_episode(row, policy=policy, max_steps=expected_environment["max_steps"], research_evidence=research)
            if seed in local_seeds:
                raise ValueError("Duplicate episode seed within shard")
            if seed not in target_seeds:
                raise ValueError("Unexpected seed outside the selected ID pool")
            local_seeds.add(seed)
            if row_fields is None:
                row_fields = set(row)
            elif set(row) != row_fields:
                raise ValueError("Episode row field schemas differ within shard")
            rows.append(row)
    if not rows or _integer(summary.get("episodes"), "summary episode count", minimum=1) != len(rows):
        raise ValueError("Shard summary episode count disagrees with its rows")
    listed_seeds = summary.get("seeds")
    if not isinstance(listed_seeds, list):
        raise ValueError("Shard summary must list seeds explicitly")
    for seed in listed_seeds:
        _integer(seed, "summary seed")
    if len(listed_seeds) != len(rows) or set(listed_seeds) != local_seeds:
        raise ValueError("Shard summary seeds disagree with episode rows or contain duplicates")
    if config["complete_seed_pool"] and local_seeds != set(target_seeds):
        raise ValueError("Shard falsely claims complete seed-pool coverage")
    recomputed = summarize(rows, research_evidence=research, seed_pool=source_pool, config=config)
    metrics = summary.get("metrics")
    if not isinstance(metrics, dict) or set(metrics) != set(recomputed["metrics"]):
        raise ValueError("Shard summary metric schema differs from current runtime")
    for name, expected in recomputed["metrics"].items():
        if not math.isclose(_number(metrics[name], name), expected, rel_tol=1e-12, abs_tol=1e-12):
            raise ValueError(f"Shard summary metric disagrees with episode rows: {name}")
    return {
        "summary": summary, "rows": rows, "identity_config": identity_config,
        "origin_flags": {name: summary.get(name, False) for name in ORIGIN_FIELDS},
        "provenance": {"directory": str(directory), "metrics_sha256": _sha256(metrics_path),
                       "episodes_sha256": _sha256(episodes_path), "episodes": len(rows),
                       "seed_min": min(local_seeds), "seed_max": max(local_seeds),
                       "source_seed_pool": source_pool, "source_complete_seed_pool": config["complete_seed_pool"],
                       "research_evidence": research},
    }


def _private_output(directory: Path) -> Path:
    directory = Path(directory).expanduser().resolve()
    if any((parent / ".git").exists() for parent in (directory, *directory.parents)):
        raise ValueError("Merged private evaluation rows must remain outside Git")
    if directory.exists() and (not directory.is_dir() or any(directory.iterdir())):
        raise ValueError("Merge output directory must be new or empty")
    return directory


def _private_file(path: Path):
    return os.fdopen(os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600), "w", encoding="utf-8", newline="")


def merge_evaluations(shard_dirs: Sequence[Path], *, seed_pool: str,
                      output_dir: Path, contract_path: Path = DEFAULT_CONTRACT) -> dict[str, Any]:
    """Validate every input before writing a complete runtime-compatible report."""
    output = _private_output(output_dir)
    directories = [Path(directory).expanduser().resolve() for directory in shard_dirs]
    if not directories or len(set(directories)) != len(directories):
        raise ValueError("Provide nonempty unique shard directories")
    seeds, expected_environment, contract_sha256 = _target_contract(Path(contract_path).resolve(), seed_pool)
    shards = [_read_shard(directory, target_seeds=seeds, seed_pool=seed_pool,
                         expected_environment=expected_environment, contract_sha256=contract_sha256)
              for directory in directories]
    first = shards[0]
    signature = (_canonical(first["identity_config"]), first["summary"]["policy"],
                 first["summary"]["research_evidence"], _canonical(first["origin_flags"]))
    episodes = {}
    row_fields = set(first["rows"][0])
    for shard in shards:
        other = (_canonical(shard["identity_config"]), shard["summary"]["policy"],
                 shard["summary"]["research_evidence"], _canonical(shard["origin_flags"]))
        if other != signature:
            raise ValueError("Shard policy, full configuration/checkpoint identity, or research/mock evidence status differs")
        for row in shard["rows"]:
            if set(row) != row_fields:
                raise ValueError("Episode row field schemas differ across shards")
            if row["seed"] in episodes:
                raise ValueError("Duplicate episode seed across shards")
            episodes[row["seed"]] = row
    if set(episodes) != set(seeds):
        raise ValueError("Shard union is missing seeds from the complete selected pool")
    rows = [episodes[seed] for seed in seeds]
    config = {**first["identity_config"], "complete_seed_pool": True, "contract_sha256": contract_sha256}
    summary = summarize(rows, research_evidence=first["summary"]["research_evidence"], seed_pool=seed_pool, config=config)
    summary.update({name: value for name, value in first["origin_flags"].items() if value})
    provenance = {
        "schema_version": MERGE_SCHEMA, "action_schema": ACTION_SCHEMA,
        "contract_sha256": contract_sha256, "seed_pool": seed_pool,
        "complete_seed_coverage": True, "episodes": len(rows), "shard_count": len(shards),
        "sources": [shard["provenance"] for shard in shards],
        "configuration_preservation": "All configuration values retained; only complete_seed_pool and contract_sha256 reflect merged coverage",
        "summary_validation": "Every source metric recomputed from its episode rows before merge",
        "research_evidence": summary["research_evidence"],
        "scope": "Aggregation only; no new rollout, model selection, or promotion claim",
    }
    summary["merge_provenance"] = provenance
    output = _private_output(output)
    output.mkdir(parents=True, exist_ok=True, mode=0o700)
    output.chmod(0o700)
    with _private_file(output / "episodes.jsonl") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")
    with _private_file(output / "episodes.csv") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        for row in rows:
            writer.writerow({key: json.dumps(value, sort_keys=True, allow_nan=False) if isinstance(value, (dict, list)) else value for key, value in row.items()})
    for filename, payload in (("metrics.json", summary), ("merge_provenance.json", provenance)):
        with _private_file(output / filename) as handle:
            handle.write(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n")
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shard-dirs", type=Path, nargs="+", required=True)
    parser.add_argument("--seed-pool", choices=SEED_POOLS, required=True)
    parser.add_argument("--contract", type=Path, default=DEFAULT_CONTRACT)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    result = merge_evaluations(args.shard_dirs, seed_pool=args.seed_pool,
                               contract_path=args.contract, output_dir=args.output_dir)
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
