"""Paired seeded closed-loop measurement for native XY destinations.

Raw episode rows and optional replay traces are private artifacts. Mock runs
must leave research_evidence=False; only the CLI's actual Cellworld run sets
it to True. A completed evaluation does not establish an improvement claim.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import time
from collections import Counter
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np

from .contracts import ACTION_SCHEMA, ContinuousDecision, validate_xy
from .policy import (ContinuousHierarchicalPolicy, DirectContinuousPolicy,
                     DiscreteDestinationAdapter, legacy_observation)


EVALUATION_SCHEMA = "mousemind_continuous_evaluation_v1"
CONTEXT_SCHEMA = "mousemind_continuous_context_v1"
DEFAULT_CONTRACT = Path(__file__).with_name("experiment.json")


def capture_count(env, info: dict | None = None, fallback: int = 0) -> int:
    """Read evaluator-only cumulative outcomes; never send info to a policy."""
    if info is not None and "captures" in info:
        return int(info["captures"])
    base = getattr(env, "unwrapped", env)
    model = getattr(base, "model", None)
    data = getattr(model, "prey_data", None)
    if data is not None and hasattr(data, "puff_count"):
        return int(data.puff_count)
    if model is not None and hasattr(model, "puff_count"):
        return int(model.puff_count)
    return int(fallback)


def run_episode(env, policy, *, seed: int, max_steps: int | None = None,
                control_budget_seconds: float = 0.25,
                trace_hook: Callable[[dict[str, Any]], None] | None = None) -> dict[str, Any]:
    """Evaluate one episode with public observations as the only policy input."""
    if not math.isfinite(control_budget_seconds) or control_budget_seconds <= 0:
        raise ValueError("control_budget_seconds must be finite and positive")
    base = getattr(env, "unwrapped", env)
    limit = max_steps if max_steps is not None else getattr(base, "max_step", 300)
    if isinstance(limit, bool) or int(limit) != limit or limit <= 0:
        raise ValueError("max_steps must be a positive integer")
    observation, reset_info = env.reset(seed=int(seed))
    policy.reset(int(seed))
    prefix: list[list[float]] = []
    captured = capture_count(env, reset_info)
    first_capture = 0 if captured else None
    episode_return, path_length = 0.0, 0.0
    previous_position = np.asarray(observation, dtype=np.float64)[:2].copy()
    latency: list[float] = []
    skills: Counter = Counter()
    valid, replans, destination_rejections = 0, 0, 0
    info: dict = {}
    terminated = truncated = False
    trace_builder = None
    if trace_hook is not None:
        from .context import ContinuousContextBuilder
        trace_builder = ContinuousContextBuilder()
    for step in range(int(limit)):
        before = np.asarray(observation, dtype=np.float64).copy()
        if trace_hook is not None:
            from .replay import replay_fingerprint
            fingerprint_before = replay_fingerprint(env)
            # Record the canonical public encoder before stepping. Rebuilding
            # signed angles from rounded 15D values can differ by >1e-7 even
            # when simulator state replay is exact.
            encoder = getattr(env, "legacy_policy_observation", None)
            legacy_before = (np.asarray(encoder()).copy() if callable(encoder)
                             else legacy_observation(before))
            fallback_context = trace_builder.observe(before)
        controller = getattr(policy, "controller", None)
        memory_before = controller.memory_state() if controller is not None else None
        start = time.perf_counter()
        decision = policy.act(before.copy())
        elapsed = time.perf_counter() - start
        if not isinstance(decision, ContinuousDecision):
            raise TypeError("Continuous policies must return ContinuousDecision")
        action = validate_xy(decision.action)
        latency.append(elapsed)
        valid += int(decision.valid)
        metadata = dict(decision.metadata)
        skill = metadata.get("skill")
        if skill is not None:
            skills[str(skill)] += 1
        replans += int(bool(metadata.get("replanned", False)))
        before_captures = captured
        observation, reward, terminated, truncated, info = env.step(action)
        destination_rejections += int(bool(info.get("invalid_destination", False)))
        next_values = np.asarray(observation, dtype=np.float64)
        if next_values.ndim != 1 or len(next_values) not in (10, 15) or not np.isfinite(next_values).all():
            raise ValueError("Environment emitted an invalid public observation")
        if not math.isfinite(float(reward)):
            raise ValueError("Environment emitted a non-finite reward")
        puffed = bool(next_values[10] if len(next_values) == 15 else next_values[7])
        captured = capture_count(env, info, fallback=captured + int(puffed))
        if captured < before_captures:
            raise ValueError("Capture counter decreased within an episode")
        if captured > before_captures and first_capture is None:
            first_capture = step + 1
        episode_return += float(reward)
        position = next_values[:2]
        path_length += float(np.linalg.norm(position - previous_position))
        previous_position = position.copy()
        if trace_hook is not None:
            context = getattr(policy, "last_context", None)
            if context is None:
                context = fallback_context
            trace_hook({
                "schema_version": "mousemind_continuous_trace_v1", "action_schema": ACTION_SCHEMA,
                "seed": int(seed), "step_index": step, "policy": policy.name,
                "prefix_xy_actions": [list(value) for value in prefix],
                "replay_fingerprint": fingerprint_before,
                "navigation_contract": getattr(env, "navigation_contract", "unverified"),
                "reset_contract": getattr(env, "reset_contract", "unverified"),
                "environment_observation": before.tolist(),
                "legacy_observation": legacy_before.tolist(),
                "context": context.payload(),
                "controller_memory_before": memory_before,
                "action_xy": list(action), "skill": skill, "metadata": metadata,
                "next_environment_observation": next_values.tolist(),
                "reward": float(reward), "captures_before": before_captures,
                "captures_after": captured,
                "terminated": bool(terminated), "truncated": bool(truncated),
                "invalid_destination": bool(info.get("invalid_destination", False)),
                "invalid_destination_reason": info.get("invalid_destination_reason"),
            })
            trace_builder.record_decision(action, skill if skill is not None else "unlabeled")
        prefix.append(list(action))
        if terminated or truncated:
            break
    success = int(bool(info.get("is_success", False)))
    steps = len(prefix)
    return {
        "policy": policy.name, "seed": int(seed), "success": success,
        "clean_success": int(success == 1 and captured == 0),
        "captured": int(captured > 0), "captures": captured,
        "survived": int(captured == 0), "steps": steps,
        "episode_return": episode_return, "path_length": path_length,
        "time_to_first_capture": first_capture,
        "valid_action_rate": valid / steps, "planner_calls": replans,
        "destination_rejection_rate": destination_rejections / steps,
        "skill_counts": dict(skills),
        "latency_mean_seconds": float(np.mean(latency)),
        "latency_p95_seconds": float(np.quantile(latency, 0.95)),
        "deadline_miss_rate": float(np.mean(np.asarray(latency) > control_budget_seconds)),
        "termination_reason": info.get("termination_reason", "goal" if success else "timeout"),
        "terminated": bool(terminated), "truncated": bool(truncated or (not terminated and steps == limit)),
    }


def run_policy(env_factory: Callable[[], Any], policy, *, seeds: Sequence[int],
               trace_hook=None, control_budget_seconds: float = 0.25) -> list[dict[str, Any]]:
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("Evaluation seeds must be nonempty and unique")
    rows = []
    for seed in seeds:
        env = env_factory()
        try:
            rows.append(run_episode(env, policy, seed=seed, trace_hook=trace_hook,
                                    control_budget_seconds=control_budget_seconds))
        finally:
            env.close()
    return rows


def summarize(rows: Sequence[dict[str, Any]], *, research_evidence: bool = False,
              seed_pool: str = "diagnostic", config: dict | None = None) -> dict[str, Any]:
    if not rows or len({row["seed"] for row in rows}) != len(rows):
        raise ValueError("Summary requires nonempty unique episode seeds")
    policies = {row["policy"] for row in rows}
    if len(policies) != 1:
        raise ValueError("A continuous summary represents exactly one policy")
    fields = {
        "success_rate": "success", "clean_success_rate": "clean_success",
        "capture_rate": "captured", "captures_per_episode": "captures",
        "mean_steps": "steps", "mean_return": "episode_return",
        "mean_path_length": "path_length", "valid_action_rate": "valid_action_rate",
        "mean_latency_seconds": "latency_mean_seconds", "deadline_miss_rate": "deadline_miss_rate",
        "destination_rejection_rate": "destination_rejection_rate",
    }
    return {
        "schema_version": EVALUATION_SCHEMA, "action_schema": ACTION_SCHEMA,
        "context_schema": CONTEXT_SCHEMA, "research_evidence": bool(research_evidence),
        "policy": next(iter(policies)), "seed_pool": seed_pool,
        "seeds": [int(row["seed"]) for row in rows], "episodes": len(rows),
        "metrics": {name: float(np.mean([row[field] for row in rows])) for name, field in fields.items()},
        "config": config or {},
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _private_output(path: Path) -> None:
    resolved = path.resolve()
    if any((ancestor / ".git").exists() for ancestor in (resolved, *resolved.parents)):
        raise ValueError("Evaluation rows and traces must be written outside Git")
    path.mkdir(parents=True, exist_ok=True, mode=0o700)
    if any(path.iterdir()):
        raise ValueError("Use an empty evaluation directory to avoid overwriting results")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy", choices=("continuous-bc", "rule", "minimind", "discrete-reference", "legacy-minimind", "sac-teacher"), default="continuous-bc")
    parser.add_argument("--specialist-checkpoint", type=Path)
    parser.add_argument("--legacy-mlp-checkpoint", type=Path)
    parser.add_argument("--teacher-checkpoint", type=Path)
    parser.add_argument("--base-weight", type=Path)
    parser.add_argument("--skill-lora-weight", type=Path)
    parser.add_argument("--tokenizer", type=Path, default=Path("model"))
    parser.add_argument("--planner-manifest", type=Path)
    parser.add_argument("--allow-legacy-planner", action="store_true", help="Explicit compatibility diagnostic; not a newly trained continuous planner")
    parser.add_argument("--planner-horizon", type=int, default=8)
    parser.add_argument("--memory-steps", type=int, default=8)
    parser.add_argument("--evade-distance", type=float, default=0.35)
    parser.add_argument("--instruction-split", choices=("train", "validation", "unseen_test"), default="train")
    parser.add_argument("--preference", choices=("survival_first", "balanced", "goal_first", "hold"), default="survival_first")
    parser.add_argument("--seed-pool", choices=("data_collection", "development", "final_id_test"))
    parser.add_argument("--seed-start", type=int)
    parser.add_argument("--episodes", type=int)
    parser.add_argument("--contract", type=Path, default=DEFAULT_CONTRACT)
    parser.add_argument("--max-steps", type=int, default=300)
    parser.add_argument("--time-step", type=float, default=0.25)
    parser.add_argument("--world", default="21_05")
    parser.add_argument("--predator-speed-ratio", type=float, default=0.15)
    parser.add_argument("--max-line-of-sight-distance", type=float, default=1.0)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--trace", action="store_true")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--action-catalog", type=Path, default=Path(__file__).parents[1] / "envs/mice/assets/action_catalog_21_05.json")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.allow_legacy_planner and args.policy != "minimind":
        raise ValueError("--allow-legacy-planner applies only to the new MiniMind interface diagnostic")
    if args.policy == "sac-teacher":
        if args.teacher_checkpoint is None:
            raise ValueError("SAC teacher evaluation requires --teacher-checkpoint")
    elif args.policy == "legacy-minimind":
        if args.legacy_mlp_checkpoint is None or args.base_weight is None or args.skill_lora_weight is None:
            raise ValueError("Legacy MiniMind reference requires --legacy-mlp-checkpoint, --base-weight and --skill-lora-weight")
    elif args.specialist_checkpoint is None:
        raise ValueError("This policy requires --specialist-checkpoint")
    if args.seed_pool is not None:
        if args.seed_start is not None or args.episodes is not None:
            raise ValueError("--seed-pool selects the complete pool; do not combine it with manual seed limits")
        contract = json.loads(args.contract.read_text())
        pool = contract["seed_pools"][args.seed_pool]
        seeds = list(range(int(pool["start"]), int(pool["stop_exclusive"])))
        if len(seeds) != int(pool["count"]):
            raise ValueError("Seed contract count disagrees with its range")
        seed_pool = args.seed_pool
    else:
        start = 51000 if args.seed_start is None else args.seed_start
        count = 40 if args.episodes is None else args.episodes
        if count <= 0:
            raise ValueError("--episodes must be positive")
        seeds = list(range(start, start + count))
        seed_pool = "development_subset" if start == 51000 and count == 40 else "diagnostic"
    _private_output(args.output_dir)
    planner_max_seq_len = 1024
    if args.policy == "sac-teacher":
        from .rl_teacher import SACCheckpointPolicy

        teacher = SACCheckpointPolicy(args.teacher_checkpoint, device=args.device)
        policy = DirectContinuousPolicy(teacher, name="continuous-sac-teacher")
    elif args.policy == "legacy-minimind":
        from mouse_llm.data.planner_schema import Preference
        from mouse_llm.evaluation.closed_loop import MLPCheckpointPolicy
        from mouse_llm.evaluation.evaluate_policy import load_action_catalog
        from mouse_llm.hierarchical.minimind_planner import MiniMindSkillPlanner
        from mouse_llm.hierarchical.verified_policy import ProposeVerifyPolicy

        destinations = load_action_catalog(args.action_catalog)
        specialist = MLPCheckpointPolicy(args.legacy_mlp_checkpoint, device=args.device,
                                         observation_indices=tuple(range(10)), name="legacy-mlp-specialist")
        planner = MiniMindSkillPlanner(base_weight=args.base_weight, lora_weight=args.skill_lora_weight,
                                       tokenizer_path=args.tokenizer, device=args.device,
                                       instruction_split=args.instruction_split)
        hierarchy = ProposeVerifyPolicy(specialist=specialist, destinations=destinations,
                                        planner=planner, preference=Preference(args.preference),
                                        planner_horizon=args.planner_horizon,
                                        evade_distance=args.evade_distance, name="legacy-minimind")
        policy = DiscreteDestinationAdapter(hierarchy, destinations, name="legacy-minimind-catalog-reference")
    elif args.policy == "discrete-reference":
        from mouse_llm.evaluation.closed_loop import MLPCheckpointPolicy
        from mouse_llm.evaluation.evaluate_policy import load_action_catalog

        specialist = MLPCheckpointPolicy(args.specialist_checkpoint, device=args.device,
                                         observation_indices=tuple(range(10)), name="discrete-mlp")
        policy = DiscreteDestinationAdapter(specialist, load_action_catalog(args.action_catalog))
    else:
        from .specialist import ContinuousCheckpointPolicy

        specialist = ContinuousCheckpointPolicy(args.specialist_checkpoint, device=args.device)
        if args.policy == "continuous-bc":
            policy = DirectContinuousPolicy(specialist)
        else:
            planner = None
            if args.policy == "minimind":
                if args.base_weight is None or args.skill_lora_weight is None:
                    raise ValueError("MiniMind evaluation requires --base-weight and --skill-lora-weight")
                if not args.allow_legacy_planner:
                    if args.planner_manifest is None:
                        raise ValueError("New continuous planners require --planner-manifest with action/context schemas")
                    manifest = json.loads(args.planner_manifest.read_text())
                    if (manifest.get("schema_version") != "mousemind_continuous_planner_checkpoint_v1"
                            or manifest.get("action_schema") != ACTION_SCHEMA or manifest.get("context_schema") != CONTEXT_SCHEMA):
                        raise ValueError("Planner manifest does not match the continuous action/context contract")
                    if manifest.get("skill_lora_sha256") != _sha256(args.skill_lora_weight):
                        raise ValueError("Planner manifest is not bound to the supplied skill LoRA checkpoint")
                    if manifest.get("specialist_sha256") != _sha256(args.specialist_checkpoint):
                        raise ValueError("Planner manifest was trained with a different specialist")
                    if manifest.get("base_weight_sha256") != _sha256(args.base_weight):
                        raise ValueError("Planner manifest is not bound to the supplied MiniMind base weights")
                    planner_max_seq_len = manifest.get("max_seq_len")
                    if (isinstance(planner_max_seq_len, bool) or not isinstance(planner_max_seq_len, int)
                            or planner_max_seq_len < 32):
                        raise ValueError("Planner manifest must declare a valid training max_seq_len")
                    expected_execution = {
                        "planner_horizon": args.planner_horizon,
                        "memory_steps": args.memory_steps,
                        "evade_distance": args.evade_distance,
                    }
                    if manifest.get("execution_contract") != expected_execution:
                        raise ValueError("Planner skill execution differs from its training-label contract; regenerate labels before changing it")
                    from .navigation import NAVIGATION_CONTRACT
                    from .environment import RESET_CONTRACT
                    training_environment = manifest.get("training_environment", {})
                    if (training_environment.get("navigation_contract") != NAVIGATION_CONTRACT
                            or training_environment.get("reset_contract") != RESET_CONTRACT):
                        raise ValueError("Planner training used a different navigation/reset contract")
                from mouse_llm.hierarchical.minimind_planner import MiniMindSkillPlanner

                planner = MiniMindSkillPlanner(base_weight=args.base_weight, lora_weight=args.skill_lora_weight,
                                               tokenizer_path=args.tokenizer, device=args.device,
                                               instruction_split=args.instruction_split,
                                               max_seq_len=planner_max_seq_len)
            policy = ContinuousHierarchicalPolicy(
                specialist=specialist, planner=planner, preference=args.preference,
                planner_horizon=args.planner_horizon, memory_steps=args.memory_steps,
                evade_distance=args.evade_distance,
                name="continuous-" + args.policy + ("-legacy-planner-diagnostic" if args.allow_legacy_planner else ""))
    from .environment import ContinuousBotEvadeEnv, RESET_CONTRACT
    from .navigation import NAVIGATION_CONTRACT

    config = {
        "world": args.world, "max_steps": args.max_steps, "time_step": args.time_step,
        "predator_speed_ratio": args.predator_speed_ratio,
        "max_line_of_sight_distance": args.max_line_of_sight_distance,
        "planner_horizon": args.planner_horizon, "preference": args.preference,
        "instruction_split": args.instruction_split, "memory_steps": args.memory_steps,
        "evade_distance": args.evade_distance, "complete_seed_pool": args.seed_pool is not None,
        "specialist_sha256": _sha256(args.legacy_mlp_checkpoint) if args.policy == "legacy-minimind" else (
            None if args.specialist_checkpoint is None else _sha256(args.specialist_checkpoint)),
        "teacher_sha256": None if args.teacher_checkpoint is None else _sha256(args.teacher_checkpoint),
        "skill_lora_sha256": None if args.skill_lora_weight is None else _sha256(args.skill_lora_weight),
        "base_weight_sha256": None if args.base_weight is None else _sha256(args.base_weight),
        "planner_max_seq_len": planner_max_seq_len,
        "planner_manifest_sha256": None if args.planner_manifest is None else _sha256(args.planner_manifest),
        "legacy_planner_diagnostic": args.allow_legacy_planner,
        "catalog_constrained_reference": args.policy in {"discrete-reference", "legacy-minimind"},
        "navigation_contract": NAVIGATION_CONTRACT,
        "reset_contract": RESET_CONTRACT,
    }
    if args.seed_pool is not None:
        config["contract_sha256"] = _sha256(args.contract)
    def env_factory():
        return ContinuousBotEvadeEnv(world_name=args.world, max_step=args.max_steps,
                                    time_step=args.time_step,
                                    predator_prey_forward_speed_ratio=args.predator_speed_ratio,
                                    max_line_of_sight_distance=args.max_line_of_sight_distance)
    trace = os.fdopen(os.open(args.output_dir / "trace_private.jsonl", os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600), "w") if args.trace else None
    try:
        def write_trace(row):
            row = {**row, "environment_config": {
                key: config[key] for key in (
                    "world", "max_steps", "time_step", "predator_speed_ratio",
                    "max_line_of_sight_distance", "navigation_contract", "reset_contract",
                )
            }}
            trace.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")
        rows = run_policy(env_factory, policy, seeds=seeds,
                          control_budget_seconds=args.time_step,
                          trace_hook=write_trace if trace else None)
    finally:
        if trace:
            trace.close()
    summary = summarize(rows, research_evidence=True, seed_pool=seed_pool, config=config)
    with os.fdopen(os.open(args.output_dir / "episodes.jsonl", os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600), "w") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")
    with os.fdopen(os.open(args.output_dir / "episodes.csv", os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600), "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        for row in rows:
            writer.writerow({**row, "skill_counts": json.dumps(row["skill_counts"], sort_keys=True)})
    with os.fdopen(os.open(args.output_dir / "metrics.json", os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600), "w") as handle:
        handle.write(json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n")
    for path in args.output_dir.iterdir():
        if path.is_file():
            path.chmod(0o600)
    print(json.dumps(summary, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
