"""Generate SFT/DPO supervision by replaying continuous skill alternatives.

Labels measure a candidate skill followed by the same public rule continuation.
They are continuation-dependent outcomes, not optimal-action ground truth.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from contextlib import nullcontext
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Callable

import numpy as np

from mouse_llm.data.planner_schema import Preference, instruction_for, planner_messages, skill_target
from mouse_llm.hierarchical.policy import Skill
from .context import CONTEXT_SCHEMA
from .contracts import ACTION_SCHEMA, validate_xy
from .replay import replay_fingerprint
from .pool import environment_pool

DATASET_SCHEMA = "mousemind_continuous_preferences_v1"
EXPERIMENT = Path(__file__).with_name("experiment.json")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def seed_split(seed: int) -> str:
    bucket = int.from_bytes(hashlib.sha256(f"continuous-v1:{seed}".encode()).digest()[:4], "big") % 100
    return "train" if bucket < 80 else "validation" if bucket < 90 else "test"


def private_directory(path: Path) -> None:
    path = path.resolve()
    if any((parent / ".git").exists() for parent in (path, *path.parents)):
        raise ValueError("Private traces, datasets and checkpoints must be outside Git")
    path.mkdir(parents=True, exist_ok=True, mode=0o700)


def private_json(path: Path, payload: Any) -> None:
    with os.fdopen(os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600), "w") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    path.chmod(0o600)


def capture_count(env: Any, info: dict) -> int:
    if "captures" in info:
        return int(info["captures"])
    base = getattr(env, "unwrapped", env)
    return int(getattr(getattr(getattr(base, "model", None), "prey_data", None), "puff_count", 0))


def outcome_utility(outcome: dict, preference: Preference, skill: Skill) -> float:
    """A single scalar ordering shared by chosen labels and preference pairs."""
    if preference == Preference.HOLD:
        return float(skill == Skill.HOLD_POSITION)
    captured, captures = float(outcome["captures"] > 0), float(outcome["captures"])
    goal, progress = float(outcome["success"]), float(outcome["goal_progress"])
    weights = {
        Preference.SURVIVAL_FIRST: (100.0, 10.0, 10.0, 1.0),
        Preference.BALANCED: (12.0, 4.0, 8.0, 4.0),
        Preference.GOAL_FIRST: (3.0, 1.0, 20.0, 10.0),
    }[preference]
    value = -weights[0] * captured - weights[1] * captures + weights[2] * goal + weights[3] * progress
    if not math.isfinite(value):
        raise ValueError("Non-finite counterfactual utility")
    return value


def evaluate_branch(env_factory: Callable, anchor: dict, specialist: Any, skill: Skill,
                    *, horizon: int, candidate_steps: int, replay_tolerance: float = 1e-7,
                    memory_steps: int = 8, evade_distance: float = 0.35,
                    environment: Any | None = None) -> dict:
    """Replay and verify one branch, optionally borrowing an owned environment.

    A supplied environment remains the caller's responsibility to close. It
    is reset at the start of every branch, and the same replay checks apply.
    """
    from .policy import ContinuousSkillController
    if not 1 <= candidate_steps <= horizon:
        raise ValueError("candidate_steps must be within the positive rollout horizon")
    if not anchor.get("replay_fingerprint"):
        raise ValueError("Anchor lacks a simulator/control fingerprint; recollect the trace")
    controller = ContinuousSkillController(specialist, memory_steps=memory_steps, evade_distance=evade_distance)
    owns_environment = environment is None
    env = env_factory() if owns_environment else environment
    try:
        controller.reset(int(anchor["seed"]))
        observation, info = env.reset(seed=int(anchor["seed"]))
        for action in anchor["prefix_xy_actions"]:
            controller.observe(env.legacy_policy_observation())
            observation, _, terminated, truncated, info = env.step(validate_xy(action))
            if terminated or truncated:
                raise ValueError("Anchor prefix reached termination")
        expected = np.asarray(anchor["environment_observation"])
        legacy = env.legacy_policy_observation()
        if (np.asarray(observation).shape != expected.shape
                or not np.allclose(observation, expected, atol=replay_tolerance, rtol=0)
                or not np.allclose(legacy, anchor["legacy_observation"], atol=replay_tolerance, rtol=0)
                or replay_fingerprint(env) != anchor["replay_fingerprint"]):
            raise ValueError("Anchor failed observation and simulator/control replay verification")
        start_captures = capture_count(env, info)
        start_goal = float(legacy[6])
        captured = False
        success = False
        steps = 0
        for step in range(horizon):
            legacy = env.legacy_policy_observation()
            # All candidates use identical, public-observation continuation.
            continuation = Skill.EVADE_PREDATOR if legacy[3] != 0 or legacy[4] != 0 else Skill.GO_TO_GOAL
            executed = skill if step < candidate_steps else continuation
            decision = controller.act(executed, legacy)
            observation, _, terminated, truncated, info = env.step(decision.action)
            steps += 1
            captured |= bool(env.legacy_policy_observation()[7])
            if terminated or truncated:
                success = bool(info.get("is_success", False))
                break
        captures = max(capture_count(env, info) - start_captures, int(captured))
        return {
            "skill": skill.value, "captures": captures, "success": int(success),
            "future_capture_free_completion": int(success and captures == 0),
            "goal_progress": start_goal - float(env.legacy_policy_observation()[6]),
            "steps": steps, "final_observation": np.asarray(observation).tolist(),
            "final_replay_fingerprint": replay_fingerprint(env),
        }
    finally:
        if owns_environment:
            env.close()


def evaluate_repeated_branch(env_factory: Callable, anchor: dict, specialist: Any, skill: Skill,
                             *, horizon: int, candidate_steps: int,
                             environments: tuple[Any, Any] | None = None) -> dict:
    """Require identical outcomes from two independently reset instances."""
    if environments is not None:
        if len(environments) != 2 or environments[0] is environments[1]:
            raise ValueError("Repeated branch verification requires two independent environments")
        first_env, second_env = environments
    else:
        first_env = second_env = None
    first = evaluate_branch(env_factory, anchor, specialist, skill, horizon=horizon,
                            candidate_steps=candidate_steps, environment=first_env)
    second = evaluate_branch(env_factory, anchor, specialist, skill, horizon=horizon,
                             candidate_steps=candidate_steps, environment=second_env)
    if first != second:
        raise ValueError("Repeated branch outcomes differ; refusing a label")
    return first


def build_rows(record: dict, *, minimum_margin: float) -> list[dict]:
    seed = int(record["seed"])
    split = seed_split(seed)
    context = record["context"]
    if not isinstance(context, dict) or context.get("schema_version") != CONTEXT_SCHEMA or context.get("action_schema") != ACTION_SCHEMA:
        raise ValueError("Cannot reuse discrete planner contexts for continuous training")
    rows = []
    for preference in Preference:
        utilities = {skill.value: outcome_utility(record["branches"][skill.value], preference, skill) for skill in Skill}
        ranked = sorted(Skill, key=lambda skill: (-utilities[skill.value], skill.value))
        best, worst = ranked[0], ranked[-1]
        # Supervise only a resolved winner; best-vs-worst alone can hide a tied
        # best/runner-up and produce arbitrary SFT targets.
        winner_gap = utilities[best.value] - utilities[ranked[1].value]
        margin = utilities[best.value] - utilities[worst.value]
        if winner_gap <= minimum_margin or margin <= minimum_margin:
            continue
        variants = [(split, "train" if split == "train" else "validation")]
        if split == "test":
            variants = [("seen_test", "train"), ("unseen_test", "unseen_test")]
        for destination, instruction_split in variants:
            prompt = planner_messages(context, preference=preference, instruction=instruction_for(
                preference, split=instruction_split, stable_key=record["anchor_id"]))
            chosen, rejected = skill_target(best), skill_target(worst)
            rows.append({
                "schema_version": DATASET_SCHEMA, "action_schema": ACTION_SCHEMA,
                "context_schema": CONTEXT_SCHEMA, "split": destination,
                "seed": seed, "anchor_id": record["anchor_id"],
                "instruction_split": instruction_split, "preference": preference.value,
                "target_skill": best.value, "prompt": prompt, "chosen": chosen,
                "rejected": rejected, "margin": margin, "winner_gap": winner_gap,
                "utilities": utilities,
                "conversations": [*prompt, {"role": "assistant", "content": chosen}],
            })
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--specialist-checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--horizon", type=int, default=32)
    parser.add_argument("--candidate-steps", type=int, default=8)
    parser.add_argument("--anchor-stride", type=int, default=8)
    parser.add_argument("--max-anchors", type=int, default=320)
    parser.add_argument("--minimum-margin", type=float, default=0.01)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--reuse-environments", action="store_true",
                        help="Reuse two independent environments for repeated branch verification; default uses fresh instances")
    args = parser.parse_args()
    if args.anchor_stride < 1 or args.max_anchors < 1 or not 1 <= args.candidate_steps <= args.horizon or not math.isfinite(args.minimum_margin) or args.minimum_margin < 0:
        parser.error("Invalid horizon, sampling limit or preference margin")
    private_directory(args.output_dir)
    if any(args.output_dir.iterdir()):
        raise ValueError("Choose an empty output directory; never overwrite a dataset version")
    from .environment import ContinuousBotEvadeEnv
    from .specialist import ContinuousCheckpointPolicy
    specialist = ContinuousCheckpointPolicy(args.specialist_checkpoint, device=args.device)
    contract = json.loads(EXPERIMENT.read_text())
    pool = contract["seed_pools"]["data_collection"]
    anchors = []
    seen = set()
    environment_config = None
    with args.trace.open() as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            seed = int(row["seed"])
            if not pool["start"] <= seed < pool["stop_exclusive"]:
                raise ValueError("Only fresh data_collection seeds may produce training labels")
            config = row.get("environment_config")
            if not isinstance(config, dict):
                raise ValueError("Trace lacks its environment configuration; recollect with the continuous evaluator")
            if environment_config is None:
                environment_config = config
            elif environment_config != config:
                raise ValueError("Cannot mix traces from different environments")
            key = (seed, len(row["prefix_xy_actions"]))
            if key in seen:
                raise ValueError("Duplicate trace seed/step")
            seen.add(key)
            if key[1] % args.anchor_stride == 0:
                row["anchor_id"] = hashlib.sha256(json.dumps(key).encode()).hexdigest()[:20]
                anchors.append(row)
    # Round-robin across seeds, with independently hashed timepoints within
    # each episode. A bounded budget must not select only the first few steps.
    by_seed = defaultdict(list)
    for row in anchors:
        by_seed[row["seed"]].append(row)
    for rows in by_seed.values():
        rows.sort(key=lambda row: hashlib.sha256(row["anchor_id"].encode()).digest())
    anchors = []
    for offset in range(max((len(rows) for rows in by_seed.values()), default=0)):
        for seed in sorted(by_seed):
            if offset < len(by_seed[seed]) and len(anchors) < args.max_anchors:
                anchors.append(by_seed[seed][offset])
    if not anchors:
        raise ValueError("No eligible training anchors")
    from .navigation import navigation_contract
    destination_mode = environment_config.get("destination_mode")
    projection_epsilon = environment_config.get("projection_epsilon")
    if (destination_mode not in {"project", "reject"}
            or isinstance(projection_epsilon, bool) or not isinstance(projection_epsilon, (int, float))
            or not math.isfinite(projection_epsilon) or not 0.0 < projection_epsilon <= 0.01):
        raise ValueError("Trace lacks a valid destination mode and projection epsilon; recollect it")
    if environment_config.get("navigation_contract") != navigation_contract(destination_mode):
        raise ValueError("Trace navigation semantics differ from this continuous implementation")
    if environment_config.get("reset_contract") != "fresh_navigation_episode_reset_v1":
        raise ValueError("Trace reset semantics differ from this continuous implementation")
    env_factory = lambda: ContinuousBotEvadeEnv(
        world_name=environment_config["world"], max_step=environment_config["max_steps"],
        time_step=environment_config["time_step"],
        predator_prey_forward_speed_ratio=environment_config["predator_speed_ratio"],
        max_line_of_sight_distance=environment_config["max_line_of_sight_distance"],
        destination_mode=destination_mode, projection_epsilon=projection_epsilon)
    outputs = {name: [] for name in ("train", "validation", "seen_test", "unseen_test")}
    failures = []
    record_path = args.output_dir / "counterfactuals_private.jsonl"
    owners = environment_pool(env_factory, 2) if args.reuse_environments else nullcontext(None)
    with owners as environments:
        with os.fdopen(os.open(record_path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600), "w") as handle:
            for anchor in anchors:
                try:
                    branches = {}
                    for skill in Skill:
                        branches[skill.value] = evaluate_repeated_branch(
                            env_factory, anchor, specialist, skill, horizon=args.horizon,
                            candidate_steps=args.candidate_steps, environments=environments)
                    record = {"anchor_id": anchor["anchor_id"], "seed": anchor["seed"], "context": anchor["context"], "branches": branches}
                    handle.write(json.dumps(record, sort_keys=True, allow_nan=False) + "\n")
                    for row in build_rows(record, minimum_margin=args.minimum_margin):
                        outputs[row["split"]].append(row)
                except ValueError as exc:
                    failures.append({"anchor_id": anchor["anchor_id"], "reason": str(exc)})
    # Context-exclusive evaluation partitions prevent copied public inputs from
    # appearing across seed splits. Retain different preferences within a split.
    owner = {}
    for split in ("train", "validation", "seen_test", "unseen_test"):
        kept = []
        for row in outputs[split]:
            key = json.loads(row["prompt"][1]["content"])["context"]
            key = json.dumps(key, sort_keys=True, separators=(",", ":"))
            group = "test" if split in {"seen_test", "unseen_test"} else split
            if owner.setdefault(key, group) == group:
                kept.append(row)
        outputs[split] = kept
        destination = args.output_dir / f"planner_{split}.jsonl"
        with os.fdopen(os.open(destination, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600), "w") as handle:
            for row in kept:
                handle.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")
    manifest = {
        "schema_version": DATASET_SCHEMA, "action_schema": ACTION_SCHEMA, "context_schema": CONTEXT_SCHEMA,
        "specialist_sha256": sha256(args.specialist_checkpoint), "trace_sha256": sha256(args.trace),
        "experiment_sha256": sha256(EXPERIMENT), "horizon": args.horizon, "candidate_steps": args.candidate_steps,
        "continuation": "public_visibility_rule", "minimum_margin": args.minimum_margin,
        "environment_config": environment_config, "memory_steps": 8, "evade_distance": 0.35,
        "reuse_environments": args.reuse_environments, "device": str(args.device),
        "replay_environment_lifecycle": "two_independent_reused_instances" if args.reuse_environments else "fresh_instance_per_branch",
        "sampled_step_counts": dict(Counter(len(row["prefix_xy_actions"]) for row in anchors)),
        "replay_scope": "public observation + physical/controller/RNG fingerprint + repeated outcomes",
        "attempted_anchors": len(anchors), "rejected_anchors": len(failures),
        "splits": {name: {"rows": len(rows), "sha256": sha256(args.output_dir / f"planner_{name}.jsonl")} for name, rows in outputs.items()},
        "research_evidence": False, "status": "generated_training_data_not_a_policy_result",
    }
    private_json(args.output_dir / "manifest.json", manifest)
    private_json(args.output_dir / "replay_failures.json", failures)
    print(json.dumps(manifest, sort_keys=True))
    if not outputs["train"] or not outputs["validation"]:
        raise SystemExit("Insufficient verified train/validation data; inspect replay failures and expand collection")


if __name__ == "__main__":
    main()
