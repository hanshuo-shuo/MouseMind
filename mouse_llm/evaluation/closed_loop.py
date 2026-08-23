"""Seeded, paired closed-loop evaluation for Cellworld policies."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import resource
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Protocol, Sequence

import numpy as np

from mouse_llm.data.schema import FEATURE_NAMES, make_conversation
from mouse_llm.evaluation.evaluate_policy import (
    ActionTokenConstraint,
    _chat_text,
    generate_response,
    load_action_catalog,
    parse_action,
)


# With frame_stack_k=1, the current Gym adapter returns the ten STACK_FIELDS
# followed by five non-stacked fields. The legacy 10D dataset predates that
# layout: its source wrapper removed predator_visible and retained goal/puff
# state. These source indices recreate the legacy order. Angles also changed
# from [-pi, pi] in the export to [0, 2*pi) in the extracted environment.
LEGACY_GYM_SOURCE_INDICES = (0, 1, 2, 4, 5, 6, 14, 10, 11, 12)
LEGACY_ANGLE_TARGET_INDICES = (2, 5)
DEFAULT_OBSERVATION_AUDIT = (
    Path(__file__).resolve().parents[1]
    / "reports/observation_contract_audit.json"
)


@dataclass(frozen=True)
class PolicyDecision:
    action: int
    valid: bool = True
    raw_response: str | None = None
    metadata: dict[str, Any] | None = None


class Policy(Protocol):
    name: str

    def reset(self, seed: int) -> None: ...

    def act(self, observation: np.ndarray) -> PolicyDecision: ...


@dataclass(frozen=True)
class EpisodeResult:
    policy: str
    seed: int
    episode_return: float
    success: int
    captured: int
    captures: int
    survived: int
    steps: int
    path_length: float
    path_efficiency: float
    valid_action_rate: float
    latency_mean_seconds: float
    latency_p50_seconds: float
    latency_p95_seconds: float
    latency_p99_seconds: float
    deadline_miss_rate: float
    latency_samples_seconds: tuple[float, ...]
    failure_mode: str
    first_predator_visible_step: int | None
    first_capture_step: int | None
    predator_visible_steps: int
    open_space_visible_rate: float
    capture_near_occlusion: int
    capture_in_open_space: int
    action_switch_rate: float
    oscillation_score: float
    net_displacement: float
    recent_path_length: float
    goal_distance_start: float
    goal_distance_min: float
    goal_distance_end: float
    planner_calls: int
    planner_call_rate: float
    skill_switch_rate: float
    dominant_skill: str | None
    skill_counts: str


def policy_observation(
    observation: Sequence[float], *, indices: tuple[int, ...]
) -> np.ndarray:
    """Apply the explicit Gym-to-legacy observation adapter."""
    array = np.asarray(observation, dtype=np.float32).reshape(-1)
    if len(array) == len(FEATURE_NAMES):
        if not np.isfinite(array).all():
            raise ValueError("Policy observation contains non-finite values")
        return array.copy()
    if len(indices) != len(FEATURE_NAMES):
        raise ValueError(f"Policy adapter must select {len(FEATURE_NAMES)} values")
    if not indices or min(indices) < 0 or max(indices) >= len(array):
        raise ValueError(
            f"Observation of length {len(array)} cannot satisfy indices {indices}"
        )
    selected = array[list(indices)]
    selected = selected.copy()
    for target_index in LEGACY_ANGLE_TARGET_INDICES:
        selected[target_index] = (
            (selected[target_index] + np.pi) % (2 * np.pi)
        ) - np.pi
    if not np.isfinite(selected).all():
        raise ValueError("Policy observation contains non-finite values")
    return selected


class RandomPolicy:
    def __init__(self, *, action_count: int, name: str = "random"):
        self.name = name
        self.action_count = action_count
        self.rng = np.random.default_rng()

    def reset(self, seed: int) -> None:
        self.rng = np.random.default_rng(seed)

    def act(self, observation: np.ndarray) -> PolicyDecision:
        del observation
        return PolicyDecision(int(self.rng.integers(0, self.action_count)))


class MLPCheckpointPolicy:
    def __init__(
        self,
        checkpoint: Path,
        *,
        device: str,
        observation_indices: tuple[int, ...],
        name: str = "mlp-bc",
    ):
        import torch

        from mouse_llm.baselines.mlp_bc import load_checkpoint

        self.name = name
        self.torch = torch
        self.device = torch.device(device)
        self.model, self.mean, self.std, self.metadata = load_checkpoint(
            checkpoint, device=self.device
        )
        self.action_count = int(self.metadata["action_count"])
        self.observation_indices = observation_indices

    def reset(self, seed: int) -> None:
        del seed

    def act(self, observation: np.ndarray) -> PolicyDecision:
        values = policy_observation(
            observation, indices=self.observation_indices
        )
        tensor = self.torch.from_numpy(values).to(self.device)
        tensor = ((tensor - self.mean) / self.std).unsqueeze(0)
        with self.torch.inference_mode():
            action = int(self.model(tensor).argmax(dim=-1).cpu())
        return PolicyDecision(action)


class MiniMindPolicy:
    def __init__(
        self,
        *,
        name: str,
        base_weight: Path,
        tokenizer_path: Path,
        action_count: int,
        hidden_size: int,
        num_hidden_layers: int,
        max_seq_len: int,
        max_new_tokens: int,
        device: str,
        observation_indices: tuple[int, ...],
        decode_mode: str,
        fallback_action: int,
        lora_weight: Path | None = None,
    ):
        import torch
        from transformers import AutoTokenizer

        from model.model_lora import apply_lora, load_lora
        from model.model_minimind import MiniMindConfig, MiniMindForCausalLM

        self.name = name
        self.torch = torch
        self.device = torch.device(device)
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        config = MiniMindConfig(
            hidden_size=hidden_size, num_hidden_layers=num_hidden_layers
        )
        model = MiniMindForCausalLM(config)
        state = torch.load(base_weight, map_location="cpu", weights_only=True)
        model.load_state_dict(state, strict=True)
        if lora_weight is not None:
            apply_lora(model)
            load_lora(model, lora_weight)
        self.model = (
            model.half().to(self.device)
            if self.device.type == "cuda"
            else model.float().to(self.device)
        )
        self.model.eval()
        self.action_count = action_count
        self.max_seq_len = max_seq_len
        self.max_new_tokens = max_new_tokens
        self.observation_indices = observation_indices
        self.fallback_action = fallback_action
        self.constraint = (
            ActionTokenConstraint(self.tokenizer, action_count=action_count)
            if decode_mode == "json-constrained"
            else None
        )

    def reset(self, seed: int) -> None:
        self.torch.manual_seed(seed)

    def act(self, observation: np.ndarray) -> PolicyDecision:
        values = policy_observation(
            observation, indices=self.observation_indices
        )
        messages = make_conversation(
            values, self.fallback_action, action_count=self.action_count
        )["conversations"][:-1]
        prompt_text = _chat_text(self.tokenizer, messages, generate=True)
        encoded = self.tokenizer(
            prompt_text,
            return_tensors="pt",
            add_special_tokens=True,
            truncation=True,
            max_length=self.max_seq_len,
        )
        prompt_inputs = {
            "input_ids": encoded["input_ids"].to(self.device),
            "attention_mask": encoded["attention_mask"].to(self.device),
        }
        response, _ = generate_response(
            self.model,
            self.tokenizer,
            prompt_inputs,
            max_new_tokens=self.max_new_tokens,
            action_constraint=self.constraint,
        )
        parsed = parse_action(response, action_count=self.action_count)
        return PolicyDecision(
            self.fallback_action if parsed is None else parsed,
            valid=parsed is not None,
            raw_response=response[:256],
        )


def _quantiles(values: Sequence[float]) -> tuple[float, float, float, float]:
    array = np.asarray(values, dtype=np.float64)
    return (
        float(array.mean()),
        float(np.quantile(array, 0.50)),
        float(np.quantile(array, 0.95)),
        float(np.quantile(array, 0.99)),
    )


def _prey_position(observation: Sequence[float]) -> np.ndarray:
    values = np.asarray(observation, dtype=np.float64).reshape(-1)
    if len(values) < 2 or not np.isfinite(values[:2]).all():
        raise ValueError("Environment observation must begin with finite prey x/y")
    return values[:2]


def _goal_distance(env: Any, observation: Sequence[float]) -> float:
    try:
        return float(env.model.prey_data.prey_goal_distance)
    except AttributeError:
        if hasattr(env, "goal_distance"):
            return float(env.goal_distance(observation))
    return 0.0


def classify_failure(
    *,
    success: int,
    captures: int,
    first_predator_visible_step: int | None,
    first_capture_step: int | None,
    capture_near_occlusion: int,
    capture_in_open_space: int,
    oscillation_score: float,
    recent_path_length: float,
    goal_distance_start: float,
    goal_distance_min: float,
    goal_distance_end: float,
) -> str:
    """Assign one exclusive, deterministic primary outcome/failure label."""
    if captures > 0:
        if (
            first_predator_visible_step is not None
            and first_capture_step is not None
            and 0 <= first_capture_step - first_predator_visible_step <= 4
        ):
            return "late_predator_response"
        if capture_near_occlusion:
            return "capture_near_occlusion"
        if capture_in_open_space:
            return "open_space_capture"
        return "capture_other"
    if success:
        return "success"
    if oscillation_score >= 0.45:
        return "navigation_oscillation"
    if recent_path_length <= 0.03:
        return "stuck_timeout"
    if goal_distance_min <= 0.15 and goal_distance_end >= goal_distance_min + 0.08:
        return "goal_overshoot"
    if goal_distance_end >= goal_distance_start + 0.10:
        return "wrong_way_navigation"
    return "timeout_other"


def rollout_episode(
    env: Any,
    policy: Policy,
    *,
    seed: int,
    control_budget_seconds: float,
) -> EpisodeResult:
    observation, _ = env.reset(seed=seed)
    policy.reset(seed)
    position = _prey_position(observation)
    initial_position = position.copy()
    positions = [position.copy()]
    initial_goal_distance = max(_goal_distance(env, observation), 0.0)
    goal_distances = [initial_goal_distance]
    path_length = 0.0
    episode_return = 0.0
    latencies: list[float] = []
    valid_actions = 0
    terminal_info: dict[str, Any] = {}
    steps = 0
    actions: list[int] = []
    first_predator_visible_step: int | None = None
    first_capture_step: int | None = None
    predator_visible_steps = 0
    open_space_visible_steps = 0
    capture_near_occlusion = 0
    capture_in_open_space = 0
    planner_calls = 0
    skills: list[str] = []
    while True:
        start = time.perf_counter()
        policy_input = (
            env.legacy_policy_observation()
            if hasattr(env, "legacy_policy_observation")
            else np.asarray(observation)
        )
        decision = policy.act(policy_input)
        latency = time.perf_counter() - start
        latencies.append(latency)
        valid_actions += int(decision.valid)
        if decision.metadata:
            planner_calls += int(bool(decision.metadata.get("replanned")))
            skill = decision.metadata.get("skill")
            if isinstance(skill, str):
                skills.append(skill)
        actions.append(decision.action)
        observation, reward, terminated, truncated, terminal_info = env.step(
            decision.action
        )
        raw_observation = np.asarray(observation).reshape(-1)
        if len(raw_observation) >= 15:
            predator_visible = bool(raw_observation[3])
            near_wall = bool(raw_observation[7])
            near_occlusion = bool(raw_observation[8])
            puffed = bool(raw_observation[10])
            if predator_visible:
                predator_visible_steps += 1
                if first_predator_visible_step is None:
                    first_predator_visible_step = steps + 1
                if not near_wall and not near_occlusion:
                    open_space_visible_steps += 1
            if puffed:
                if first_capture_step is None:
                    first_capture_step = steps + 1
                capture_near_occlusion |= int(near_occlusion)
                capture_in_open_space |= int(not near_wall and not near_occlusion)
        next_position = _prey_position(observation)
        path_length += float(np.linalg.norm(next_position - position))
        position = next_position
        positions.append(position.copy())
        goal_distances.append(max(_goal_distance(env, observation), 0.0))
        episode_return += float(reward)
        steps += 1
        if terminated or truncated:
            break
    captures = int(terminal_info.get("captures", 0))
    if captures > 0 and first_capture_step is None:
        first_capture_step = steps
    success = int(bool(terminal_info.get("is_success", False)))
    survived = int(bool(terminal_info.get("survived", captures == 0)))
    latency_mean, latency_p50, latency_p95, latency_p99 = _quantiles(latencies)
    if success and initial_goal_distance > 0:
        path_efficiency = initial_goal_distance / max(
            path_length, initial_goal_distance
        )
    else:
        path_efficiency = 0.0
    action_switch_rate = (
        float(np.mean(np.asarray(actions[1:]) != np.asarray(actions[:-1])))
        if len(actions) >= 2
        else 0.0
    )
    oscillation_score = (
        float(
            np.mean(
                [
                    actions[index] == actions[index - 2]
                    and actions[index] != actions[index - 1]
                    for index in range(2, len(actions))
                ]
            )
        )
        if len(actions) >= 3
        else 0.0
    )
    recent_positions = positions[-min(len(positions), 21) :]
    recent_path_length = float(
        sum(
            np.linalg.norm(right - left)
            for left, right in zip(
                recent_positions[:-1], recent_positions[1:], strict=True
            )
        )
    )
    goal_distance_min = min(goal_distances)
    goal_distance_end = goal_distances[-1]
    failure_mode = classify_failure(
        success=success,
        captures=captures,
        first_predator_visible_step=first_predator_visible_step,
        first_capture_step=first_capture_step,
        capture_near_occlusion=capture_near_occlusion,
        capture_in_open_space=capture_in_open_space,
        oscillation_score=oscillation_score,
        recent_path_length=recent_path_length,
        goal_distance_start=initial_goal_distance,
        goal_distance_min=goal_distance_min,
        goal_distance_end=goal_distance_end,
    )
    skill_counts_map: dict[str, int] = {}
    for skill in skills:
        skill_counts_map[skill] = skill_counts_map.get(skill, 0) + 1
    skill_switch_rate = (
        float(np.mean(np.asarray(skills[1:]) != np.asarray(skills[:-1])))
        if len(skills) >= 2
        else 0.0
    )
    dominant_skill = (
        max(skill_counts_map, key=skill_counts_map.get) if skill_counts_map else None
    )
    return EpisodeResult(
        policy=policy.name,
        seed=seed,
        episode_return=episode_return,
        success=success,
        captured=int(captures > 0),
        captures=captures,
        survived=survived,
        steps=steps,
        path_length=path_length,
        path_efficiency=path_efficiency,
        valid_action_rate=valid_actions / steps,
        latency_mean_seconds=latency_mean,
        latency_p50_seconds=latency_p50,
        latency_p95_seconds=latency_p95,
        latency_p99_seconds=latency_p99,
        deadline_miss_rate=float(
            np.mean(np.asarray(latencies) > control_budget_seconds)
        ),
        latency_samples_seconds=tuple(latencies),
        failure_mode=failure_mode,
        first_predator_visible_step=first_predator_visible_step,
        first_capture_step=first_capture_step,
        predator_visible_steps=predator_visible_steps,
        open_space_visible_rate=(
            open_space_visible_steps / predator_visible_steps
            if predator_visible_steps
            else 0.0
        ),
        capture_near_occlusion=capture_near_occlusion,
        capture_in_open_space=capture_in_open_space,
        action_switch_rate=action_switch_rate,
        oscillation_score=oscillation_score,
        net_displacement=float(np.linalg.norm(position - initial_position)),
        recent_path_length=recent_path_length,
        goal_distance_start=initial_goal_distance,
        goal_distance_min=goal_distance_min,
        goal_distance_end=goal_distance_end,
        planner_calls=planner_calls,
        planner_call_rate=planner_calls / steps,
        skill_switch_rate=skill_switch_rate,
        dominant_skill=dominant_skill,
        skill_counts=json.dumps(skill_counts_map, sort_keys=True, separators=(",", ":")),
    )


def run_policy(
    env_factory: Callable[[], Any],
    policy: Policy,
    *,
    seeds: Sequence[int],
    control_budget_seconds: float,
    warmup_actions: int = 0,
) -> list[EpisodeResult]:
    env = env_factory()
    try:
        policy_action_count = getattr(policy, "action_count", None)
        environment_action_count = getattr(getattr(env, "action_space", None), "n", None)
        if (
            policy_action_count is not None
            and environment_action_count is not None
            and int(policy_action_count) != int(environment_action_count)
        ):
            raise ValueError(
                f"{policy.name} has {policy_action_count} actions but the "
                f"environment has {environment_action_count}"
            )
        if warmup_actions > 0:
            observation, _ = env.reset(seed=int(seeds[0]))
            policy.reset(int(seeds[0]))
            for _ in range(warmup_actions):
                policy_input = (
                    env.legacy_policy_observation()
                    if hasattr(env, "legacy_policy_observation")
                    else np.asarray(observation)
                )
                policy.act(policy_input)
        rows: list[EpisodeResult] = []
        for index, seed in enumerate(seeds, start=1):
            rows.append(
                rollout_episode(
                    env,
                    policy,
                    seed=int(seed),
                    control_budget_seconds=control_budget_seconds,
                )
            )
            print(
                f"{policy.name}: episode {index}/{len(seeds)} seed={seed}",
                flush=True,
            )
        return rows
    finally:
        env.close()


def bootstrap_mean(
    values: Sequence[float], *, seed: int, iterations: int = 2000
) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    if not len(array):
        raise ValueError("Cannot bootstrap an empty metric")
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(array), size=(iterations, len(array)))
    sampled_means = array[indices].mean(axis=1)
    return {
        "mean": float(array.mean()),
        "ci_low": float(np.quantile(sampled_means, 0.025)),
        "ci_high": float(np.quantile(sampled_means, 0.975)),
    }


SUMMARY_FIELDS = {
    "return": "episode_return",
    "success_rate": "success",
    "capture_rate": "captured",
    "survival_rate": "survived",
    "captures": "captures",
    "steps": "steps",
    "path_length": "path_length",
    "path_efficiency": "path_efficiency",
    "valid_action_rate": "valid_action_rate",
    "deadline_miss_rate": "deadline_miss_rate",
    "planner_call_rate": "planner_call_rate",
    "skill_switch_rate": "skill_switch_rate",
    "open_space_visible_rate": "open_space_visible_rate",
    "oscillation_score": "oscillation_score",
}

FAILURE_TAXONOMY_DEFINITIONS = {
    "late_predator_response": "capture within four steps of first predator visibility",
    "capture_near_occlusion": "capture while the prey is near an occlusion",
    "open_space_capture": "capture away from walls and occlusions",
    "capture_other": "capture without a more specific spatial/temporal label",
    "navigation_oscillation": "timeout with A/B/A action alternation score >= 0.45",
    "stuck_timeout": "timeout with <= 0.03 path length over the final 20 steps",
    "goal_overshoot": "approached within 0.15 then ended at least 0.08 farther away",
    "wrong_way_navigation": "final goal distance at least 0.10 worse than start",
    "timeout_other": "timeout not matched by another deterministic rule",
}


def summarize_policy(
    rows: Sequence[EpisodeResult], *, seed: int
) -> dict[str, Any]:
    summary: dict[str, Any] = {"episode_count": len(rows)}
    for offset, (name, field) in enumerate(SUMMARY_FIELDS.items()):
        summary[name] = bootstrap_mean(
            [float(getattr(row, field)) for row in rows],
            seed=seed + offset,
        )
    action_latencies = np.asarray(
        [latency for row in rows for latency in row.latency_samples_seconds],
        dtype=np.float64,
    )
    summary["latency_seconds"] = {
        "mean": float(action_latencies.mean()),
        "p50": float(np.quantile(action_latencies, 0.50)),
        "p95": float(np.quantile(action_latencies, 0.95)),
        "p99": float(np.quantile(action_latencies, 0.99)),
        "mean_action_hz": (
            float(1.0 / action_latencies.mean())
            if action_latencies.mean() > 0
            else float("inf")
        ),
    }
    failure_rows = [row for row in rows if row.failure_mode != "success"]
    failure_counts: dict[str, int] = {}
    for row in failure_rows:
        failure_counts[row.failure_mode] = failure_counts.get(row.failure_mode, 0) + 1
    summary["failure_taxonomy"] = {
        "failed_episode_count": len(failure_rows),
        "counts": dict(sorted(failure_counts.items())),
        "rates_among_failures": {
            name: count / len(failure_rows)
            for name, count in sorted(failure_counts.items())
        }
        if failure_rows
        else {},
        "definitions_version": 1,
    }
    return summary


def paired_differences(
    reference: Sequence[EpisodeResult],
    candidate: Sequence[EpisodeResult],
    *,
    seed: int,
) -> dict[str, Any]:
    reference_by_seed = {row.seed: row for row in reference}
    candidate_by_seed = {row.seed: row for row in candidate}
    if reference_by_seed.keys() != candidate_by_seed.keys():
        raise ValueError("Paired policies must have identical episode seeds")
    seeds = sorted(reference_by_seed)
    result: dict[str, Any] = {
        "candidate_minus_reference": True,
        "episode_count": len(seeds),
    }
    for offset, (name, field) in enumerate(SUMMARY_FIELDS.items()):
        differences = [
            float(getattr(candidate_by_seed[item], field))
            - float(getattr(reference_by_seed[item], field))
            for item in seeds
        ]
        result[name] = bootstrap_mean(differences, seed=seed + offset)
    return result


def _peak_rss_bytes() -> int:
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(value if sys.platform == "darwin" else value * 1024)


def _system_memory() -> dict[str, int]:
    memory = {"peak_process_rss_bytes": _peak_rss_bytes()}
    try:
        import psutil

        memory["process_rss_bytes"] = int(psutil.Process().memory_info().rss)
    except ImportError:
        pass
    try:
        import torch

        if torch.cuda.is_available():
            memory["cuda_peak_allocated_bytes"] = int(
                torch.cuda.max_memory_allocated()
            )
    except ImportError:
        pass
    return memory


def build_report(
    results: dict[str, list[EpisodeResult]],
    *,
    seed: int,
    reference_policy: str,
    metadata: dict[str, Any],
) -> dict[str, Any]:
    if reference_policy not in results:
        raise ValueError(f"Unknown reference policy {reference_policy!r}")
    report: dict[str, Any] = {
        "schema_version": 1,
        "experiment": "mousemind_seeded_closed_loop",
        "metadata": metadata,
        "policies": {},
        "paired_comparisons": {},
        "failure_taxonomy": {
            "definitions_version": 1,
            "definitions": FAILURE_TAXONOMY_DEFINITIONS,
        },
        "system": _system_memory(),
    }
    for offset, (name, rows) in enumerate(results.items()):
        report["policies"][name] = summarize_policy(rows, seed=seed + offset * 100)
        if name != reference_policy:
            key = f"{name}_minus_{reference_policy}"
            report["paired_comparisons"][key] = paired_differences(
                results[reference_policy], rows, seed=seed + offset * 1000
            )
    return report


def _write_outputs(
    output_dir: Path,
    report: dict[str, Any],
    results: dict[str, list[EpisodeResult]],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
    metrics_path = output_dir / "closed_loop_metrics.json"
    temporary = metrics_path.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(temporary, metrics_path)
    rows = []
    for policy_rows in results.values():
        for row in policy_rows:
            payload = asdict(row)
            # Per-action samples feed exact aggregate percentiles but would make
            # the episode CSV unwieldy; its per-episode p50/p95/p99 remain.
            payload.pop("latency_samples_seconds")
            rows.append(payload)
    csv_path = output_dir / "closed_loop_episodes.csv"
    temporary_csv = csv_path.with_suffix(".csv.tmp")
    with temporary_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary_csv, csv_path)
    metrics_path.chmod(0o600)
    csv_path.chmod(0o600)


def _parse_indices(raw: str) -> tuple[int, ...]:
    try:
        indices = tuple(int(value) for value in raw.split(","))
    except ValueError as exc:
        raise argparse.ArgumentTypeError("indices must be comma-separated integers") from exc
    if len(indices) != len(FEATURE_NAMES) or len(set(indices)) != len(indices):
        raise argparse.ArgumentTypeError(
            f"exactly {len(FEATURE_NAMES)} unique indices are required"
        )
    return indices


def load_verified_observation_audit(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if (
        payload.get("schema_version") != 1
        or payload.get("audit") != "legacy_mouse_observation_contract"
        or payload.get("verified") is not True
        or payload.get("research_evidence") is not True
    ):
        raise ValueError(f"Observation audit is not verified: {path}")
    contract = payload.get("contract", {})
    if contract.get("schema_name") != "legacy_mouse_vector_10d_v1":
        raise ValueError(f"Observation audit has the wrong contract: {path}")
    return {
        "artifact": path.name,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "verified": True,
        "source_commit": contract["source"]["commit"],
        "source_sha256": contract["source"]["sha256"],
        "state_replay_coverage": payload["state_replay"]["coverage"],
        "dataset_rows": payload["dataset_distribution"]["row_count"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run paired seeded Cellworld policy rollouts"
    )
    parser.add_argument(
        "--policies",
        nargs="+",
        choices=(
            "random",
            "mlp-bc",
            "hierarchical-mlp",
            "minimind-base",
            "minimind-lora",
        ),
        default=("random",),
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--seed-start", type=int, default=1000)
    parser.add_argument("--reference-policy")
    parser.add_argument("--world", default="21_05")
    parser.add_argument("--max-steps", type=int, default=300)
    parser.add_argument("--time-step", type=float, default=0.25)
    parser.add_argument("--action-count", type=int, default=295)
    parser.add_argument("--warmup-actions", type=int, default=3)
    parser.add_argument(
        "--observation-indices",
        type=_parse_indices,
        default=LEGACY_GYM_SOURCE_INDICES,
    )
    parser.add_argument("--mlp-checkpoint", type=Path)
    parser.add_argument(
        "--action-catalog",
        type=Path,
        default=Path(__file__).resolve().parents[1]
        / "envs/mice/assets/action_catalog_21_05.json",
    )
    parser.add_argument("--base-weight", type=Path)
    parser.add_argument("--lora-weight", type=Path)
    parser.add_argument("--tokenizer", type=Path, default=Path("model"))
    parser.add_argument("--hidden-size", type=int, default=768)
    parser.add_argument("--num-hidden-layers", type=int, default=8)
    parser.add_argument("--max-seq-len", type=int, default=256)
    parser.add_argument("--max-new-tokens", type=int, default=24)
    parser.add_argument(
        "--decode-mode",
        choices=("free", "json-constrained"),
        default="json-constrained",
    )
    parser.add_argument("--fallback-action", type=int, default=0)
    parser.add_argument(
        "--instruction",
        default="Reach the goal while prioritizing survival and avoiding the predator.",
    )
    parser.add_argument("--planner-horizon", type=int, default=4)
    parser.add_argument("--evade-distance", type=float, default=0.35)
    parser.add_argument(
        "--observation-audit",
        type=Path,
        default=DEFAULT_OBSERVATION_AUDIT,
        help="Machine-readable verified state-replay audit used as evidence.",
    )
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument(
        "--torch-threads",
        type=int,
        default=1,
        help="CPU threads; one avoids overhead in Cellworld's small geometry kernels.",
    )
    args = parser.parse_args()

    if args.episodes <= 0 or args.max_steps <= 0 or args.time_step <= 0:
        raise ValueError("episodes, max-steps, and time-step must be positive")
    if args.torch_threads <= 0:
        raise ValueError("torch-threads must be positive")
    if args.planner_horizon <= 0 or args.evade_distance <= 0:
        raise ValueError("planner-horizon and evade-distance must be positive")
    if not 0 <= args.fallback_action < args.action_count:
        raise ValueError("fallback-action is outside the action space")
    if len(set(args.policies)) != len(args.policies):
        raise ValueError("policies must be unique")
    try:
        import torch

        torch.set_num_threads(args.torch_threads)

        device = (
            "cuda"
            if args.device == "auto" and torch.cuda.is_available()
            else "cpu" if args.device == "auto" else args.device
        )
    except ImportError:
        device = "cpu" if args.device == "auto" else args.device

    from mouse_llm.envs.mice import BotEvadeEnv, custom_reward

    def env_factory():
        return BotEvadeEnv(
            world_name=args.world,
            use_lppos=False,
            use_predator=True,
            max_step=args.max_steps,
            reward_function=custom_reward,
            time_step=args.time_step,
            frame_stack_k=1,
        )

    policies: list[Policy] = []
    for name in args.policies:
        if name == "random":
            policies.append(RandomPolicy(action_count=args.action_count))
        elif name in ("mlp-bc", "hierarchical-mlp"):
            if args.mlp_checkpoint is None:
                parser.error(f"--mlp-checkpoint is required for {name}")
            specialist = MLPCheckpointPolicy(
                args.mlp_checkpoint,
                device=device,
                observation_indices=args.observation_indices,
                name="mlp-bc" if name == "mlp-bc" else "hierarchical-specialist",
            )
            if name == "mlp-bc":
                policies.append(specialist)
            else:
                from mouse_llm.hierarchical import HierarchicalPolicy

                destinations = load_action_catalog(args.action_catalog)
                policies.append(
                    HierarchicalPolicy(
                        specialist=specialist,
                        destinations=destinations,
                        instruction=args.instruction,
                        planner_horizon=args.planner_horizon,
                        evade_distance=args.evade_distance,
                    )
                )
        else:
            if args.base_weight is None:
                parser.error(f"--base-weight is required for {name}")
            lora_weight = None
            if name == "minimind-lora":
                if args.lora_weight is None:
                    parser.error("--lora-weight is required for minimind-lora")
                lora_weight = args.lora_weight
            policies.append(
                MiniMindPolicy(
                    name=name,
                    base_weight=args.base_weight,
                    lora_weight=lora_weight,
                    tokenizer_path=args.tokenizer,
                    action_count=args.action_count,
                    hidden_size=args.hidden_size,
                    num_hidden_layers=args.num_hidden_layers,
                    max_seq_len=args.max_seq_len,
                    max_new_tokens=args.max_new_tokens,
                    device=device,
                    observation_indices=args.observation_indices,
                    decode_mode=args.decode_mode,
                    fallback_action=args.fallback_action,
                )
            )
    seeds = list(range(args.seed_start, args.seed_start + args.episodes))
    observation_audit = load_verified_observation_audit(args.observation_audit)
    results: dict[str, list[EpisodeResult]] = {}
    for policy in policies:
        results[policy.name] = run_policy(
            env_factory,
            policy,
            seeds=seeds,
            control_budget_seconds=args.time_step,
            warmup_actions=args.warmup_actions,
        )
    reference = args.reference_policy or policies[0].name
    seed_digest = hashlib.sha256(
        ",".join(map(str, seeds)).encode("ascii")
    ).hexdigest()
    metadata = {
        "environment": "BotEvadeEnv",
        "world": args.world,
        "episode_count": args.episodes,
        "seed_start": args.seed_start,
        "seed_sha256": seed_digest,
        "max_steps": args.max_steps,
        "control_budget_seconds": args.time_step,
        "torch_threads": args.torch_threads,
        "policy_device": device,
        "hierarchical_policy": {
            "instruction": args.instruction,
            "planner_horizon": args.planner_horizon,
            "evade_distance": args.evade_distance,
            "planner": "InstructionSkillPlanner",
            "status": "auditable language baseline; MiniMind skill planner pending",
        },
        "decode_mode": args.decode_mode,
        "fallback_action": args.fallback_action,
        "reference_policy": reference,
        "synthetic": False,
        "research_evidence": True,
        "research_evidence_blockers": [],
        "observation_contract_audit": observation_audit,
        "observation_adapter": {
            "gym_indices": list(args.observation_indices),
            "wrapped_angle_target_indices": list(LEGACY_ANGLE_TARGET_INDICES),
            "target_schema": "legacy_mouse_vector_10d_v1",
            "runtime_encoder": "BotEvadeEnv.legacy_policy_observation",
            "semantic_equivalence_to_legacy_exporter_verified": True,
        },
    }
    report = build_report(
        results,
        seed=args.seed_start,
        reference_policy=reference,
        metadata=metadata,
    )
    _write_outputs(args.output_dir, report, results)
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
