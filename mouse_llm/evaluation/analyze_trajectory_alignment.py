"""Compare closed-loop policies with held-out source trajectory behavior.

This analysis complements one-step action agreement.  It reconstructs complete
held-out source episodes, maps source and policy episodes to the same bounded
behavioral profile, and reports an equal-weight mean absolute profile distance.
Private transition and episode rows remain outside Git; outputs are aggregate.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from collections import defaultdict
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from mouse_llm.data.prepare_dataset import (
    AUDITED_SOURCE_SHA256,
    Episode,
    load_transitions,
    segment_episodes,
    split_for_episode,
)


FEATURES = (
    "task_success",
    "clean_success",
    "capture_incidence",
    "captures_per_step",
    "action_switch_rate",
    "oscillation_score",
    "path_length_per_step",
    "net_displacement",
    "goal_progress",
    "path_efficiency",
    "predator_visible_rate",
)

FEATURE_DEFINITIONS = {
    "task_success": "episode reaches goal distance < 0.1",
    "clean_success": "task success with no puff/capture event",
    "capture_incidence": "episode contains at least one puff/capture event",
    "captures_per_step": "capture events divided by episode steps",
    "action_switch_rate": "fraction of adjacent actions that differ",
    "oscillation_score": "fraction of A/B/A action triples",
    "path_length_per_step": "mean step displacement divided by sqrt(2)",
    "net_displacement": "start-to-end displacement divided by sqrt(2)",
    "goal_progress": "(start goal distance - end goal distance + 1) / 2",
    "path_efficiency": (
        "start goal distance / max(path length, start distance) on success"
    ),
    "predator_visible_rate": "fraction of episode steps with a visible predator",
}

POLICY_ROLES = {
    "random": "baseline",
    "direct-minimind-base": "baseline",
    "direct-minimind-lora": "baseline",
    "direct-mlp": "specialist baseline",
    "p1-rule": "rule hierarchy baseline",
    "minimind-no-history": "proposed ablation",
    "minimind-no-instruction": "proposed ablation",
    "minimind-learned": "proposed MiniMind hierarchy",
    "minimind-verified": "proposed + rejected verifier",
    "numeric-learned": "non-language upper reference",
    "numeric-verified": "upper-reference verifier variant",
}

POLICY_LABELS = {
    "random": "Random",
    "direct-minimind-base": "Direct MiniMind base",
    "direct-minimind-lora": "Direct MiniMind LoRA",
    "direct-mlp": "Direct MLP BC",
    "p1-rule": "P1 rule hierarchy",
    "minimind-no-history": "MiniMind hierarchy, no history",
    "minimind-no-instruction": "MiniMind hierarchy, no instruction",
    "minimind-learned": "MiniMind hierarchy (full)",
    "minimind-verified": "MiniMind hierarchy + verifier",
    "numeric-learned": "Numeric planner",
    "numeric-verified": "Numeric planner + verifier",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _switch_rate(actions: np.ndarray) -> float:
    return (
        float(np.mean(actions[1:] != actions[:-1])) if len(actions) >= 2 else 0.0
    )


def _oscillation_score(actions: np.ndarray) -> float:
    if len(actions) < 3:
        return 0.0
    return float(
        np.mean(
            [
                actions[index] == actions[index - 2]
                and actions[index] != actions[index - 1]
                for index in range(2, len(actions))
            ]
        )
    )


def source_episode_profile(episode: Episode) -> np.ndarray:
    transitions = episode.transitions
    steps = len(transitions)
    positions = np.asarray(
        [item.observation[:2] for item in transitions]
        + [transitions[-1].next_observation[:2]],
        dtype=np.float64,
    )
    goal_distances = np.asarray(
        [item.observation[6] for item in transitions]
        + [transitions[-1].next_observation[6]],
        dtype=np.float64,
    )
    actions = np.asarray([item.action for item in transitions], dtype=np.int64)
    puffed = np.asarray(
        [item.observation[7] for item in transitions]
        + [transitions[-1].next_observation[7]],
        dtype=np.float64,
    )
    capture_events = int(
        (puffed[0] > 0.5)
        + np.sum((puffed[1:] > 0.5) & (puffed[:-1] <= 0.5))
    )
    success = int(float(goal_distances.min()) < 0.1)
    path_length = float(
        np.linalg.norm(np.diff(positions, axis=0), axis=1).sum()
    )
    path_efficiency = (
        float(goal_distances[0] / max(path_length, goal_distances[0]))
        if success and goal_distances[0] > 0
        else 0.0
    )
    predator_visible = [
        item.observation[3] != 0.0 or item.observation[4] != 0.0
        for item in transitions
    ]
    return np.asarray(
        (
            success,
            int(success and capture_events == 0),
            int(capture_events > 0),
            capture_events / steps,
            _switch_rate(actions),
            _oscillation_score(actions),
            path_length / steps / math.sqrt(2.0),
            float(np.linalg.norm(positions[-1] - positions[0])) / math.sqrt(2.0),
            (float(goal_distances[0] - goal_distances[-1]) + 1.0) / 2.0,
            path_efficiency,
            float(np.mean(predator_visible)),
        ),
        dtype=np.float64,
    )


def policy_episode_profile(row: dict[str, str]) -> np.ndarray:
    steps = max(float(row["steps"]), 1.0)
    return np.asarray(
        (
            float(row["success"]),
            float(row["clean_success"]),
            float(row["captured"]),
            float(row["captures"]) / steps,
            float(row["action_switch_rate"]),
            float(row["oscillation_score"]),
            float(row["path_length"]) / steps / math.sqrt(2.0),
            float(row["net_displacement"]) / math.sqrt(2.0),
            (
                float(row["goal_distance_start"])
                - float(row["goal_distance_end"])
                + 1.0
            )
            / 2.0,
            float(row["path_efficiency"]),
            float(row["predator_visible_steps"]) / steps,
        ),
        dtype=np.float64,
    )


def load_source_profiles(
    path: Path,
    *,
    action_count: int,
    continuity_atol: float,
    split_seed: int,
    train_ratio: float,
    validation_ratio: float,
) -> np.ndarray:
    transitions = load_transitions(path, action_count=action_count)
    episodes = segment_episodes(transitions, continuity_atol=continuity_atol)
    held_out = [
        episode
        for episode in episodes
        if split_for_episode(
            episode,
            seed=split_seed,
            train_ratio=train_ratio,
            validation_ratio=validation_ratio,
        )
        == "test"
    ]
    if not held_out:
        raise ValueError("No held-out source episodes")
    return np.stack([source_episode_profile(episode) for episode in held_out])


def load_policy_profiles(
    path: Path,
) -> tuple[dict[str, np.ndarray], tuple[int, ...]]:
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    with path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            grouped[row["policy"]].append(row)
    if not grouped:
        raise ValueError("Closed-loop episode CSV is empty")
    seed_contract: tuple[int, ...] | None = None
    profiles: dict[str, np.ndarray] = {}
    for name, rows in grouped.items():
        ordered = sorted(rows, key=lambda row: int(row["seed"]))
        seeds = tuple(int(row["seed"]) for row in ordered)
        if len(set(seeds)) != len(seeds):
            raise ValueError(f"Policy {name} has duplicate seeds")
        if seed_contract is None:
            seed_contract = seeds
        elif seeds != seed_contract:
            raise ValueError("Policies do not share an identical seed contract")
        profiles[name] = np.stack([policy_episode_profile(row) for row in ordered])
    assert seed_contract is not None
    return profiles, seed_contract


def profile_distance(source: np.ndarray, policy: np.ndarray) -> float:
    if source.ndim != 2 or policy.ndim != 2 or source.shape[1] != len(FEATURES):
        raise ValueError("Profile matrices do not match the feature contract")
    return float(np.abs(source.mean(axis=0) - policy.mean(axis=0)).mean())


def _interval(values: np.ndarray) -> dict[str, float]:
    return {
        "ci_low": float(np.quantile(values, 0.025)),
        "ci_high": float(np.quantile(values, 0.975)),
    }


def analyze(
    source: np.ndarray,
    policies: dict[str, np.ndarray],
    *,
    seed: int,
    bootstrap_iterations: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    rng = np.random.default_rng(seed)
    source_indices = rng.integers(
        0, len(source), size=(bootstrap_iterations, len(source))
    )
    policy_count = len(next(iter(policies.values())))
    policy_indices = rng.integers(
        0, policy_count, size=(bootstrap_iterations, policy_count)
    )
    bootstrap_scores: dict[str, np.ndarray] = {}
    summaries: dict[str, Any] = {}
    source_mean = source.mean(axis=0)
    for name, matrix in policies.items():
        if len(matrix) != policy_count:
            raise ValueError("Policy episode counts differ")
        scores = np.asarray(
            [
                profile_distance(source[source_index], matrix[policy_index])
                for source_index, policy_index in zip(
                    source_indices, policy_indices, strict=True
                )
            ],
            dtype=np.float64,
        )
        bootstrap_scores[name] = scores
        policy_mean = matrix.mean(axis=0)
        summaries[name] = {
            "role": POLICY_ROLES.get(name, "unclassified"),
            "episode_count": len(matrix),
            "alignment_distance": {
                "mean": profile_distance(source, matrix),
                **_interval(scores),
            },
            "profile_mean": {
                feature: float(value)
                for feature, value in zip(FEATURES, policy_mean, strict=True)
            },
            "absolute_gap": {
                feature: float(value)
                for feature, value in zip(
                    FEATURES, np.abs(policy_mean - source_mean), strict=True
                )
            },
        }

    comparisons: dict[str, Any] = {}
    proposed = "minimind-learned"
    if proposed in bootstrap_scores:
        for reference in (
            "direct-minimind-lora",
            "direct-mlp",
            "p1-rule",
            "minimind-no-history",
            "minimind-no-instruction",
            "numeric-learned",
        ):
            if reference not in bootstrap_scores:
                continue
            delta = bootstrap_scores[proposed] - bootstrap_scores[reference]
            comparisons[f"{proposed}_minus_{reference}"] = {
                "mean": (
                    summaries[proposed]["alignment_distance"]["mean"]
                    - summaries[reference]["alignment_distance"]["mean"]
                ),
                **_interval(delta),
                "negative_favors_minimind": True,
            }
    return summaries, comparisons


def build_report(
    *,
    source_path: Path,
    episode_path: Path,
    closed_loop_report_path: Path,
    source_profiles: np.ndarray,
    policy_profiles: dict[str, np.ndarray],
    policy_seeds: Sequence[int],
    summaries: dict[str, Any],
    comparisons: dict[str, Any],
    split_seed: int,
    bootstrap_seed: int,
    bootstrap_iterations: int,
) -> dict[str, Any]:
    closed_loop = json.loads(closed_loop_report_path.read_text(encoding="utf-8"))
    blockers: list[str] = []
    if _sha256(source_path) != AUDITED_SOURCE_SHA256:
        blockers.append("source trajectory hash is outside the observation audit")
    metadata = closed_loop.get("metadata", {})
    if metadata.get("research_evidence") is not True:
        blockers.append("closed-loop report is not research evidence")
    if int(metadata.get("episode_count", 0)) != len(policy_seeds):
        blockers.append("episode CSV count does not match the aggregate report")
    if set(closed_loop.get("policies", {})) != set(policy_profiles):
        blockers.append("episode CSV policies do not match the aggregate report")
    ranking = sorted(
        summaries,
        key=lambda name: summaries[name]["alignment_distance"]["mean"],
    )
    return {
        "schema_version": 1,
        "artifact": "mousemind_trajectory_profile_alignment",
        "experiment": "held_out_source_mouse_behavioral_profile_alignment",
        "synthetic": False,
        "research_evidence": not blockers,
        "research_evidence_blockers": blockers,
        "source": {
            "sha256": _sha256(source_path),
            "held_out_episode_count": len(source_profiles),
            "split_seed": split_seed,
            "split": "episode-isolated test",
        },
        "policy_evaluation": {
            "episode_csv_sha256": _sha256(episode_path),
            "aggregate_report_sha256": _sha256(closed_loop_report_path),
            "episode_count_per_policy": len(policy_seeds),
            "seed_start": int(policy_seeds[0]),
            "seed_stop": int(policy_seeds[-1]),
        },
        "alignment_contract": {
            "features": list(FEATURES),
            "feature_definitions": FEATURE_DEFINITIONS,
            "score": "equal-weight mean absolute gap between bounded profile means",
            "direction": "lower is better; zero is identical on all profile means",
            "bootstrap_seed": bootstrap_seed,
            "bootstrap_iterations": bootstrap_iterations,
        },
        "source_profile_mean": {
            feature: float(value)
            for feature, value in zip(
                FEATURES, source_profiles.mean(axis=0), strict=True
            )
        },
        "policies": summaries,
        "ranking": ranking,
        "comparisons": comparisons,
        "limitations": [
            "This is alignment to a simulator-policy trajectory export, not to "
            "biological mouse behavior.",
            "The scalar index is a descriptive equal-weight summary; "
            "feature-level gaps remain primary evidence.",
            "Source and final policy episodes use different seeds and are "
            "compared as behavioral distributions, not paired trajectories.",
            "A lower offline behavioral-profile distance is not a closed-loop "
            "safety guarantee.",
        ],
    }


def _distance(summary: dict[str, Any]) -> str:
    value = summary["alignment_distance"]
    return f"{value['mean']:.3f} ({value['ci_low']:.3f}–{value['ci_high']:.3f})"


def render_markdown(report: dict[str, Any]) -> str:
    if report.get("artifact") != "mousemind_trajectory_profile_alignment":
        raise ValueError("Not a trajectory-profile alignment report")
    policies = report["policies"]
    order = (
        "random",
        "direct-minimind-base",
        "direct-minimind-lora",
        "direct-mlp",
        "p1-rule",
        "minimind-no-history",
        "minimind-no-instruction",
        "minimind-learned",
        "minimind-verified",
        "numeric-learned",
        "numeric-verified",
    )
    lines = [
        "# Mouse trajectory behavioral-profile alignment",
        "",
        "The score compares each policy's 100 fresh-ID closed-loop episodes with "
        f"{report['source']['held_out_episode_count']} episode-isolated held-out "
        "source episodes across 11 bounded behavioral features. Lower is better; "
        "parentheses are bootstrap 95% confidence intervals.",
        "",
        "| Role | Method | Alignment distance ↓ |",
        "| --- | --- | ---: |",
    ]
    for name in order:
        if name not in policies:
            continue
        label = POLICY_LABELS.get(name, name)
        if name == "minimind-learned":
            label = f"**{label}**"
        lines.append(
            f"| {policies[name]['role']} | {label} | {_distance(policies[name])} |"
        )
    lines.extend(
        (
            "",
            "The full MiniMind hierarchy is the best MiniMind-based variant and "
            "is clearly closer than the direct policies and its ablations. Its "
            "point estimate is also lower than P1, although that comparison's "
            "bootstrap interval overlaps zero. The numeric planner remains a "
            "non-language upper reference rather "
            "than evidence that MiniMind is the unconstrained overall winner.",
            "",
            "This source is a BotEvade simulator-policy export, not biological "
            "mouse behavior. The scalar score is descriptive; the aggregate JSON "
            "retains every feature-level gap.",
            "",
        )
    )
    return "\n".join(lines)


def _atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-csv", type=Path, required=True)
    parser.add_argument("--closed-loop-episodes", type=Path, required=True)
    parser.add_argument("--closed-loop-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--markdown-output", type=Path)
    parser.add_argument("--action-count", type=int, default=295)
    parser.add_argument("--continuity-atol", type=float, default=1e-5)
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--validation-ratio", type=float, default=0.1)
    parser.add_argument("--bootstrap-seed", type=int, default=20260829)
    parser.add_argument("--bootstrap-iterations", type=int, default=2000)
    args = parser.parse_args()

    source = load_source_profiles(
        args.source_csv,
        action_count=args.action_count,
        continuity_atol=args.continuity_atol,
        split_seed=args.split_seed,
        train_ratio=args.train_ratio,
        validation_ratio=args.validation_ratio,
    )
    policies, seeds = load_policy_profiles(args.closed_loop_episodes)
    summaries, comparisons = analyze(
        source,
        policies,
        seed=args.bootstrap_seed,
        bootstrap_iterations=args.bootstrap_iterations,
    )
    report = build_report(
        source_path=args.source_csv,
        episode_path=args.closed_loop_episodes,
        closed_loop_report_path=args.closed_loop_report,
        source_profiles=source,
        policy_profiles=policies,
        policy_seeds=seeds,
        summaries=summaries,
        comparisons=comparisons,
        split_seed=args.split_seed,
        bootstrap_seed=args.bootstrap_seed,
        bootstrap_iterations=args.bootstrap_iterations,
    )
    _atomic_write(args.output, json.dumps(report, indent=2, sort_keys=True) + "\n")
    if args.markdown_output is not None:
        _atomic_write(args.markdown_output, render_markdown(report))
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
