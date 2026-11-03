"""Prepare private, episode-isolated supervision for continuous XY control.

``next_position`` is an achieved next-position waypoint, not a reconstruction
of the original commanded destination. Only ``observation`` is a policy input;
next observations and rewards remain supervision/audit fields.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from mouse_llm.continuous.contracts import ACTION_SCHEMA, DATA_SCHEMA, LABEL_MODES, validate_xy
from mouse_llm.data.prepare_dataset import (
    AUDITED_SOURCE_SHA256,
    Episode,
    Transition,
    _git_root_containing,
    _sha256,
    _stable_sample,
    load_transitions,
    segment_episodes,
    split_for_episode,
)
from mouse_llm.data.schema import SCHEMA_NAME, SEMANTIC_FEATURE_NAMES, user_prompt


DEFAULT_ACTION_CATALOG = (
    Path(__file__).resolve().parents[1] / "envs/mice/assets/action_catalog_21_05.json"
)
SPLITS = ("train", "validation", "test")
LABEL_DESCRIPTIONS = {
    "next_position": (
        "Achieved next-position waypoint: next_observation[:2]. This is not "
        "the original commanded destination and does not identify motor intent."
    ),
    "commanded_destination": (
        "Original discrete command mapped to its absolute XY destination by "
        "the supplied action catalog; no next-state information enters the label."
    ),
}


def load_action_catalog(path: Path) -> tuple[tuple[float, float], ...]:
    """Read the project's indexed catalog without importing model dependencies."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or not isinstance(payload.get("actions"), list):
        raise ValueError("Action catalog must contain an indexed actions list")
    rows = payload["actions"]
    count = payload.get("action_count", len(rows))
    if isinstance(count, bool) or not isinstance(count, int) or count != len(rows) or count <= 0:
        raise ValueError("Action catalog count must match its nonempty actions list")
    destinations: dict[int, tuple[float, float]] = {}
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError("Each action catalog entry must be an object")
        action = row.get("action")
        if isinstance(action, bool) or not isinstance(action, int) or not 0 <= action < count:
            raise ValueError("Action catalog IDs must be integers in its declared range")
        if action in destinations:
            raise ValueError("Action catalog contains a duplicate action ID")
        destinations[action] = validate_xy(row.get("destination"))
    if set(destinations) != set(range(count)):
        raise ValueError("Action catalog IDs must cover a contiguous range starting at zero")
    return tuple(destinations[index] for index in range(count))


def transition_record(
    episode: Episode,
    transition: Transition,
    *,
    label_mode: str,
    destinations: Sequence[Sequence[float]],
) -> dict[str, Any]:
    """Serialize inputs separately from targets; never append future state to input."""
    if label_mode not in LABEL_MODES:
        raise ValueError(f"Unknown continuous label mode: {label_mode}")
    validate_xy(transition.observation[:2])
    achieved = validate_xy(transition.next_observation[:2])
    commanded = validate_xy(destinations[transition.action])
    target = achieved if label_mode == "next_position" else commanded
    return {
        "schema_version": DATA_SCHEMA,
        "action_schema": ACTION_SCHEMA,
        "label_mode": label_mode,
        "observation": list(transition.observation),
        "action_xy": list(target),
        "next_observation": list(transition.next_observation),
        "reward": transition.reward,
        "done": transition.done,
        "episode_id": episode.episode_id,
        "row_number": transition.row_number,
    }


def deduplicate_inputs(
    split_items: dict[str, list[tuple[Episode, Transition]]], *, precision: int
) -> dict[str, dict[str, int]]:
    """Remove cross-split input overlap without discarding multimodal targets.

    Episode ownership is fixed before filtering. Train, validation, then test
    own shared inputs in that order. All observations within the owning split
    remain, including repeated inputs with different achieved/commanded XY.
    Rounded keys are only a conservative leakage check; written observations
    retain their original numeric precision.
    """
    owners: dict[str, str] = {}
    removed: dict[str, dict[str, int]] = {}
    for split in SPLITS:
        counts = {"within_split": 0, "cross_split": 0}
        kept = []
        for item in split_items[split]:
            key = user_prompt(item[1].observation, precision=precision)
            owner = owners.get(key)
            if owner is not None and owner != split:
                counts["cross_split"] += 1
                continue
            owners[key] = split
            kept.append(item)
        split_items[split] = kept
        removed[split] = counts
    return removed


def _quantile(sorted_values: Sequence[float], fraction: float) -> float:
    position = (len(sorted_values) - 1) * fraction
    low, high = math.floor(position), math.ceil(position)
    return sorted_values[low] + (sorted_values[high] - sorted_values[low]) * (position - low)


def label_difference_audit(
    transitions: Sequence[Transition], destinations: Sequence[Sequence[float]]
) -> dict[str, Any]:
    """Aggregate achieved-versus-commanded differences without revealing rows."""
    if not transitions:
        raise ValueError("A label-difference audit requires at least one transition")
    distances, displacements = [], []
    for transition in transitions:
        current = validate_xy(transition.observation[:2])
        achieved = validate_xy(transition.next_observation[:2])
        commanded = validate_xy(destinations[transition.action])
        distances.append(math.dist(achieved, commanded))
        displacements.append(math.dist(current, achieved))
    distances.sort()
    displacements.sort()
    tolerance = 1e-6
    equal = sum(distance <= tolerance for distance in distances)
    return {
        "comparison": "achieved_next_position versus catalog_commanded_destination",
        "scope": "all validated source transitions before splitting, deduplication, or sampling",
        "rows": len(distances),
        "equal_within_tolerance": equal,
        "different_beyond_tolerance": len(distances) - equal,
        "equal_fraction": equal / len(distances),
        "xy_distance_tolerance": tolerance,
        "euclidean_xy_distance": {
            "mean": math.fsum(distances) / len(distances),
            "median": _quantile(distances, 0.5),
            "p95": _quantile(distances, 0.95),
            "max": distances[-1],
        },
        "current_to_next_displacement": {
            "mean": math.fsum(displacements) / len(displacements),
            "median": _quantile(displacements, 0.5),
            "p95": _quantile(displacements, 0.95),
            "fraction_le_0_01": sum(value <= 0.01 for value in displacements) / len(displacements),
            "zeroish_fraction": sum(value <= tolerance for value in displacements) / len(displacements),
            "zeroish_tolerance": tolerance,
            "scope": "Observed current-to-next XY displacement; values <= 0.01 may fall inside the native navigation arrival threshold when reused as target waypoints",
        },
        "interpretation": LABEL_DESCRIPTIONS["next_position"],
    }


def _private_atomic_write(path: Path, rows: Iterable[Mapping[str, Any]], *, jsonl: bool) -> int:
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    count = 0
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, sort_keys=True, allow_nan=False, indent=None if jsonl else 2))
                handle.write("\n")
                count += 1
        os.replace(temporary, path)
        path.chmod(0o600)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    return count


def _settings(args: argparse.Namespace) -> dict[str, Any]:
    defaults = {
        "label_mode": "next_position", "action_catalog": DEFAULT_ACTION_CATALOG,
        "seed": 42, "train_ratio": 0.8, "validation_ratio": 0.1,
        "continuity_atol": 1e-5, "precision": 4,
        "max_train_samples": 0, "max_validation_samples": 0, "max_test_samples": 0,
        "allow_repo_output": False,
    }
    return {key: getattr(args, key, value) for key, value in defaults.items()}


def prepare(args: argparse.Namespace) -> dict[str, Any]:
    """Prepare three private JSONL files and return an aggregate-only manifest."""
    settings = _settings(args)
    input_path, output_dir = Path(args.input).resolve(), Path(args.output_dir).resolve()
    catalog_path = Path(settings["action_catalog"]).resolve()
    if not input_path.is_file():
        raise FileNotFoundError(input_path)
    if not settings["allow_repo_output"] and _git_root_containing(output_dir) is not None:
        raise ValueError("Refusing to write private processed data inside a Git repository; use --allow-repo-output explicitly to override")
    if settings["label_mode"] not in LABEL_MODES:
        raise ValueError("Unknown continuous label mode")
    train_ratio, validation_ratio = settings["train_ratio"], settings["validation_ratio"]
    if not math.isfinite(train_ratio) or not math.isfinite(validation_ratio) or train_ratio <= 0 or validation_ratio < 0 or train_ratio + validation_ratio >= 1:
        raise ValueError("Require train_ratio > 0, validation_ratio >= 0, and their sum < 1")
    if not math.isfinite(settings["continuity_atol"]) or settings["continuity_atol"] < 0:
        raise ValueError("continuity_atol must be finite and nonnegative")
    if isinstance(settings["precision"], bool) or not isinstance(settings["precision"], int) or not 0 <= settings["precision"] <= 15:
        raise ValueError("precision must be an integer between 0 and 15")
    for split in SPLITS:
        limit = settings[f"max_{split}_samples"]
        if isinstance(limit, bool) or not isinstance(limit, int) or limit < 0:
            raise ValueError("Sample limits must be nonnegative integers; zero means unlimited")
    destinations = load_action_catalog(catalog_path)
    transitions = load_transitions(input_path, action_count=len(destinations))
    audit = label_difference_audit(transitions, destinations)
    episodes = segment_episodes(transitions, continuity_atol=settings["continuity_atol"])
    split_items: dict[str, list[tuple[Episode, Transition]]] = {split: [] for split in SPLITS}
    episode_ids: dict[str, list[int]] = {split: [] for split in SPLITS}
    for episode in episodes:
        split = split_for_episode(episode, seed=settings["seed"], train_ratio=train_ratio, validation_ratio=validation_ratio)
        episode_ids[split].append(episode.episode_id)
        split_items[split].extend((episode, row) for row in episode.transitions)
    before_dedup = {split: len(items) for split, items in split_items.items()}
    removed = deduplicate_inputs(split_items, precision=settings["precision"])
    after_dedup = {split: len(items) for split, items in split_items.items()}
    for split in SPLITS:
        split_items[split] = _stable_sample(split_items[split], limit=settings[f"max_{split}_samples"], seed=settings["seed"], split=split)

    output_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
    output_dir.chmod(0o700)
    sample_counts, artifacts = {}, {}
    for split in SPLITS:
        path = output_dir / f"{split}.jsonl"
        records = (transition_record(episode, row, label_mode=settings["label_mode"], destinations=destinations) for episode, row in split_items[split])
        sample_counts[split] = _private_atomic_write(path, records, jsonl=True)
        artifacts[split] = {"filename": path.name, "sha256": _sha256(path), "rows": sample_counts[split]}

    parameters = {key: str(value) if isinstance(value, Path) else value for key, value in settings.items()}
    parameters.update(input=str(input_path), output_dir=str(output_dir), action_catalog=str(catalog_path))
    command = ["python", "-m", "mouse_llm.continuous.data"]
    for name, value in parameters.items():
        if name == "allow_repo_output":
            if value:
                command.append("--allow-repo-output")
        else:
            command.extend(["--" + name.replace("_", "-"), str(value)])
    boundaries = Counter(episode.boundary_reason for episode in episodes)
    source_hash = _sha256(input_path)
    manifest = {
        "schema_version": 1,
        "data_schema": DATA_SCHEMA,
        "action_schema": ACTION_SCHEMA,
        "label_mode": settings["label_mode"],
        "label_description": LABEL_DESCRIPTIONS[settings["label_mode"]],
        "permitted_use": "behavior_cloning_supervision",
        "offline_rl_replay_validated": False,
        "transition_warning": "reward/next_observation belong to the historical command. A next_position waypoint was not the executed action; do not insert these relabeled rows into RL replay.",
        "policy_input_fields": ["observation"],
        "supervision_only_fields": ["action_xy", "next_observation", "reward", "done"],
        "observation_schema": SCHEMA_NAME,
        "semantic_feature_names": list(SEMANTIC_FEATURE_NAMES),
        "source": {"filename": input_path.name, "sha256": source_hash, "rows": len(transitions), "matches_audited_legacy_source": source_hash == AUDITED_SOURCE_SHA256},
        "action_catalog": {"filename": catalog_path.name, "sha256": _sha256(catalog_path), "action_count": len(destinations)},
        "validation": {
            "episodes": len(episodes), "explicit_done_boundaries": boundaries["done"],
            "implicit_discontinuity_boundaries": boundaries["discontinuity"], "end_of_file_boundaries": boundaries["end_of_file"],
        },
        "split": {
            "seed": settings["seed"], "train_ratio": train_ratio, "validation_ratio": validation_ratio,
            "test_ratio": 1 - train_ratio - validation_ratio,
            "episode_counts": {split: len(ids) for split, ids in episode_ids.items()},
            "episode_ids": episode_ids, "samples_before_deduplication": before_dedup,
            "samples_after_deduplication": after_dedup, "sample_counts": sample_counts,
            "removed_input_duplicates": removed, "input_ownership_priority": list(SPLITS),
            "within_split_policy": "preserve_all_rows_including_different_targets_for_the_same_input",
            "deduplication_input": "rounded current observation only; never label, reward, done, or next observation",
            "deduplication_precision": settings["precision"],
        },
        "label_difference_audit": audit,
        "artifacts": artifacts,
        "parameters": parameters,
        "command": command,
        "research_evidence": False,
        "evidence_scope": "Data preparation and aggregate audit only; not a trained-policy or closed-loop result",
    }
    _private_atomic_write(output_dir / "manifest.json", [manifest], jsonl=False)
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--label-mode", choices=LABEL_MODES, default="next_position")
    parser.add_argument("--action-catalog", type=Path, default=DEFAULT_ACTION_CATALOG)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--validation-ratio", type=float, default=0.1)
    parser.add_argument("--continuity-atol", type=float, default=1e-5)
    parser.add_argument("--precision", type=int, default=4, help="Current-input deduplication precision; output values are not rounded")
    parser.add_argument("--max-train-samples", type=int, default=0)
    parser.add_argument("--max-validation-samples", type=int, default=0)
    parser.add_argument("--max-test-samples", type=int, default=0)
    parser.add_argument("--allow-repo-output", action="store_true", help="Explicitly allow private outputs inside a Git worktree")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    manifest = prepare(args)
    # Only aggregate counts and provenance are printed; never transition rows.
    print(json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
