"""Validate transitions and create episode-isolated MiniMind SFT splits."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from .schema import (
    FEATURE_NAMES,
    SCHEMA_NAME,
    SEMANTIC_FEATURE_NAMES,
    make_conversation,
    user_prompt,
)


REQUIRED_COLUMNS = ("obs", "action", "reward", "next_obs", "done")
AUDITED_SOURCE_SHA256 = "fdb93c5a874bf2587d9f0be5b3d1dd0739224eeb78bbacfbb410c5e912de39ea"


@dataclass(frozen=True)
class Transition:
    row_number: int
    observation: tuple[float, ...]
    action: int
    reward: float
    next_observation: tuple[float, ...]
    done: bool


@dataclass(frozen=True)
class Episode:
    episode_id: int
    transitions: tuple[Transition, ...]
    boundary_reason: str


def _parse_vector(raw: str, *, row_number: int, column: str) -> tuple[float, ...]:
    try:
        values = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Row {row_number}: invalid JSON in {column}") from exc
    if not isinstance(values, list) or len(values) != len(FEATURE_NAMES):
        raise ValueError(
            f"Row {row_number}: {column} must contain {len(FEATURE_NAMES)} values"
        )
    parsed = tuple(float(value) for value in values)
    if not all(math.isfinite(value) for value in parsed):
        raise ValueError(f"Row {row_number}: {column} contains non-finite values")
    return parsed


def _parse_done(raw: str, *, row_number: int) -> bool:
    value = float(raw)
    if value not in (0.0, 1.0):
        raise ValueError(f"Row {row_number}: done must be 0 or 1, got {raw!r}")
    return bool(value)


def load_transitions(path: Path, *, action_count: int) -> list[Transition]:
    transitions: list[Transition] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != REQUIRED_COLUMNS:
            raise ValueError(
                f"Expected columns {REQUIRED_COLUMNS}, got {tuple(reader.fieldnames or ())}"
            )
        for row_number, row in enumerate(reader, start=2):
            action_float = float(row["action"])
            action = int(action_float)
            if action_float != action or not 0 <= action < action_count:
                raise ValueError(
                    f"Row {row_number}: action must be an integer in "
                    f"[0, {action_count - 1}]"
                )
            reward = float(row["reward"])
            if not math.isfinite(reward):
                raise ValueError(f"Row {row_number}: reward is non-finite")
            transitions.append(
                Transition(
                    row_number=row_number,
                    observation=_parse_vector(
                        row["obs"], row_number=row_number, column="obs"
                    ),
                    action=action,
                    reward=reward,
                    next_observation=_parse_vector(
                        row["next_obs"], row_number=row_number, column="next_obs"
                    ),
                    done=_parse_done(row["done"], row_number=row_number),
                )
            )
    if not transitions:
        raise ValueError("The transition CSV is empty")
    return transitions


def observations_match(
    left: tuple[float, ...], right: tuple[float, ...], *, atol: float
) -> bool:
    return all(abs(a - b) <= atol for a, b in zip(left, right, strict=True))


def segment_episodes(
    transitions: list[Transition], *, continuity_atol: float
) -> list[Episode]:
    episodes: list[Episode] = []
    current: list[Transition] = []
    for index, transition in enumerate(transitions):
        current.append(transition)
        if transition.done:
            boundary_reason = "done"
        elif index == len(transitions) - 1:
            boundary_reason = "end_of_file"
        elif not observations_match(
            transition.next_observation,
            transitions[index + 1].observation,
            atol=continuity_atol,
        ):
            boundary_reason = "discontinuity"
        else:
            continue
        episodes.append(
            Episode(
                episode_id=len(episodes),
                transitions=tuple(current),
                boundary_reason=boundary_reason,
            )
        )
        current = []
    return episodes


def split_for_episode(
    episode: Episode, *, seed: int, train_ratio: float, validation_ratio: float
) -> str:
    first = episode.transitions[0]
    last = episode.transitions[-1]
    identity = (
        f"{seed}:{episode.episode_id}:{first.row_number}:{last.row_number}:"
        f"{len(episode.transitions)}"
    ).encode("utf-8")
    bucket = int.from_bytes(hashlib.sha256(identity).digest()[:8], "big") / 2**64
    if bucket < train_ratio:
        return "train"
    if bucket < train_ratio + validation_ratio:
        return "validation"
    return "test"


def _stable_sample(
    items: list[tuple[Episode, Transition]], *, limit: int, seed: int, split: str
) -> list[tuple[Episode, Transition]]:
    if limit <= 0 or len(items) <= limit:
        return items

    def score(item: tuple[Episode, Transition]) -> bytes:
        episode, transition = item
        token = f"{seed}:{split}:{episode.episode_id}:{transition.row_number}"
        return hashlib.sha256(token.encode("utf-8")).digest()

    selected = sorted(items, key=score)[:limit]
    return sorted(selected, key=lambda item: item[1].row_number)


def remove_cross_split_prompt_duplicates(
    split_items: dict[str, list[tuple[Episode, Transition]]], *, precision: int
) -> dict[str, int]:
    """Drop exact serialized-state overlap from lower-priority splits.

    Episode assignment happens first and is never changed. This filter only
    removes evaluation transitions whose rounded prompt is already owned by an
    earlier split, preventing exact prompt leakage after serialization.
    """
    prior_split_prompts: set[str] = set()
    removed: dict[str, int] = {}
    for split in ("train", "validation", "test"):
        kept: list[tuple[Episode, Transition]] = []
        current_prompts: set[str] = set()
        removed_count = 0
        for item in split_items[split]:
            prompt = user_prompt(item[1].observation, precision=precision)
            if prompt in prior_split_prompts:
                removed_count += 1
                continue
            kept.append(item)
            current_prompts.add(prompt)
        split_items[split] = kept
        prior_split_prompts.update(current_prompts)
        removed[split] = removed_count
    return removed


def _atomic_json(path: Path, payload: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _atomic_jsonl(path: Path, rows: Iterable[object]) -> int:
    temporary = path.with_suffix(path.suffix + ".tmp")
    count = 0
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")))
            handle.write("\n")
            count += 1
    os.replace(temporary, path)
    return count


def _git_root_containing(path: Path) -> Path | None:
    resolved = path.resolve()
    for candidate in (resolved, *resolved.parents):
        if (candidate / ".git").exists():
            return candidate
    return None


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def prepare(args: argparse.Namespace) -> dict[str, object]:
    input_path = args.input.resolve()
    output_dir = args.output_dir.resolve()
    if not input_path.is_file():
        raise FileNotFoundError(input_path)
    if not args.allow_repo_output:
        git_root = _git_root_containing(output_dir)
        if git_root is not None:
            raise ValueError(
                f"Refusing to write private processed data inside Git repository "
                f"{git_root}. Use a shared/private output directory."
            )
    if args.train_ratio <= 0 or args.validation_ratio < 0:
        raise ValueError("Split ratios must be non-negative and train_ratio > 0")
    if args.train_ratio + args.validation_ratio >= 1:
        raise ValueError("train_ratio + validation_ratio must be less than 1")

    transitions = load_transitions(input_path, action_count=args.action_count)
    episodes = segment_episodes(
        transitions, continuity_atol=args.continuity_atol
    )
    split_items: dict[str, list[tuple[Episode, Transition]]] = {
        "train": [],
        "validation": [],
        "test": [],
    }
    episode_splits: Counter[str] = Counter()
    boundary_reasons = Counter(episode.boundary_reason for episode in episodes)
    for episode in episodes:
        split = split_for_episode(
            episode,
            seed=args.seed,
            train_ratio=args.train_ratio,
            validation_ratio=args.validation_ratio,
        )
        episode_splits[split] += 1
        split_items[split].extend((episode, item) for item in episode.transitions)

    removed_prompt_duplicates = remove_cross_split_prompt_duplicates(
        split_items, precision=args.precision
    )

    split_items["train"] = _stable_sample(
        split_items["train"],
        limit=args.max_train_samples,
        seed=args.seed,
        split="train",
    )
    split_items["validation"] = _stable_sample(
        split_items["validation"],
        limit=args.max_validation_samples,
        seed=args.seed,
        split="validation",
    )
    split_items["test"] = _stable_sample(
        split_items["test"],
        limit=args.max_test_samples,
        seed=args.seed,
        split="test",
    )

    output_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
    output_dir.chmod(0o700)
    sample_counts: dict[str, int] = {}
    for split, items in split_items.items():
        path = output_dir / f"{split}.jsonl"
        sample_counts[split] = _atomic_jsonl(
            path,
            (
                make_conversation(
                    transition.observation,
                    transition.action,
                    action_count=args.action_count,
                    precision=args.precision,
                )
                for _, transition in items
            ),
        )
        path.chmod(0o600)

    action_counts = Counter(transition.action for transition in transitions)
    reward_counts = Counter(transition.reward for transition in transitions)
    source_sha256 = _sha256(input_path)
    observation_contract_verified = source_sha256 == AUDITED_SOURCE_SHA256
    manifest: dict[str, object] = {
        "schema_version": 1,
        "synthetic": False,
        "research_evidence": observation_contract_verified,
        "research_evidence_blockers": (
            []
            if observation_contract_verified
            else ["input hash is not covered by the checked observation audit"]
        ),
        "prompt_schema": SCHEMA_NAME,
        "feature_names": list(FEATURE_NAMES),
        "semantic_feature_names": list(SEMANTIC_FEATURE_NAMES),
        "observation_contract": {
            "source_commit": "67e769fd410b325b5d2c517d9d5966e5e80fac23",
            "audit": "mouse_llm/reports/observation_contract_audit.json",
            "verified": observation_contract_verified,
        },
        "source": {
            "filename": input_path.name,
            "sha256": source_sha256,
            "rows": len(transitions),
        },
        "validation": {
            "explicit_done_boundaries": boundary_reasons["done"],
            "implicit_discontinuity_boundaries": boundary_reasons["discontinuity"],
            "end_of_file_boundaries": boundary_reasons["end_of_file"],
            "episodes": len(episodes),
        },
        "split": {
            "seed": args.seed,
            "train_ratio": args.train_ratio,
            "validation_ratio": args.validation_ratio,
            "test_ratio": 1 - args.train_ratio - args.validation_ratio,
            "episode_counts": dict(episode_splits),
            "sample_counts": sample_counts,
            "removed_cross_split_prompt_duplicates": removed_prompt_duplicates,
            "prompt_ownership_priority": ["train", "validation", "test"],
        },
        "action_count": args.action_count,
        "action_histogram": {str(k): v for k, v in sorted(action_counts.items())},
        "reward_histogram": {str(k): v for k, v in sorted(reward_counts.items())},
        "continuity_atol": args.continuity_atol,
        "precision": args.precision,
    }
    manifest_path = output_dir / "manifest.json"
    _atomic_json(manifest_path, manifest)
    manifest_path.chmod(0o600)
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build episode-isolated MiniMind SFT data from mouse transitions"
    )
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--action-count", type=int, default=295)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--validation-ratio", type=float, default=0.1)
    parser.add_argument("--continuity-atol", type=float, default=1e-5)
    parser.add_argument("--precision", type=int, default=4)
    parser.add_argument("--max-train-samples", type=int, default=0)
    parser.add_argument("--max-validation-samples", type=int, default=0)
    parser.add_argument("--max-test-samples", type=int, default=0)
    parser.add_argument(
        "--allow-repo-output",
        action="store_true",
        help="Override the safety check that keeps processed data out of Git",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    manifest = prepare(args)
    print(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
