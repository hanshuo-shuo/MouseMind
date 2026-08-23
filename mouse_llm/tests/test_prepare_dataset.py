from __future__ import annotations

import argparse
import csv
import json

from mouse_llm.data.prepare_dataset import (
    Episode,
    Transition,
    prepare,
    remove_cross_split_prompt_duplicates,
)


def _vector(value: float) -> str:
    return json.dumps([value] * 10)


def test_prepare_detects_implicit_episode_boundary(tmp_path):
    source = tmp_path / "mouse.csv"
    with source.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=["obs", "action", "reward", "next_obs", "done"]
        )
        writer.writeheader()
        writer.writerows(
            [
                {"obs": _vector(0), "action": 0, "reward": 0, "next_obs": _vector(1), "done": 0},
                {"obs": _vector(1), "action": 1, "reward": 10, "next_obs": _vector(2), "done": 1},
                {"obs": _vector(5), "action": 2, "reward": 0, "next_obs": _vector(6), "done": 0},
                {"obs": _vector(9), "action": 3, "reward": 0, "next_obs": _vector(10), "done": 0},
            ]
        )
    output = tmp_path / "processed"
    args = argparse.Namespace(
        input=source,
        output_dir=output,
        action_count=295,
        seed=42,
        train_ratio=0.8,
        validation_ratio=0.1,
        continuity_atol=1e-5,
        precision=4,
        max_train_samples=0,
        max_validation_samples=0,
        max_test_samples=0,
        allow_repo_output=False,
    )
    manifest = prepare(args)
    assert manifest["validation"] == {
        "explicit_done_boundaries": 1,
        "implicit_discontinuity_boundaries": 1,
        "end_of_file_boundaries": 1,
        "episodes": 3,
    }
    assert sum(manifest["split"]["sample_counts"].values()) == 4
    assert (output / "manifest.json").is_file()
    assert sum(1 for path in output.glob("*.jsonl") for _ in path.open()) == 4


def test_exact_prompt_overlap_is_removed_from_evaluation_splits():
    shared = tuple([0.25] * 10)
    unique = tuple([0.75] * 10)

    def item(episode_id, row, observation):
        transition = Transition(row, observation, 1, 0.0, observation, True)
        episode = Episode(episode_id, (transition,), "done")
        return episode, transition

    splits = {
        "train": [item(0, 2, shared)],
        "validation": [item(1, 3, shared), item(2, 4, unique)],
        "test": [item(3, 5, shared), item(4, 6, unique)],
    }
    removed = remove_cross_split_prompt_duplicates(splits, precision=4)
    assert removed == {"train": 0, "validation": 1, "test": 2}
    assert len(splits["train"]) == 1
    assert len(splits["validation"]) == 1
    assert splits["test"] == []
