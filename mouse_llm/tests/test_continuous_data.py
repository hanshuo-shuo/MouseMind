from __future__ import annotations

import csv
import json
import math

import pytest

from mouse_llm.continuous.contracts import ACTION_SCHEMA, DATA_SCHEMA
from mouse_llm.continuous.data import build_parser, deduplicate_inputs, label_difference_audit, prepare
from mouse_llm.data.prepare_dataset import Episode, Transition


def observation(x, y=0.4):
    return [x, y, -0.4, 0.0, 0.0, 0.0, 0.8, 0.0, 1.0, 0.0]


def source_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["obs", "action", "reward", "next_obs", "done"])
        for obs, action, reward, next_obs, done in rows:
            writer.writerow([json.dumps(obs), action, reward, json.dumps(next_obs), int(done)])


def catalog(path, destinations=((0.9, 0.8), (0.8, 0.9))):
    path.write_text(json.dumps({"action_count": len(destinations), "actions": [{"action": i, "destination": xy} for i, xy in enumerate(destinations)]}))


def args_for(source, output, catalog_path, *extra):
    return build_parser().parse_args(["--input", str(source), "--output-dir", str(output), "--action-catalog", str(catalog_path), *extra])


def read_splits(output):
    return {split: [json.loads(line) for line in (output / f"{split}.jsonl").read_text().splitlines()] for split in ("train", "validation", "test")}


def test_achieved_waypoint_and_commanded_destination_remain_distinct(tmp_path):
    source, action_catalog = tmp_path / "source.csv", tmp_path / "catalog.json"
    catalog(action_catalog)
    obs, next_obs = observation(0.2), observation(0.21)
    source_csv(source, [(obs, 0, 0.5, next_obs, True)])
    records = {}
    for mode in ("next_position", "commanded_destination"):
        output = tmp_path / mode
        manifest = prepare(args_for(source, output, action_catalog, "--label-mode", mode))
        row = next(row for rows in read_splits(output).values() for row in rows)
        assert row["schema_version"] == DATA_SCHEMA
        assert row["action_schema"] == ACTION_SCHEMA
        assert row["label_mode"] == mode
        assert row["observation"] == obs
        assert row["next_observation"] == next_obs
        assert manifest["policy_input_fields"] == ["observation"]
        assert manifest["label_difference_audit"]["different_beyond_tolerance"] == 1
        assert manifest["label_difference_audit"]["euclidean_xy_distance"]["mean"] == pytest.approx(math.dist(next_obs[:2], (0.9, 0.8)))
        records[mode] = row
    assert records["next_position"]["action_xy"] == next_obs[:2]
    assert records["commanded_destination"]["action_xy"] == [0.9, 0.8]


def test_whole_episodes_stay_in_one_split_and_output_is_reproducible(tmp_path):
    source, action_catalog = tmp_path / "source.csv", tmp_path / "catalog.json"
    catalog(action_catalog)
    rows = []
    for episode in range(80):
        start = 0.1 + episode * 0.009
        states = [observation(start + index * 0.001) for index in range(4)]
        rows.extend((states[index], index % 2, 0.0, states[index + 1], index == 2) for index in range(3))
    source_csv(source, rows)
    manifests = []
    for name in ("first", "second"):
        output = tmp_path / name
        manifests.append(prepare(args_for(source, output, action_catalog)))
        owners = {}
        splits = read_splits(output)
        assert all(splits.values())
        for split, samples in splits.items():
            for row in samples:
                assert owners.setdefault(row["episode_id"], split) == split
        assert len(owners) == 80
        assert sum(len(samples) for samples in splits.values()) == 240
    assert manifests[0]["artifacts"] == manifests[1]["artifacts"]
    assert manifests[0]["split"] == manifests[1]["split"]


def test_deduplication_never_uses_future_target_reward_or_done():
    current = tuple(observation(0.2))

    def item(episode_id, next_x, reward, done):
        transition = Transition(episode_id + 2, current, episode_id % 2, reward, tuple(observation(next_x)), done)
        return Episode(episode_id, (transition,), "done"), transition

    splits = {"train": [item(0, 0.3, 0.0, False), item(1, 0.4, 1.0, True)], "validation": [item(2, 0.5, -1.0, True)], "test": [item(3, 0.6, 2.0, False)]}
    removed = deduplicate_inputs(splits, precision=4)
    assert len(splits["train"]) == 2
    assert splits["train"][0][1].next_observation != splits["train"][1][1].next_observation
    assert splits["validation"] == splits["test"] == []
    assert removed["train"]["within_split"] == 0
    assert removed["validation"]["cross_split"] == removed["test"]["cross_split"] == 1


def test_displacement_audit_identifies_waypoints_inside_navigation_threshold():
    rows = [
        Transition(index + 2, tuple(observation(0.2)), 0, 0.0,
                   tuple(observation(next_x)), True)
        for index, next_x in enumerate((0.2, 0.205, 0.22))
    ]
    audit = label_difference_audit(rows, ((0.9, 0.8),))
    displacement = audit["current_to_next_displacement"]
    assert displacement["mean"] == pytest.approx(0.025 / 3)
    assert displacement["median"] == pytest.approx(0.005)
    assert displacement["p95"] == pytest.approx(0.0185)
    assert displacement["fraction_le_0_01"] == pytest.approx(2 / 3)
    assert displacement["zeroish_fraction"] == pytest.approx(1 / 3)
    assert "not the original commanded destination" in audit["interpretation"]


@pytest.mark.parametrize("invalid_xy", [(-0.01, 0.5), (0.5, 1.01), (float("nan"), 0.5)])
def test_invalid_target_bounds_fail_before_writing(tmp_path, invalid_xy):
    source, action_catalog = tmp_path / "source.csv", tmp_path / "catalog.json"
    catalog(action_catalog)
    next_obs = observation(*invalid_xy)
    source_csv(source, [(observation(0.2), 0, 0.0, next_obs, True)])
    output = tmp_path / "processed"
    with pytest.raises(ValueError):
        prepare(args_for(source, output, action_catalog))
    assert not output.exists()


def test_invalid_command_catalog_is_rejected_even_for_next_position_mode(tmp_path):
    source, action_catalog = tmp_path / "source.csv", tmp_path / "catalog.json"
    catalog(action_catalog, ((1.2, 0.8),))
    source_csv(source, [(observation(0.2), 0, 0.0, observation(0.3), True)])
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        prepare(args_for(source, tmp_path / "processed", action_catalog))


def test_git_output_requires_explicit_override_and_files_are_private(tmp_path):
    source, action_catalog = tmp_path / "source.csv", tmp_path / "catalog.json"
    catalog(action_catalog)
    source_csv(source, [(observation(0.2), 0, 0.0, observation(0.3), True)])
    repo = tmp_path / "repository"
    repo.mkdir()
    (repo / ".git").write_text("gitdir: /some/worktree/metadata")
    output = repo / "private"
    with pytest.raises(ValueError, match="Git repository"):
        prepare(args_for(source, output, action_catalog))
    manifest = prepare(args_for(source, output, action_catalog, "--allow-repo-output"))
    assert output.stat().st_mode & 0o777 == 0o700
    assert all(path.stat().st_mode & 0o777 == 0o600 for path in output.iterdir())
    assert "--allow-repo-output" in manifest["command"]
    assert len(manifest["source"]["sha256"]) == 64


def test_sample_limits_do_not_change_episode_ownership(tmp_path):
    source, action_catalog = tmp_path / "source.csv", tmp_path / "catalog.json"
    catalog(action_catalog)
    source_csv(source, [(observation(0.1 + index * 0.01), 0, 0.0, observation(0.105 + index * 0.01), True) for index in range(70)])
    full = prepare(args_for(source, tmp_path / "full", action_catalog))
    limited = prepare(args_for(source, tmp_path / "limited", action_catalog, "--max-train-samples", "3"))
    assert limited["split"]["sample_counts"]["train"] == 3
    assert limited["split"]["episode_ids"] == full["split"]["episode_ids"]
    for split in ("validation", "test"):
        assert limited["artifacts"][split] == full["artifacts"][split]
