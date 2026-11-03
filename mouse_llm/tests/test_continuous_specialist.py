"""Continuous BC tests use tiny synthetic records, never research trajectories."""

import json
import math

import pytest

np = pytest.importorskip("numpy")
torch = pytest.importorskip("torch")

from mouse_llm.continuous.contracts import ACTION_SCHEMA, DATA_SCHEMA
from mouse_llm.continuous.specialist import (
    ContinuousCheckpointPolicy,
    ContinuousMixturePolicy,
    ContinuousSplit,
    _private_output,
    checkpoint_payload,
    evaluate_specialist,
    load_split,
    main,
    mixture_nll,
    train_specialist,
    validate_split_isolation,
)


def _row(*, episode="train-1", row_number=0, mode="next_position"):
    return {
        "schema_version": DATA_SCHEMA,
        "action_schema": ACTION_SCHEMA,
        "label_mode": mode,
        "observation": [0.2, 0.3, 0.0, 0.0, 0.0, 0.0, 0.5, 0.0, 1.0, 0.0],
        "next_observation": [0.25, 0.35, 0.0, 0.0, 0.0, 0.0, 0.45, 0.0, 1.0, 0.0],
        "action_xy": [0.25, 0.35],
        "reward": 0.0,
        "done": False,
        "episode_id": episode,
        "row_number": row_number,
    }


def _write(path, rows):
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    return path


def _synthetic_split(episode, *, offset=0.0):
    rng = np.random.default_rng(42)
    observations = rng.uniform(0.1, 0.7, size=(12, 10)).astype(np.float32) + offset
    return ContinuousSplit(
        observations=observations,
        actions=np.tile(np.asarray([[0.2, 0.8]], dtype=np.float32), (12, 1)),
        episode_ids=(episode,) * 12,
        label_mode="commanded_destination",
        source_sha256=episode,
    )


def test_load_split_checks_label_semantics_and_provenance(tmp_path):
    path = _write(tmp_path / "train.jsonl", [_row(), _row(row_number=1)])
    split = load_split(path, expected_label_mode="next_position")
    assert split.observations.shape == (2, 10)
    assert split.actions.shape == (2, 2)
    assert split.provenance()["episodes"] == 1
    assert len(split.source_sha256) == 64
    assert split.provenance()["action_schema"] == ACTION_SCHEMA
    with pytest.raises(ValueError, match="Mixed label"):
        load_split(path, expected_label_mode="commanded_destination")


@pytest.mark.parametrize("field,value,match", [
    ("schema_version", "legacy", "schema"),
    ("action_schema", "discrete_295", "schema"),
    ("label_mode", "inferred", "label_mode"),
    ("observation", [0.0] * 9, "ten"),
    ("observation", [True] + [0.0] * 9, "real"),
    ("observation", [float("nan")] + [0.0] * 9, "finite"),
    ("action_xy", [1.01, 0.3], "\\[0, 1\\]"),
    ("action_xy", [0.1, 0.1], "next_position"),
    ("reward", float("inf"), "Reward"),
    ("done", 1, "boolean"),
    ("episode_id", "", "episode_id"),
    ("row_number", True, "row_number"),
])
def test_loader_rejects_invalid_transitions(tmp_path, field, value, match):
    row = _row()
    row[field] = value
    with pytest.raises(ValueError, match=match):
        load_split(_write(tmp_path / "invalid.jsonl", [row]))


def test_loader_rejects_mixed_modes_and_duplicate_rows(tmp_path):
    first = _row()
    second = _row(row_number=1, mode="commanded_destination")
    with pytest.raises(ValueError, match="Mixed label"):
        load_split(_write(tmp_path / "mixed.jsonl", [first, second]))
    with pytest.raises(ValueError, match="Duplicate"):
        load_split(_write(tmp_path / "duplicate.jsonl", [first, first]))


def test_split_isolation_rejects_episodes_shared_with_validation_or_test():
    train = _synthetic_split("same-episode")
    with pytest.raises(ValueError, match="Episode leakage"):
        validate_split_isolation(train=train, validation=_synthetic_split("same-episode"))
    with pytest.raises(ValueError, match="Episode leakage"):
        validate_split_isolation(
            train=train, validation=_synthetic_split("validation"), test=_synthetic_split("same-episode")
        )


def test_component_selection_preserves_distinct_paths():
    model = ContinuousMixturePolicy(components=2, hidden_sizes=(8,))
    with torch.no_grad():
        model.head.weight.zero_()
        # Unequal mixture weights; distinct means at (0.1,0.9), (0.9,0.1).
        logit = math.log(0.1 / 0.9)
        model.head.bias.copy_(torch.tensor([0.0, logit, -logit, -2.0, -2.0,
                                           1.0, -logit, logit, -2.0, -2.0]))
    predictions = model.predict(torch.zeros(3, 10))
    assert torch.allclose(predictions, torch.tensor([[0.9, 0.1]]).repeat(3, 1), atol=1e-6)
    assert not torch.allclose(predictions, torch.full_like(predictions, 0.5))


def test_single_component_likelihood_matches_diagonal_gaussian_and_has_gradients():
    logits = torch.zeros(2, 1, requires_grad=True)
    means = torch.tensor([[[0.2, 0.8]], [[0.4, 0.6]]], requires_grad=True)
    log_scales = torch.full((2, 1, 2), math.log(0.1), requires_grad=True)
    targets = torch.tensor([[0.3, 0.9], [0.4, 0.6]])
    actual = mixture_nll((logits, means, log_scales), targets)
    expected = -torch.distributions.Normal(means[:, 0], log_scales[:, 0].exp()).log_prob(targets).sum(-1)
    assert torch.allclose(actual, expected, atol=1e-6)
    actual.mean().backward()
    assert torch.isfinite(means.grad).all()
    assert torch.isfinite(log_scales.grad).all()


def test_tiny_training_uses_train_normalization_and_checkpoint_round_trip(tmp_path):
    train, validation = _synthetic_split("train"), _synthetic_split("validation", offset=0.4)
    model, mean, std, history = train_specialist(
        train, validation, components=2, hidden_sizes=(8,), epochs=2,
        batch_size=6, learning_rate=1e-3, seed=3,
    )
    assert np.allclose(mean, train.observations.mean(0))
    assert not np.allclose(mean, np.concatenate((train.observations, validation.observations)).mean(0))
    metrics = evaluate_specialist(model, validation, mean=mean, std=std)
    assert math.isfinite(metrics["nll"])
    assert "success_rate" not in metrics
    assert metrics["scope"] == "offline_component_diagnostic_only"
    payload = checkpoint_payload(model, mean=mean, std=std, train=train, validation=validation, history=history)
    path = tmp_path / "specialist.pt"
    torch.save(payload, path)
    policy = ContinuousCheckpointPolicy(path, expected_label_mode="commanded_destination")
    decision = policy.act(train.observations[0])
    expected = model.predict(torch.from_numpy(((train.observations[:1] - mean) / std)))[0]
    assert np.allclose(decision.action, expected.detach().numpy())
    assert all(0.0 <= value <= 1.0 for value in decision.action)
    assert decision.metadata["label_mode"] == "commanded_destination"
    policy.reset(10)
    assert decision == policy.act(train.observations[0])
    with pytest.raises(ValueError, match="label_mode"):
        ContinuousCheckpointPolicy(path, expected_label_mode="next_position")
    payload["action_schema"] = "old-discrete-action-schema"
    torch.save(payload, path)
    with pytest.raises(ValueError, match="action_schema"):
        ContinuousCheckpointPolicy(path)


def test_private_output_rejects_git_worktrees_and_existing_artifacts(tmp_path):
    checkout = tmp_path / "checkout"
    checkout.mkdir()
    (checkout / ".git").write_text("gitdir: /unused/worktrees/test")
    with pytest.raises(ValueError, match="outside"):
        _private_output(checkout / "runs" / "one")
    existing = tmp_path / "existing"
    existing.mkdir()
    (existing / "checkpoint.pt").write_bytes(b"do not overwrite")
    with pytest.raises(ValueError, match="not empty"):
        _private_output(existing)


def test_cli_writes_separate_offline_test_metrics_after_validation_selection(tmp_path, monkeypatch):
    paths = {}
    for split in ("train", "validation", "test"):
        paths[split] = _write(
            tmp_path / f"{split}.jsonl",
            [_row(episode=split, row_number=index) for index in range(4)],
        )
    output = tmp_path / "run"
    monkeypatch.setattr("sys.argv", [
        "specialist", "--train-data", str(paths["train"]),
        "--validation-data", str(paths["validation"]), "--test-data", str(paths["test"]),
        "--output-dir", str(output), "--components", "1", "--hidden-sizes", "8",
        "--epochs", "1", "--batch-size", "4", "--device", "cpu",
    ])
    main()
    report = json.loads((output / "metrics.json").read_text())
    assert report["selection_split"] == "validation"
    assert report["selected_epoch"] == 1
    assert report["closed_loop_evaluated"] is False
    assert set(report["evaluations"]) == {"train", "validation", "test"}
    assert report["provenance"]["test"]["source_sha256"] == load_split(paths["test"]).source_sha256
    assert ContinuousCheckpointPolicy(output / "continuous_specialist.pt").label_mode == "next_position"
