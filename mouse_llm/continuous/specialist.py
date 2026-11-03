"""Continuous XY behavior cloning with a mixture-density specialist.

The policy predicts bounded destination means and executes the mean of the
highest-weight component, never the average of distinct paths. The likelihood
is an untruncated diagonal Gaussian mixture; only its means and deployed
destinations are bounded. ``next_position`` labels are short-step imitation
targets, not recovered historical commanded destinations. Offline XY error and
NLL are component diagnostics and are never reported as task success.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
from dataclasses import dataclass
from numbers import Real
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from mouse_llm.continuous.contracts import (
    ACTION_SCHEMA,
    DATA_SCHEMA,
    LABEL_MODES,
    ContinuousDecision,
    validate_xy,
)

CHECKPOINT_SCHEMA = "mousemind_continuous_mixture_bc_v1"
OBSERVATION_DIM = 10


@dataclass(frozen=True)
class ContinuousSplit:
    observations: np.ndarray
    actions: np.ndarray
    episode_ids: tuple[str, ...]
    label_mode: str
    source_sha256: str

    def provenance(self) -> dict[str, Any]:
        episodes = sorted(set(self.episode_ids))
        return {
            "source_sha256": self.source_sha256,
            "rows": len(self.actions),
            "episodes": len(episodes),
            "episode_ids_sha256": hashlib.sha256(
                json.dumps(episodes, separators=(",", ":")).encode()
            ).hexdigest(),
            "schema_version": DATA_SCHEMA,
            "action_schema": ACTION_SCHEMA,
            "label_mode": self.label_mode,
        }


def _observation(value: Any) -> np.ndarray:
    if not isinstance(value, (list, tuple, np.ndarray)):
        raise ValueError("Observation must be a ten-element numeric sequence")
    if len(value) != OBSERVATION_DIM or any(
        isinstance(item, (bool, np.bool_))
        or not isinstance(item, Real)
        for item in value
    ):
        raise ValueError("Observation must contain exactly ten real numbers")
    result = np.asarray(value, dtype=np.float32)
    if result.shape != (OBSERVATION_DIM,) or not np.isfinite(result).all():
        raise ValueError("Observation must contain ten finite float32 values")
    return result


def load_split(path: Path, *, expected_label_mode: str | None = None) -> ContinuousSplit:
    """Validate the entire transition contract before using its BC fields."""
    path = Path(path)
    observations, actions, episode_ids = [], [], []
    identities: set[tuple[str, int]] = set()
    label_mode = expected_label_mode
    if label_mode is not None and label_mode not in LABEL_MODES:
        raise ValueError(f"Unknown expected label mode: {label_mode}")
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for line_number, raw in enumerate(handle, start=1):
            digest.update(raw)
            if not raw.strip():
                continue
            try:
                row = json.loads(raw)
                if not isinstance(row, dict):
                    raise ValueError("Transition must be an object")
                if row.get("schema_version") != DATA_SCHEMA:
                    raise ValueError("Wrong continuous transition schema")
                if row.get("action_schema") != ACTION_SCHEMA:
                    raise ValueError("Wrong continuous action schema")
                mode = row.get("label_mode")
                if mode not in LABEL_MODES:
                    raise ValueError("Missing or unknown label_mode")
                if label_mode is None:
                    label_mode = mode
                if mode != label_mode:
                    raise ValueError("Mixed label modes are not allowed")
                observation = _observation(row["observation"])
                next_observation = _observation(row["next_observation"])
                action = validate_xy(row["action_xy"])
                if mode == "next_position" and not np.allclose(
                    action, next_observation[:2], atol=1e-7, rtol=0.0
                ):
                    raise ValueError("next_position action must match next_observation XY")
                reward = row["reward"]
                if isinstance(reward, bool) or not isinstance(reward, (int, float)) or not math.isfinite(reward):
                    raise ValueError("Reward must be finite and numeric")
                if not isinstance(row["done"], bool):
                    raise ValueError("done must be a JSON boolean")
                episode = row["episode_id"]
                if isinstance(episode, bool) or not isinstance(episode, (str, int)) or str(episode) == "":
                    raise ValueError("episode_id must be a nonempty string or integer")
                row_number = row["row_number"]
                if isinstance(row_number, bool) or not isinstance(row_number, int) or row_number < 0:
                    raise ValueError("row_number must be a non-negative integer")
                identity = (str(episode), row_number)
                if identity in identities:
                    raise ValueError("Duplicate episode/row identity")
                identities.add(identity)
            except (KeyError, TypeError, ValueError, OverflowError) as exc:
                raise ValueError(f"{path}:{line_number}: {exc}") from exc
            observations.append(observation)
            actions.append(action)
            episode_ids.append(str(episode))
    if not observations:
        raise ValueError(f"Empty continuous dataset: {path}")
    assert label_mode is not None
    return ContinuousSplit(
        observations=np.asarray(observations, dtype=np.float32),
        actions=np.asarray(actions, dtype=np.float32),
        episode_ids=tuple(episode_ids),
        label_mode=label_mode,
        source_sha256=digest.hexdigest(),
    )


def validate_split_isolation(**splits: ContinuousSplit) -> None:
    """Reject episode leakage and mixed label semantics across supplied splits."""
    modes = {split.label_mode for split in splits.values()}
    if len(modes) != 1:
        raise ValueError("All splits must use the same label_mode")
    owners: dict[str, str] = {}
    for name, split in splits.items():
        for episode in set(split.episode_ids):
            previous = owners.setdefault(episode, name)
            if previous != name:
                raise ValueError(f"Episode leakage between {previous} and {name}")


class ContinuousMixturePolicy(nn.Module):
    """Small MLP with component weights, bounded means and diagonal scales."""

    def __init__(self, *, components: int = 5, hidden_sizes: Sequence[int] = (256, 256)):
        super().__init__()
        if components < 1 or not hidden_sizes or any(size < 1 for size in hidden_sizes):
            raise ValueError("components and hidden sizes must be positive")
        self.components = int(components)
        self.hidden_sizes = tuple(int(size) for size in hidden_sizes)
        layers: list[nn.Module] = []
        input_size = OBSERVATION_DIM
        for size in self.hidden_sizes:
            layers.extend((nn.Linear(input_size, size), nn.LayerNorm(size), nn.GELU()))
            input_size = size
        self.encoder = nn.Sequential(*layers)
        self.head = nn.Linear(input_size, self.components * 5)

    def forward(self, observation: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        output = self.head(self.encoder(observation)).reshape(-1, self.components, 5)
        logits = output[..., 0]
        means = torch.sigmoid(output[..., 1:3])
        log_scales = output[..., 3:5].clamp(min=-5.0, max=0.0)
        return logits, means, log_scales

    def predict(self, observation: torch.Tensor) -> torch.Tensor:
        logits, means, _ = self(observation)
        indices = logits.argmax(dim=-1)
        return means[torch.arange(len(means), device=means.device), indices]


def mixture_nll(
    outputs: tuple[torch.Tensor, torch.Tensor, torch.Tensor], targets: torch.Tensor
) -> torch.Tensor:
    logits, means, log_scales = outputs
    scaled = (targets[:, None, :] - means) * torch.exp(-log_scales)
    component_log_prob = -0.5 * (
        scaled.square() + 2.0 * log_scales + math.log(2.0 * math.pi)
    ).sum(dim=-1)
    return -torch.logsumexp(torch.log_softmax(logits, dim=-1) + component_log_prob, dim=-1)


def normalization(observations: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = observations.astype(np.float64).mean(axis=0).astype(np.float32)
    std = observations.astype(np.float64).std(axis=0).astype(np.float32)
    std[std < 1e-6] = 1.0
    return mean, std


def _dataset(split: ContinuousSplit, mean: np.ndarray, std: np.ndarray) -> TensorDataset:
    return TensorDataset(
        torch.from_numpy(((split.observations - mean) / std).astype(np.float32)),
        torch.from_numpy(split.actions),
    )


def evaluate_specialist(
    model: ContinuousMixturePolicy,
    split: ContinuousSplit,
    *,
    mean: np.ndarray,
    std: np.ndarray,
    device: str | torch.device = "cpu",
    batch_size: int = 512,
) -> dict[str, Any]:
    loader = DataLoader(_dataset(split, mean, std), batch_size=batch_size, shuffle=False)
    losses, errors = [], []
    model.eval()
    with torch.inference_mode():
        for observations, targets in loader:
            observations, targets = observations.to(device), targets.to(device)
            outputs = model(observations)
            logits, means, _ = outputs
            indices = logits.argmax(dim=-1)
            predictions = means[torch.arange(len(means), device=means.device), indices]
            losses.extend(mixture_nll(outputs, targets).cpu().tolist())
            errors.extend(torch.linalg.vector_norm(predictions - targets, dim=-1).cpu().tolist())
    result = {
        "scope": "offline_component_diagnostic_only",
        "label_mode": split.label_mode,
        "rows": len(split.actions),
        "nll": float(np.mean(losses)),
        "xy_l2_mean": float(np.mean(errors)),
        "xy_l2_median": float(np.median(errors)),
        "xy_l2_p95": float(np.quantile(errors, 0.95)),
        "normalized_xy_l2_mean": float(np.mean(errors) / math.sqrt(2.0)),
    }
    if not all(math.isfinite(result[key]) for key in ("nll", "xy_l2_mean", "xy_l2_p95")):
        raise RuntimeError("Non-finite specialist evaluation metrics")
    return result


def train_specialist(
    train: ContinuousSplit,
    validation: ContinuousSplit,
    *,
    components: int = 5,
    hidden_sizes: Sequence[int] = (256, 256),
    epochs: int = 50,
    batch_size: int = 256,
    learning_rate: float = 3e-4,
    weight_decay: float = 1e-4,
    patience: int = 8,
    seed: int = 42,
    device: str | torch.device = "cpu",
) -> tuple[ContinuousMixturePolicy, np.ndarray, np.ndarray, list[dict[str, Any]]]:
    validate_split_isolation(train=train, validation=validation)
    if epochs < 1 or batch_size < 1 or learning_rate <= 0 or weight_decay < 0 or patience < 0:
        raise ValueError("Invalid specialist training hyperparameters")
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    mean, std = normalization(train.observations)
    loader = DataLoader(
        _dataset(train, mean, std), batch_size=batch_size, shuffle=True,
        generator=torch.Generator().manual_seed(seed),
    )
    model = ContinuousMixturePolicy(components=components, hidden_sizes=hidden_sizes).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    best_nll = math.inf
    best_state = None
    stale = 0
    history: list[dict[str, Any]] = []
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        samples = 0
        for observations, targets in loader:
            observations, targets = observations.to(device), targets.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = mixture_nll(model(observations), targets).mean()
            if not torch.isfinite(loss):
                raise RuntimeError("Non-finite specialist training loss")
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0, error_if_nonfinite=True)
            optimizer.step()
            total_loss += float(loss.detach()) * len(targets)
            samples += len(targets)
        metrics = evaluate_specialist(model, validation, mean=mean, std=std, device=device, batch_size=batch_size)
        improved = metrics["nll"] < best_nll - 1e-8
        row = {
            "epoch": epoch, "train_nll": total_loss / samples,
            "validation_nll": metrics["nll"], "validation_xy_l2": metrics["xy_l2_mean"],
            "selected": improved,
        }
        history.append(row)
        print(json.dumps(row, sort_keys=True), flush=True)
        if improved:
            best_nll = metrics["nll"]
            best_state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
            if patience and stale >= patience:
                break
    if best_state is None:
        raise RuntimeError("Specialist training did not select a checkpoint")
    model.load_state_dict(best_state)
    return model.eval(), mean, std, history


def checkpoint_payload(
    model: ContinuousMixturePolicy,
    *,
    mean: np.ndarray,
    std: np.ndarray,
    train: ContinuousSplit,
    validation: ContinuousSplit,
    history: list[dict[str, Any]],
    config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    validate_split_isolation(train=train, validation=validation)
    return {
        "checkpoint_schema": CHECKPOINT_SCHEMA,
        "data_schema": DATA_SCHEMA,
        "action_schema": ACTION_SCHEMA,
        "label_mode": train.label_mode,
        "observation_dim": OBSERVATION_DIM,
        "action_dim": 2,
        "components": model.components,
        "hidden_sizes": list(model.hidden_sizes),
        "prediction_rule": "highest_weight_component_mean",
        "likelihood": "untruncated_diagonal_gaussian_mixture_with_bounded_means",
        "normalization_source": "train_only",
        "normalization_mean": torch.as_tensor(mean, dtype=torch.float32),
        "normalization_std": torch.as_tensor(std, dtype=torch.float32),
        "state_dict": {name: tensor.detach().cpu() for name, tensor in model.state_dict().items()},
        "train_provenance": train.provenance(),
        "validation_provenance": validation.provenance(),
        "model_selection": "minimum_validation_nll",
        "selected_epoch": [row["epoch"] for row in history if row["selected"]][-1],
        "training_history": copy.deepcopy(history),
        "training_config": config or {},
        "closed_loop_evaluated": False,
    }


class ContinuousCheckpointPolicy:
    """A frozen checkpoint adapter for the continuous environment interface."""

    observation_mode = "legacy"
    action_schema = ACTION_SCHEMA

    def __init__(
        self,
        checkpoint: str | Path,
        *,
        device: str | torch.device = "cpu",
        expected_label_mode: str | None = None,
        name: str = "continuous-mixture-bc",
    ):
        self.name = name
        self.device = torch.device(device)
        payload = torch.load(Path(checkpoint), map_location="cpu", weights_only=True)
        if not isinstance(payload, dict):
            raise ValueError("Continuous checkpoint must be a metadata dictionary")
        expected = {
            "checkpoint_schema": CHECKPOINT_SCHEMA, "data_schema": DATA_SCHEMA,
            "action_schema": ACTION_SCHEMA, "observation_dim": OBSERVATION_DIM,
            "action_dim": 2, "prediction_rule": "highest_weight_component_mean",
            "normalization_source": "train_only",
        }
        for key, value in expected.items():
            if payload.get(key) != value:
                raise ValueError(f"Continuous checkpoint contract mismatch: {key}")
        self.label_mode = payload.get("label_mode")
        if self.label_mode not in LABEL_MODES or (
            expected_label_mode is not None and self.label_mode != expected_label_mode
        ):
            raise ValueError("Continuous checkpoint label_mode mismatch")
        self.mean = torch.as_tensor(payload["normalization_mean"], dtype=torch.float32, device=self.device)
        self.std = torch.as_tensor(payload["normalization_std"], dtype=torch.float32, device=self.device)
        if self.mean.shape != (OBSERVATION_DIM,) or self.std.shape != (OBSERVATION_DIM,):
            raise ValueError("Continuous checkpoint normalization has the wrong dimension")
        if not torch.isfinite(self.mean).all() or not torch.isfinite(self.std).all() or not (self.std > 0).all():
            raise ValueError("Continuous checkpoint normalization must be finite with positive scales")
        self.model = ContinuousMixturePolicy(
            components=int(payload["components"]), hidden_sizes=payload["hidden_sizes"]
        ).to(self.device)
        self.model.load_state_dict(payload["state_dict"], strict=True)
        if any(not torch.isfinite(parameter).all() for parameter in self.model.parameters()):
            raise ValueError("Continuous checkpoint contains non-finite weights")
        self.model.eval()
        self.metadata = {key: payload[key] for key in expected}
        self.metadata["label_mode"] = self.label_mode

    def reset(self, seed: int) -> None:
        # Deterministic component-mode inference has no recurrent or RNG state.
        del seed

    def act(self, observation: Sequence[float]) -> ContinuousDecision:
        values = _observation(observation)
        tensor = torch.as_tensor(values, device=self.device)
        with torch.inference_mode():
            prediction = self.model.predict(((tensor - self.mean) / self.std).unsqueeze(0))[0]
        xy = validate_xy(prediction.cpu().tolist())
        return ContinuousDecision(
            action=xy,
            metadata={"action_schema": ACTION_SCHEMA, "label_mode": self.label_mode,
                      "prediction_rule": "highest_weight_component_mean"},
        )


def _private_output(path: Path) -> None:
    resolved = path.expanduser().resolve()
    if any((parent / ".git").exists() for parent in (resolved, *resolved.parents)):
        raise ValueError("Private checkpoints and training artifacts must be outside every Git checkout")
    if resolved.exists() and (not resolved.is_dir() or any(resolved.iterdir())):
        raise ValueError("Output directory is not empty; select a new run directory")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-data", type=Path, required=True)
    parser.add_argument("--validation-data", type=Path, required=True)
    parser.add_argument("--test-data", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--label-mode", choices=LABEL_MODES)
    parser.add_argument("--components", type=int, default=5)
    parser.add_argument("--hidden-sizes", type=int, nargs="+", default=[256, 256])
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    _private_output(args.output_dir)
    train = load_split(args.train_data, expected_label_mode=args.label_mode)
    validation = load_split(args.validation_data, expected_label_mode=train.label_mode)
    test = load_split(args.test_data, expected_label_mode=train.label_mode) if args.test_data else None
    splits = {"train": train, "validation": validation}
    if test is not None:
        splits["test"] = test
    validate_split_isolation(**splits)
    config = {key: getattr(args, key) for key in (
        "components", "hidden_sizes", "epochs", "batch_size", "learning_rate",
        "weight_decay", "patience", "seed", "device",
    )}
    model, mean, std, history = train_specialist(train, validation, **config)
    payload = checkpoint_payload(
        model, mean=mean, std=std, train=train, validation=validation, history=history, config=config,
    )
    metrics = {
        "schema_version": CHECKPOINT_SCHEMA,
        "action_schema": ACTION_SCHEMA,
        "label_mode": train.label_mode,
        "scope": "offline_component_diagnostic_only",
        "closed_loop_evaluated": False,
        "selection_split": "validation",
        "selected_epoch": payload["selected_epoch"],
        "prediction_rule": payload["prediction_rule"],
        "likelihood": payload["likelihood"],
        "training_config": config,
        "history": history,
        "provenance": {name: split.provenance() for name, split in splits.items()},
        "evaluations": {
            name: evaluate_specialist(model, split, mean=mean, std=std, device=args.device)
            for name, split in splits.items()
        },
    }
    output = args.output_dir.expanduser().resolve()
    _private_output(output)
    output.mkdir(parents=True, exist_ok=True, mode=0o700)
    os.chmod(output, 0o700)
    checkpoint_path = output / "continuous_specialist.pt"
    torch.save(payload, checkpoint_path)
    checkpoint_path.chmod(0o600)
    metrics_path = output / "metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2, sort_keys=True, allow_nan=False) + "\n")
    metrics_path.chmod(0o600)
    print(json.dumps({"selected_epoch": payload["selected_epoch"], "evaluations": metrics["evaluations"]}, sort_keys=True))


if __name__ == "__main__":
    main()
