"""Train and evaluate a compact MLP behavior-cloning baseline.

The baseline consumes the same versioned 10D observations and 295 action
targets as MiniMind. Checkpoints and sample-level predictions are deliberately
kept outside Git; only aggregate metrics are suitable for publication.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from mouse_llm.data.schema import FEATURE_NAMES, conversation_observation
from mouse_llm.evaluation.evaluate_policy import load_action_catalog
from mouse_llm.evaluation.evaluate_policy import _bootstrap_mean


@dataclass(frozen=True)
class SupervisedSplit:
    observations: np.ndarray
    actions: np.ndarray


def load_supervised_jsonl(
    path: Path,
    *,
    action_count: int,
    max_samples: int = 0,
    seed: int = 42,
) -> SupervisedSplit:
    records: list[tuple[str, tuple[float, ...], int]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            payload = json.loads(line)
            conversations = payload.get("conversations")
            if not isinstance(conversations, list) or not conversations:
                raise ValueError(f"{path}:{line_number}: missing conversations")
            observation = conversation_observation(conversations)
            assistant = conversations[-1]
            if assistant.get("role") != "assistant":
                raise ValueError(f"{path}:{line_number}: missing assistant target")
            try:
                target = json.loads(assistant["content"])
            except (KeyError, TypeError, json.JSONDecodeError) as exc:
                raise ValueError(
                    f"{path}:{line_number}: invalid assistant target"
                ) from exc
            action = (
                target.get("action")
                if isinstance(target, dict) and set(target) == {"action"}
                else None
            )
            if (
                isinstance(action, bool)
                or not isinstance(action, int)
                or not 0 <= action < action_count
            ):
                raise ValueError(f"{path}:{line_number}: invalid action {action!r}")
            canonical = json.dumps(
                conversations, sort_keys=True, separators=(",", ":")
            )
            sample_id = hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]
            records.append((sample_id, observation, action))
    if not records:
        raise ValueError(f"No supervised samples found in {path}")
    if max_samples > 0 and len(records) > max_samples:
        records.sort(
            key=lambda record: hashlib.sha256(
                f"{seed}:{record[0]}".encode("utf-8")
            ).digest()
        )
        records = records[:max_samples]
        records.sort(key=lambda record: record[0])
    return SupervisedSplit(
        observations=np.asarray([record[1] for record in records], dtype=np.float32),
        actions=np.asarray([record[2] for record in records], dtype=np.int64),
    )


class MLPPolicy(nn.Module):
    """A deliberately strong, sub-million-parameter specialist baseline."""

    def __init__(
        self,
        *,
        observation_size: int = len(FEATURE_NAMES),
        action_count: int = 295,
        hidden_sizes: tuple[int, ...] = (256, 256),
    ):
        super().__init__()
        sizes = (observation_size, *hidden_sizes, action_count)
        layers: list[nn.Module] = []
        for index, (input_size, output_size) in enumerate(
            zip(sizes[:-1], sizes[1:], strict=True)
        ):
            layers.append(nn.Linear(input_size, output_size))
            if index < len(sizes) - 2:
                layers.extend((nn.LayerNorm(output_size), nn.GELU()))
        self.network = nn.Sequential(*layers)

    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        return self.network(observation)


def _normalization(observations: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = observations.astype(np.float64).mean(axis=0).astype(np.float32)
    std = observations.astype(np.float64).std(axis=0).astype(np.float32)
    std[std < 1e-6] = 1.0
    return mean, std


def _tensor_split(
    split: SupervisedSplit, *, mean: np.ndarray, std: np.ndarray
) -> TensorDataset:
    normalized = (split.observations - mean) / std
    return TensorDataset(
        torch.from_numpy(normalized.astype(np.float32, copy=False)),
        torch.from_numpy(split.actions),
    )


def _accuracy(
    model: MLPPolicy,
    dataset: TensorDataset,
    *,
    device: torch.device,
    batch_size: int,
) -> float:
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    correct = 0
    count = 0
    model.eval()
    with torch.inference_mode():
        for observations, actions in loader:
            predicted = model(observations.to(device)).argmax(dim=-1).cpu()
            correct += int((predicted == actions).sum())
            count += len(actions)
    return correct / count


def train_mlp(
    train: SupervisedSplit,
    validation: SupervisedSplit,
    *,
    action_count: int,
    hidden_sizes: tuple[int, ...],
    epochs: int,
    batch_size: int,
    learning_rate: float,
    weight_decay: float,
    patience: int,
    seed: int,
    device: torch.device,
) -> tuple[MLPPolicy, np.ndarray, np.ndarray, list[dict[str, float]]]:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    mean, std = _normalization(train.observations)
    train_dataset = _tensor_split(train, mean=mean, std=std)
    validation_dataset = _tensor_split(validation, mean=mean, std=std)
    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        generator=generator,
    )
    model = MLPPolicy(
        observation_size=train.observations.shape[1],
        action_count=action_count,
        hidden_sizes=hidden_sizes,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    criterion = nn.CrossEntropyLoss()
    best_accuracy = -math.inf
    best_state: dict[str, torch.Tensor] | None = None
    epochs_without_improvement = 0
    history: list[dict[str, float]] = []
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        sample_count = 0
        for observations, actions in loader:
            observations = observations.to(device)
            actions = actions.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(observations), actions)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.detach()) * len(actions)
            sample_count += len(actions)
        validation_accuracy = _accuracy(
            model,
            validation_dataset,
            device=device,
            batch_size=batch_size,
        )
        row = {
            "epoch": float(epoch),
            "train_loss": total_loss / sample_count,
            "validation_accuracy": validation_accuracy,
        }
        history.append(row)
        print(json.dumps(row, sort_keys=True), flush=True)
        if validation_accuracy > best_accuracy + 1e-12:
            best_accuracy = validation_accuracy
            best_state = {
                name: tensor.detach().cpu().clone()
                for name, tensor in model.state_dict().items()
            }
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if patience > 0 and epochs_without_improvement >= patience:
                break
    if best_state is None:
        raise RuntimeError("MLP training produced no checkpoint")
    model.load_state_dict(best_state)
    model.to(device).eval()
    return model, mean, std, history


def evaluate_mlp(
    model: MLPPolicy,
    split: SupervisedSplit,
    *,
    mean: np.ndarray,
    std: np.ndarray,
    destinations: np.ndarray,
    device: torch.device,
    warmup: int = 10,
    seed: int = 42,
) -> tuple[dict[str, Any], np.ndarray]:
    normalized = (split.observations - mean) / std
    tensor = torch.from_numpy(normalized.astype(np.float32, copy=False))
    if len(tensor) and warmup > 0:
        with torch.inference_mode():
            for _ in range(warmup):
                model(tensor[:1].to(device))
        if device.type == "cuda":
            torch.cuda.synchronize(device)
    predictions: list[int] = []
    latencies: list[float] = []
    model.eval()
    with torch.inference_mode():
        for observation in tensor:
            start = time.perf_counter()
            logits = model(observation.unsqueeze(0).to(device))
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            latencies.append(time.perf_counter() - start)
            predictions.append(int(logits.argmax(dim=-1).cpu()))
    predicted = np.asarray(predictions, dtype=np.int64)
    latency = np.asarray(latencies, dtype=np.float64)
    spatial_errors = np.linalg.norm(
        destinations[predicted] - destinations[split.actions], axis=1
    ) / math.sqrt(2.0)
    metrics = {
        "sample_count": len(split.actions),
        "exact_action_accuracy": float((predicted == split.actions).mean()),
        "exact_action_accuracy_95_ci": _bootstrap_mean(
            (predicted == split.actions).astype(np.float64).tolist(),
            seed=seed + 1,
        ),
        "normalized_destination_error": float(spatial_errors.mean()),
        "normalized_destination_error_95_ci": _bootstrap_mean(
            spatial_errors.tolist(), seed=seed + 2
        ),
        "latency_seconds": {
            "mean": float(latency.mean()),
            "p50": float(np.quantile(latency, 0.50)),
            "p95": float(np.quantile(latency, 0.95)),
            "p99": float(np.quantile(latency, 0.99)),
        },
    }
    return metrics, predicted


def save_checkpoint(
    path: Path,
    model: MLPPolicy,
    *,
    mean: np.ndarray,
    std: np.ndarray,
    hidden_sizes: tuple[int, ...],
    action_count: int,
) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(
        {
            "schema_version": 1,
            "policy": "mlp_bc",
            "observation_size": len(FEATURE_NAMES),
            "action_count": action_count,
            "hidden_sizes": list(hidden_sizes),
            "observation_mean": torch.from_numpy(mean),
            "observation_std": torch.from_numpy(std),
            "model_state": model.state_dict(),
        },
        temporary,
    )
    os.replace(temporary, path)


def load_checkpoint(
    path: Path, *, device: torch.device
) -> tuple[MLPPolicy, torch.Tensor, torch.Tensor, dict[str, Any]]:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if payload.get("schema_version") != 1 or payload.get("policy") != "mlp_bc":
        raise ValueError(f"Unsupported MLP checkpoint: {path}")
    metadata = {
        "observation_size": int(payload["observation_size"]),
        "action_count": int(payload["action_count"]),
        "hidden_sizes": tuple(int(value) for value in payload["hidden_sizes"]),
    }
    model = MLPPolicy(**metadata).to(device)
    model.load_state_dict(payload["model_state"], strict=True)
    model.eval()
    mean = payload["observation_mean"].to(device=device, dtype=torch.float32)
    std = payload["observation_std"].to(device=device, dtype=torch.float32)
    return model, mean, std, metadata


def _atomic_json(path: Path, payload: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def _evidence_metadata(train_data: Path) -> dict[str, Any]:
    manifest_path = train_data.parent / "manifest.json"
    if not manifest_path.is_file():
        return {
            "synthetic": None,
            "research_evidence": False,
            "evidence_note": "dataset manifest not found",
        }
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    synthetic = bool(manifest.get("synthetic", False))
    return {
        "synthetic": synthetic,
        "research_evidence": bool(
            manifest.get("research_evidence", not synthetic)
        ),
        "evidence_note": manifest.get("purpose", "held-out dataset evaluation"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train the fair MLP behavior-cloning baseline"
    )
    parser.add_argument("--train-data", type=Path, required=True)
    parser.add_argument("--validation-data", type=Path, required=True)
    parser.add_argument("--test-data", type=Path, required=True)
    parser.add_argument("--action-catalog", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--action-count", type=int, default=295)
    parser.add_argument("--hidden-sizes", type=int, nargs="+", default=(256, 256))
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument(
        "--max-test-samples",
        type=int,
        default=512,
        help="Use 512 and the same seed to match MiniMind's deterministic subset.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    args = parser.parse_args()

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    destinations = load_action_catalog(args.action_catalog)
    if len(destinations) != args.action_count:
        raise ValueError("action-count does not match action catalog")
    train = load_supervised_jsonl(args.train_data, action_count=args.action_count)
    validation = load_supervised_jsonl(
        args.validation_data, action_count=args.action_count
    )
    test = load_supervised_jsonl(
        args.test_data,
        action_count=args.action_count,
        max_samples=args.max_test_samples,
        seed=args.seed,
    )
    hidden_sizes = tuple(args.hidden_sizes)
    model, mean, std, history = train_mlp(
        train,
        validation,
        action_count=args.action_count,
        hidden_sizes=hidden_sizes,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        patience=args.patience,
        seed=args.seed,
        device=device,
    )
    metrics, _ = evaluate_mlp(
        model,
        test,
        mean=mean,
        std=std,
        destinations=destinations,
        device=device,
        seed=args.seed,
    )
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    result = {
        "schema_version": 1,
        "experiment": "mouse_policy_mlp_bc",
        **_evidence_metadata(args.train_data),
        "seed": args.seed,
        "device": str(device),
        "architecture": {
            "observation_size": len(FEATURE_NAMES),
            "hidden_sizes": list(hidden_sizes),
            "action_count": args.action_count,
            "parameter_count": parameter_count,
        },
        "best_epoch": int(
            max(history, key=lambda row: row["validation_accuracy"])["epoch"]
        ),
        "best_validation_accuracy": max(
            row["validation_accuracy"] for row in history
        ),
        "evaluation": {
            "split": "held-out episodes",
            "sample_count": len(test.actions),
            "deterministic_subset_seed": args.seed,
            "max_test_samples": args.max_test_samples,
        },
        "test": metrics,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
    args.output_dir.chmod(0o700)
    checkpoint_path = args.output_dir / "mlp_bc.pt"
    metrics_path = args.output_dir / "mlp_metrics.json"
    save_checkpoint(
        checkpoint_path,
        model,
        mean=mean,
        std=std,
        hidden_sizes=hidden_sizes,
        action_count=args.action_count,
    )
    _atomic_json(metrics_path, result)
    checkpoint_path.chmod(0o600)
    metrics_path.chmod(0o600)
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)
    print(f"CHECKPOINT={checkpoint_path}", flush=True)


if __name__ == "__main__":
    main()
