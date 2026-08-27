"""Strong language-free numeric baseline for counterfactual skill targets."""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from mouse_llm.data.planner_schema import Preference, preference_one_hot
from mouse_llm.hierarchical.context import PlannerContext
from mouse_llm.hierarchical.policy import PlannerDecision, Skill


SKILLS = tuple(Skill)


class PlannerMLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dims: tuple[int, ...] = (128, 64)):
        super().__init__()
        layers: list[nn.Module] = []
        previous = input_dim
        for width in hidden_dims:
            layers.extend((nn.Linear(previous, width), nn.ReLU()))
            previous = width
        layers.append(nn.Linear(previous, len(SKILLS)))
        self.network = nn.Sequential(*layers)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.network(features)


def load_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("target_skill") not in {skill.value for skill in SKILLS}:
                raise ValueError(f"Line {line_number}: invalid planner target")
            rows.append(row)
    if not rows:
        raise ValueError(f"No planner rows in {path}")
    return rows


def _arrays(rows: list[dict[str, Any]]) -> tuple[np.ndarray, np.ndarray]:
    features = np.asarray([row["context_vector"] for row in rows], dtype=np.float32)
    targets = np.asarray(
        [SKILLS.index(Skill(row["target_skill"])) for row in rows], dtype=np.int64
    )
    if features.ndim != 2 or not np.isfinite(features).all():
        raise ValueError("Planner features must be a finite matrix")
    return features, targets


def classification_metrics(targets: np.ndarray, predictions: np.ndarray) -> dict[str, Any]:
    confusion = np.zeros((len(SKILLS), len(SKILLS)), dtype=np.int64)
    for target, prediction in zip(targets, predictions, strict=True):
        confusion[int(target), int(prediction)] += 1
    per_skill: dict[str, Any] = {}
    f1_values = []
    for index, skill in enumerate(SKILLS):
        true_positive = int(confusion[index, index])
        predicted_count = int(confusion[:, index].sum())
        target_count = int(confusion[index, :].sum())
        precision = true_positive / predicted_count if predicted_count else 0.0
        recall = true_positive / target_count if target_count else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        f1_values.append(f1)
        per_skill[skill.value] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": target_count,
        }
    return {
        "accuracy": float(np.mean(targets == predictions)),
        "macro_f1": float(np.mean(f1_values)),
        "per_skill": per_skill,
        "confusion_matrix": confusion.tolist(),
        "skill_order": [skill.value for skill in SKILLS],
    }


def evaluate_rows(
    model: PlannerMLP,
    rows: list[dict[str, Any]],
    *,
    mean: torch.Tensor,
    std: torch.Tensor,
    device: torch.device,
) -> dict[str, Any]:
    features, targets = _arrays(rows)
    tensor = torch.from_numpy(features).to(device)
    latencies = []
    predictions: list[int] = []
    with torch.inference_mode():
        for feature in tensor:
            start = time.perf_counter()
            prediction = int(model(((feature - mean) / std).unsqueeze(0)).argmax(-1))
            latencies.append(time.perf_counter() - start)
            predictions.append(prediction)
    predicted = np.asarray(predictions, dtype=np.int64)
    metrics = classification_metrics(targets, predicted)
    regrets = []
    for row, prediction in zip(rows, predicted, strict=True):
        utilities = row["utilities"]
        chosen = utilities[SKILLS[int(prediction)].value]
        oracle = max(float(value) for value in utilities.values())
        regrets.append(float(chosen) - oracle)
    metrics["counterfactual_regret_chosen_minus_oracle"] = {
        "mean": float(np.mean(regrets)),
        "p05": float(np.quantile(regrets, 0.05)),
        "minimum": float(np.min(regrets)),
    }
    metrics["latency_seconds"] = {
        "p50": float(np.quantile(latencies, 0.50)),
        "p95": float(np.quantile(latencies, 0.95)),
        "p99": float(np.quantile(latencies, 0.99)),
    }
    metrics["sample_count"] = len(rows)
    return metrics


class NumericSkillPlanner:
    def __init__(self, checkpoint: Path, *, device: str = "cpu"):
        payload = torch.load(checkpoint, map_location=device, weights_only=False)
        if payload.get("schema_version") != 1:
            raise ValueError(f"Unsupported numeric planner checkpoint: {checkpoint}")
        self.device = torch.device(device)
        self.input_dim = int(payload["input_dim"])
        self.model = PlannerMLP(
            self.input_dim, tuple(int(value) for value in payload["hidden_dims"])
        ).to(self.device)
        self.model.load_state_dict(payload["model_state"])
        self.model.eval()
        self.mean = payload["feature_mean"].to(self.device)
        self.std = payload["feature_std"].to(self.device)

    def plan_context(
        self, context: PlannerContext, preference: Preference
    ) -> PlannerDecision:
        features = np.asarray(
            [*context.numeric().tolist(), *preference_one_hot(preference)],
            dtype=np.float32,
        )
        if len(features) != self.input_dim:
            raise ValueError(
                f"Planner expects {self.input_dim} features, got {len(features)}"
            )
        tensor = torch.from_numpy(features).to(self.device)
        with torch.inference_mode():
            prediction = int(
                self.model(((tensor - self.mean) / self.std).unsqueeze(0)).argmax(-1)
            )
        return PlannerDecision(SKILLS[prediction], "numeric_counterfactual_planner")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the P2 numeric skill planner")
    parser.add_argument("--train-data", type=Path, required=True)
    parser.add_argument("--validation-data", type=Path, required=True)
    parser.add_argument("--seen-test-data", type=Path)
    parser.add_argument("--unseen-test-data", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--research-evidence", action="store_true")
    args = parser.parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device)
    train_rows = load_rows(args.train_data)
    validation_rows = load_rows(args.validation_data)
    train_features, train_targets = _arrays(train_rows)
    validation_features, validation_targets = _arrays(validation_rows)
    mean_np = train_features.mean(axis=0)
    std_np = train_features.std(axis=0)
    std_np[std_np < 1e-6] = 1.0
    mean = torch.from_numpy(mean_np).to(device)
    std = torch.from_numpy(std_np).to(device)
    train_x = torch.from_numpy(train_features).to(device)
    train_y = torch.from_numpy(train_targets).to(device)
    validation_x = torch.from_numpy(validation_features).to(device)
    validation_y = torch.from_numpy(validation_targets).to(device)
    hidden_dims = (128, 64)
    model = PlannerMLP(train_features.shape[1], hidden_dims).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    rng = np.random.default_rng(args.seed)
    best_state: dict[str, torch.Tensor] | None = None
    best_accuracy = -1.0
    stale_epochs = 0
    for epoch in range(args.epochs):
        model.train()
        order = rng.permutation(len(train_x))
        for start in range(0, len(order), args.batch_size):
            indices = torch.from_numpy(order[start : start + args.batch_size]).to(device)
            logits = model((train_x[indices] - mean) / std)
            loss = nn.functional.cross_entropy(logits, train_y[indices])
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        model.eval()
        with torch.inference_mode():
            predictions = model((validation_x - mean) / std).argmax(-1)
            accuracy = float((predictions == validation_y).float().mean())
        if accuracy > best_accuracy + 1e-8:
            best_accuracy = accuracy
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
            stale_epochs = 0
        else:
            stale_epochs += 1
        if stale_epochs >= args.patience:
            break
    if best_state is None:
        raise RuntimeError("Numeric planner training produced no model")
    model.load_state_dict(best_state)
    model.eval()
    metrics: dict[str, Any] = {
        "schema_version": 1,
        "artifact": "p2_offline_numeric_planner",
        "research_evidence": args.research_evidence,
        "research_evidence_blockers": (
            [] if args.research_evidence else ["smoke or incomplete dataset"]
        ),
        "parameter_count": sum(value.numel() for value in model.parameters()),
        "epochs_completed": epoch + 1,
        "validation": evaluate_rows(
            model, validation_rows, mean=mean, std=std, device=device
        ),
    }
    for name, path in (
        ("seen_instruction_test", args.seen_test_data),
        ("unseen_paraphrase_test", args.unseen_test_data),
    ):
        if path is not None:
            metrics[name] = evaluate_rows(
                model, load_rows(path), mean=mean, std=std, device=device
            )
    args.output_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
    checkpoint = {
        "schema_version": 1,
        "input_dim": train_features.shape[1],
        "hidden_dims": hidden_dims,
        "model_state": best_state,
        "feature_mean": mean.detach().cpu(),
        "feature_std": std.detach().cpu(),
    }
    torch.save(checkpoint, args.output_dir / "planner_mlp.pt")
    temporary = args.output_dir / "planner_mlp_metrics.json.tmp"
    temporary.write_text(
        json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(temporary, args.output_dir / "planner_mlp_metrics.json")
    print(json.dumps(metrics, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
