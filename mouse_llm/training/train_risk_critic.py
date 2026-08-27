"""Train and calibrate the compact P2 candidate-skill risk verifier."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from mouse_llm.hierarchical.policy import Skill
from mouse_llm.hierarchical.risk_critic import (
    RiskCriticMLP,
    SKILLS,
    skill_one_hot,
)


def load_rows(path: Path) -> list[dict[str, Any]]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    if not rows:
        raise ValueError(f"No risk rows in {path}")
    return rows


def arrays(rows: list[dict[str, Any]]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    features = np.asarray(
        [
            [
                *row["context_vector"],
                *skill_one_hot(Skill(row["candidate_skill"])),
            ]
            for row in rows
        ],
        dtype=np.float32,
    )
    targets = np.asarray([row["capture_within_h"] for row in rows], dtype=np.float32)
    skills = np.asarray(
        [SKILLS.index(Skill(row["candidate_skill"])) for row in rows], dtype=np.int64
    )
    if features.ndim != 2 or not np.isfinite(features).all():
        raise ValueError("Risk features must be a finite matrix")
    return features, targets, skills


def auroc(targets: np.ndarray, probabilities: np.ndarray) -> float | None:
    positives = probabilities[targets == 1]
    negatives = probabilities[targets == 0]
    if not len(positives) or not len(negatives):
        return None
    comparisons = (positives[:, None] > negatives[None, :]).mean()
    ties = (positives[:, None] == negatives[None, :]).mean()
    return float(comparisons + 0.5 * ties)


def auprc(targets: np.ndarray, probabilities: np.ndarray) -> float | None:
    positive_count = int(targets.sum())
    if positive_count == 0:
        return None
    order = np.argsort(-probabilities, kind="stable")
    ordered = targets[order]
    true_positives = np.cumsum(ordered)
    precision = true_positives / np.arange(1, len(ordered) + 1)
    return float(np.sum(precision * ordered) / positive_count)


def calibration_metrics(
    targets: np.ndarray, probabilities: np.ndarray, *, bins: int = 10
) -> dict[str, Any]:
    ece = 0.0
    bin_rows = []
    edges = np.linspace(0.0, 1.0, bins + 1)
    for index in range(bins):
        mask = (probabilities >= edges[index]) & (
            probabilities <= edges[index + 1]
            if index == bins - 1
            else probabilities < edges[index + 1]
        )
        if not mask.any():
            continue
        confidence = float(probabilities[mask].mean())
        frequency = float(targets[mask].mean())
        weight = float(mask.mean())
        ece += weight * abs(confidence - frequency)
        bin_rows.append(
            {
                "lower": float(edges[index]),
                "upper": float(edges[index + 1]),
                "count": int(mask.sum()),
                "mean_probability": confidence,
                "capture_frequency": frequency,
            }
        )
    return {
        "brier_score": float(np.mean((probabilities - targets) ** 2)),
        "ece": float(ece),
        "reliability_bins": bin_rows,
    }


def threshold_metrics(targets: np.ndarray, probabilities: np.ndarray) -> list[dict[str, Any]]:
    rows = []
    for threshold in np.linspace(0.05, 0.95, 19):
        predicted = probabilities >= threshold
        true_positive = int(np.sum(predicted & (targets == 1)))
        false_positive = int(np.sum(predicted & (targets == 0)))
        false_negative = int(np.sum(~predicted & (targets == 1)))
        precision = true_positive / (true_positive + false_positive) if true_positive + false_positive else 0.0
        recall = true_positive / (true_positive + false_negative) if true_positive + false_negative else 0.0
        rows.append(
            {
                "threshold": float(threshold),
                "precision": precision,
                "recall": recall,
                "predicted_risk_rate": float(predicted.mean()),
            }
        )
    return rows


def evaluate(
    targets: np.ndarray,
    probabilities: np.ndarray,
    skills: np.ndarray,
) -> dict[str, Any]:
    result = {
        "sample_count": len(targets),
        "positive_count": int(targets.sum()),
        "positive_rate": float(targets.mean()),
        "auroc": auroc(targets, probabilities),
        "auprc": auprc(targets, probabilities),
        **calibration_metrics(targets, probabilities),
        "operating_thresholds": threshold_metrics(targets, probabilities),
        "per_skill": {},
    }
    for index, skill in enumerate(SKILLS):
        mask = skills == index
        result["per_skill"][skill.value] = {
            "sample_count": int(mask.sum()),
            "positive_rate": float(targets[mask].mean()) if mask.any() else None,
            "brier_score": float(np.mean((probabilities[mask] - targets[mask]) ** 2)) if mask.any() else None,
        }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the P2 capture-risk critic")
    parser.add_argument("--train-data", type=Path, required=True)
    parser.add_argument("--validation-data", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=160)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--research-evidence", action="store_true")
    args = parser.parse_args()
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    device = torch.device(args.device)
    train_rows = load_rows(args.train_data)
    validation_rows = load_rows(args.validation_data)
    train_features, train_targets, _ = arrays(train_rows)
    validation_features, validation_targets, validation_skills = arrays(validation_rows)
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
    model = RiskCriticMLP(train_features.shape[1], hidden_dims).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)
    positive_count = max(float(train_targets.sum()), 1.0)
    negative_count = max(float(len(train_targets) - train_targets.sum()), 1.0)
    positive_weight = torch.tensor(negative_count / positive_count, device=device)
    best_state = None
    best_loss = float("inf")
    stale = 0
    for epoch in range(args.epochs):
        model.train()
        order = rng.permutation(len(train_x))
        for start in range(0, len(order), args.batch_size):
            indices = torch.from_numpy(order[start : start + args.batch_size]).to(device)
            logits = model((train_x[indices] - mean) / std)
            loss = nn.functional.binary_cross_entropy_with_logits(
                logits, train_y[indices], pos_weight=positive_weight
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        model.eval()
        with torch.inference_mode():
            validation_logits = model((validation_x - mean) / std)
            validation_loss = float(
                nn.functional.binary_cross_entropy_with_logits(
                    validation_logits, validation_y
                )
            )
        if validation_loss < best_loss - 1e-6:
            best_loss = validation_loss
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
        if stale >= args.patience:
            break
    if best_state is None:
        raise RuntimeError("Risk critic training produced no model")
    model.load_state_dict(best_state)
    model.eval()
    with torch.inference_mode():
        logits = model((validation_x - mean) / std).detach().cpu().numpy()
    temperatures = np.linspace(0.25, 4.0, 151)
    losses = []
    for temperature in temperatures:
        probabilities = 1.0 / (1.0 + np.exp(-logits / temperature))
        losses.append(
            -float(
                np.mean(
                    validation_targets * np.log(np.clip(probabilities, 1e-7, 1.0))
                    + (1.0 - validation_targets)
                    * np.log(np.clip(1.0 - probabilities, 1e-7, 1.0))
                )
            )
        )
    temperature = float(temperatures[int(np.argmin(losses))])
    probabilities = 1.0 / (1.0 + np.exp(-logits / temperature))
    operating_rows = threshold_metrics(validation_targets, probabilities)
    eligible = [row for row in operating_rows if row["recall"] >= 0.8]
    selected = max(eligible or operating_rows, key=lambda row: (row["precision"], row["recall"]))
    metrics = {
        "schema_version": 1,
        "artifact": "p2_risk_critic",
        "research_evidence": args.research_evidence,
        "research_evidence_blockers": (
            [] if args.research_evidence else ["smoke or incomplete dataset"]
        ),
        "parameter_count": sum(value.numel() for value in model.parameters()),
        "epochs_completed": epoch + 1,
        "class_imbalance": {
            "train_positive": int(train_targets.sum()),
            "train_negative": int(len(train_targets) - train_targets.sum()),
            "positive_weight": float(negative_count / positive_count),
        },
        "temperature": temperature,
        "offline_recommended_threshold": selected["threshold"],
        "threshold_status": "initial only; final threshold must be frozen on development rollouts",
        "validation": evaluate(validation_targets, probabilities, validation_skills),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
    torch.save(
        {
            "schema_version": 1,
            "context_dim": train_features.shape[1] - len(SKILLS),
            "hidden_dims": hidden_dims,
            "model_state": best_state,
            "feature_mean": mean.detach().cpu(),
            "feature_std": std.detach().cpu(),
            "temperature": temperature,
            "recommended_threshold": selected["threshold"],
        },
        args.output_dir / "risk_critic.pt",
    )
    temporary = args.output_dir / "risk_critic_metrics.json.tmp"
    temporary.write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, args.output_dir / "risk_critic_metrics.json")
    print(json.dumps(metrics, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
