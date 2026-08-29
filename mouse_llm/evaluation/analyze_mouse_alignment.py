"""Measure one-step policy alignment with held-out source mouse trajectories.

The private transition rows and per-sample predictions stay outside Git.  This
module joins the frozen 512-sample evaluation subset to existing constrained
MiniMind predictions, evaluates the MLP checkpoint on the identical states,
and writes aggregate-only JSON and Markdown artifacts.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from mouse_llm.data.prepare_dataset import AUDITED_SOURCE_SHA256
from mouse_llm.data.schema import conversation_observation
from mouse_llm.evaluation.evaluate_policy import load_action_catalog, load_samples


POLICY_LABELS = {
    "base-minimind-constrained": "Base MiniMind (constrained)",
    "mouse-policy-lora-constrained": "Mouse-policy LoRA (constrained)",
    "mlp-bc": "MLP BC (low-level upper reference)",
}


@dataclass(frozen=True)
class AlignmentSample:
    sample_id: str
    observation: tuple[float, ...]
    target_action: int

    @property
    def predator_visible(self) -> bool:
        return self.observation[3] != 0.0 or self.observation[4] != 0.0


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_alignment_samples(
    path: Path, *, max_samples: int, seed: int
) -> list[AlignmentSample]:
    samples = load_samples(path, max_samples=max_samples, seed=seed)
    return [
        AlignmentSample(
            sample_id=sample.sample_id,
            observation=conversation_observation(sample.conversations),
            target_action=sample.target_action,
        )
        for sample in samples
    ]


def load_direct_predictions(
    path: Path,
    samples: Sequence[AlignmentSample],
) -> dict[str, list[int | None]]:
    rows: dict[str, dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            payload = json.loads(line)
            sample_id = payload.get("sample_id")
            if not isinstance(sample_id, str) or sample_id in rows:
                raise ValueError(
                    f"{path}:{line_number}: missing or duplicate sample_id"
                )
            rows[sample_id] = payload

    expected_ids = {sample.sample_id for sample in samples}
    if set(rows) != expected_ids:
        missing = len(expected_ids - set(rows))
        extra = len(set(rows) - expected_ids)
        raise ValueError(
            "Direct predictions do not match the frozen sample set: "
            f"missing={missing}, extra={extra}"
        )

    result: dict[str, list[int | None]] = {
        "base-minimind-constrained": [],
        "mouse-policy-lora-constrained": [],
    }
    for sample in samples:
        row = rows[sample.sample_id]
        for source_name, policy_name in (
            ("base", "base-minimind-constrained"),
            ("lora", "mouse-policy-lora-constrained"),
        ):
            prediction = row.get(source_name)
            if not isinstance(prediction, dict):
                raise ValueError(
                    f"Prediction {sample.sample_id} has no {source_name} result"
                )
            if (
                prediction.get("sample_id") != sample.sample_id
                or int(prediction.get("target_action", -1)) != sample.target_action
            ):
                raise ValueError(
                    f"Prediction target mismatch for {sample.sample_id}/{source_name}"
                )
            action = prediction.get("predicted_action")
            if action is not None and (
                isinstance(action, bool) or not isinstance(action, int)
            ):
                raise ValueError(
                    f"Invalid predicted action for {sample.sample_id}/{source_name}"
                )
            result[policy_name].append(action)
    return result


def mlp_predictions(
    checkpoint: Path,
    samples: Sequence[AlignmentSample],
    *,
    device_name: str,
) -> list[int]:
    import torch

    from mouse_llm.baselines.mlp_bc import load_checkpoint

    if device_name == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")

    model, mean, std, metadata = load_checkpoint(checkpoint, device=device)
    observations = np.asarray(
        [sample.observation for sample in samples], dtype=np.float32
    )
    if observations.shape[1] != int(metadata["observation_size"]):
        raise ValueError("MLP checkpoint observation size does not match the data")
    tensor = torch.from_numpy(observations).to(device)
    tensor = (tensor - mean) / std
    predictions: list[int] = []
    with torch.inference_mode():
        for batch in tensor.split(512):
            predictions.extend(
                int(value) for value in model(batch).argmax(dim=-1).cpu().tolist()
            )
    return predictions


def _bootstrap_mean(
    values: np.ndarray, *, seed: int, iterations: int
) -> dict[str, float]:
    if values.size == 0:
        raise ValueError("Cannot bootstrap an empty alignment stratum")
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, values.size, size=(iterations, values.size))
    means = values[indices].mean(axis=1)
    return {
        "mean": float(values.mean()),
        "ci_low": float(np.quantile(means, 0.025)),
        "ci_high": float(np.quantile(means, 0.975)),
    }


def _js_divergence_bits(
    targets: np.ndarray, predictions: np.ndarray, *, action_count: int
) -> float:
    target_counts = np.bincount(targets, minlength=action_count + 1).astype(
        np.float64
    )
    prediction_counts = np.bincount(
        predictions, minlength=action_count + 1
    ).astype(np.float64)
    target_probability = target_counts / target_counts.sum()
    prediction_probability = prediction_counts / prediction_counts.sum()
    midpoint = (target_probability + prediction_probability) / 2.0

    def kl(left: np.ndarray, right: np.ndarray) -> float:
        mask = left > 0
        return float(np.sum(left[mask] * np.log2(left[mask] / right[mask])))

    return (kl(target_probability, midpoint) + kl(prediction_probability, midpoint)) / 2


def summarize_alignment(
    samples: Sequence[AlignmentSample],
    predictions: Sequence[int | None],
    destinations: np.ndarray,
    *,
    seed: int,
    bootstrap_iterations: int = 2000,
) -> dict[str, Any]:
    if len(samples) != len(predictions) or not samples:
        raise ValueError("Samples and predictions must be non-empty and paired")
    action_count = len(destinations)
    if destinations.shape != (action_count, 2):
        raise ValueError("Action destinations must be a two-column array")

    targets = np.asarray([sample.target_action for sample in samples], dtype=np.int64)
    predicted = np.asarray(
        [action_count if action is None else int(action) for action in predictions],
        dtype=np.int64,
    )
    if np.any(targets < 0) or np.any(targets >= action_count):
        raise ValueError("Target action is outside the action catalog")
    if np.any(predicted < 0) or np.any(predicted > action_count):
        raise ValueError("Predicted action is outside the action catalog")

    valid = predicted < action_count
    exact = (predicted == targets).astype(np.float64)
    destination_error = np.ones(len(samples), dtype=np.float64)
    destination_error[valid] = np.linalg.norm(
        destinations[predicted[valid]] - destinations[targets[valid]], axis=1
    ) / math.sqrt(2.0)
    visible = np.asarray(
        [sample.predator_visible for sample in samples], dtype=np.bool_
    )

    per_class: list[float] = []
    for target in sorted(set(targets.tolist())):
        mask = targets == target
        if int(mask.sum()) >= 2:
            per_class.append(float(exact[mask].mean()))

    strata: dict[str, Any] = {}
    for offset, (name, mask) in enumerate(
        (
            ("overall", np.ones(len(samples), dtype=np.bool_)),
            ("predator_hidden", ~visible),
            ("predator_visible", visible),
        )
    ):
        if not np.any(mask):
            raise ValueError(f"The frozen sample set has no {name} observations")
        strata[name] = {
            "sample_count": int(mask.sum()),
            "valid_output_rate": _bootstrap_mean(
                valid[mask].astype(np.float64),
                seed=seed + offset * 20 + 1,
                iterations=bootstrap_iterations,
            ),
            "exact_action_agreement": _bootstrap_mean(
                exact[mask],
                seed=seed + offset * 20 + 2,
                iterations=bootstrap_iterations,
            ),
            "normalized_destination_error": _bootstrap_mean(
                destination_error[mask],
                seed=seed + offset * 20 + 3,
                iterations=bootstrap_iterations,
            ),
        }
    strata["overall"]["macro_action_recall"] = (
        float(np.mean(per_class)) if per_class else 0.0
    )
    # The plug-in JS estimator is upward-biased under ordinary resampling when
    # there are hundreds of sparse action bins, so publish the transparent
    # point estimate rather than a misleading percentile interval.
    strata["overall"]["action_distribution_js_divergence_bits"] = {
        "mean": _js_divergence_bits(
            targets, predicted, action_count=action_count
        )
    }
    return strata


def build_report(
    *,
    samples: Sequence[AlignmentSample],
    predictions_by_policy: dict[str, Sequence[int | None]],
    destinations: np.ndarray,
    manifest: dict[str, Any],
    test_data_sha256: str,
    direct_predictions_sha256: str,
    mlp_checkpoint_sha256: str,
    seed: int,
    bootstrap_iterations: int,
) -> dict[str, Any]:
    source = manifest.get("source", {})
    blockers: list[str] = []
    if manifest.get("research_evidence") is False:
        blockers.append("processed dataset manifest is not research evidence")
    if manifest.get("synthetic") is True:
        blockers.append("processed dataset manifest is synthetic")
    if source.get("sha256") != AUDITED_SOURCE_SHA256:
        blockers.append("source trajectory hash is outside the observation audit")
    if len(samples) != 512:
        blockers.append("evaluation does not use the frozen 512-sample subset")

    policies = {
        name: summarize_alignment(
            samples,
            predictions,
            destinations,
            seed=seed + index * 1000,
            bootstrap_iterations=bootstrap_iterations,
        )
        for index, (name, predictions) in enumerate(predictions_by_policy.items())
    }
    return {
        "schema_version": 1,
        "artifact": "mousemind_source_trajectory_alignment",
        "experiment": "held_out_source_mouse_action_alignment",
        "synthetic": False,
        "research_evidence": not blockers,
        "research_evidence_blockers": blockers,
        "evaluation": {
            "interpretation": "teacher-forced one-step action alignment",
            "split": "episode-isolated held-out source trajectories",
            "sample_count": len(samples),
            "deterministic_subset_seed": seed,
            "bootstrap_iterations": bootstrap_iterations,
            "predator_visible_definition": (
                "legacy predator_x or predator_y is nonzero"
            ),
            "source_filename": source.get("filename"),
            "source_rows": source.get("rows"),
            "source_sha256": source.get("sha256"),
            "test_data_sha256": test_data_sha256,
            "direct_predictions_sha256": direct_predictions_sha256,
            "mlp_checkpoint_sha256": mlp_checkpoint_sha256,
            "action_count": len(destinations),
            "invalid_prediction_destination_error": 1.0,
        },
        "metrics": {
            "exact_action_agreement": "higher is better",
            "normalized_destination_error": (
                "lower is better; Euclidean destination distance / sqrt(2)"
            ),
            "action_distribution_js_divergence_bits": (
                "lower is better; 0 identical, 1 maximally different"
            ),
        },
        "policies": policies,
        "limitations": [
            "The legacy export contains simulator policy transitions; this "
            "analysis does not establish alignment with biological mouse behavior.",
            "Teacher-forced one-step agreement is an offline diagnostic and is "
            "not a closed-loop safety or task-success result.",
            "The predator-visible stratum is reported separately and has a "
            "smaller sample count, so its interval is wider.",
        ],
    }


def _percent_interval(value: dict[str, float]) -> str:
    return (
        f"{100 * value['mean']:.1f}% "
        f"({100 * value['ci_low']:.1f}–{100 * value['ci_high']:.1f}%)"
    )


def render_markdown(report: dict[str, Any]) -> str:
    if report.get("artifact") != "mousemind_source_trajectory_alignment":
        raise ValueError("Not a MouseMind source-trajectory alignment report")
    policies = report["policies"]
    first = next(iter(policies.values()))
    overall_count = first["overall"]["sample_count"]
    hidden_count = first["predator_hidden"]["sample_count"]
    visible_count = first["predator_visible"]["sample_count"]
    lines = [
        "# Source mouse trajectory alignment",
        "",
        "This is a low-level component diagnostic on the frozen, episode-isolated "
        f"held-out subset (N={overall_count}). It measures one-step agreement with "
        "the source trajectory actions. Hierarchical methods are compared separately "
        "with complete closed-loop episodes in `TRAJECTORY_ALIGNMENT.md`.",
        "",
        "| Policy | Overall action agreement | Predator hidden "
        f"(N={hidden_count}) | Predator visible (N={visible_count}) | "
        "Destination error | Action-distribution JS |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    ordered_names = [name for name in POLICY_LABELS if name in policies]
    ordered_names.extend(name for name in policies if name not in POLICY_LABELS)
    for name in ordered_names:
        summary = policies[name]
        overall = summary["overall"]
        hidden_agreement = summary["predator_hidden"]["exact_action_agreement"]
        visible_agreement = summary["predator_visible"]["exact_action_agreement"]
        lines.append(
            f"| {POLICY_LABELS.get(name, name)} | "
            f"{_percent_interval(overall['exact_action_agreement'])} | "
            f"{_percent_interval(hidden_agreement)} | "
            f"{_percent_interval(visible_agreement)} | "
            f"{_percent_interval(overall['normalized_destination_error'])} | "
            f"{overall['action_distribution_js_divergence_bits']['mean']:.3f} bits |"
        )
    lines.extend(
        (
            "",
            "Agreement is exact action-ID match (higher is better). Destination "
            "error and Jensen–Shannon divergence are lower-is-better. Parentheses "
            "are sample-bootstrap 95% confidence intervals; JS is shown as "
            "a point estimate.",
            "",
            "The source file is a legacy BotEvade simulator-policy transition "
            "export. It should not be described as biological mouse behavior. "
            "One-step alignment also must not be used as a substitute for the "
            "separate closed-loop task and capture results.",
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
    parser.add_argument("--test-data", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--action-catalog", type=Path, required=True)
    parser.add_argument("--direct-predictions", type=Path, required=True)
    parser.add_argument("--mlp-checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--markdown-output", type=Path)
    parser.add_argument("--max-samples", type=int, default=512)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--bootstrap-iterations", type=int, default=2000)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    args = parser.parse_args()

    if args.max_samples <= 0 or args.bootstrap_iterations <= 0:
        parser.error("sample and bootstrap counts must be positive")
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    samples = load_alignment_samples(
        args.test_data, max_samples=args.max_samples, seed=args.seed
    )
    predictions = load_direct_predictions(args.direct_predictions, samples)
    predictions["mlp-bc"] = mlp_predictions(
        args.mlp_checkpoint, samples, device_name=args.device
    )
    destinations = load_action_catalog(args.action_catalog)
    report = build_report(
        samples=samples,
        predictions_by_policy=predictions,
        destinations=destinations,
        manifest=manifest,
        test_data_sha256=_sha256(args.test_data),
        direct_predictions_sha256=_sha256(args.direct_predictions),
        mlp_checkpoint_sha256=_sha256(args.mlp_checkpoint),
        seed=args.seed,
        bootstrap_iterations=args.bootstrap_iterations,
    )
    _atomic_write(
        args.output,
        json.dumps(report, indent=2, sort_keys=True) + "\n",
    )
    if args.markdown_output is not None:
        _atomic_write(args.markdown_output, render_markdown(report))
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
