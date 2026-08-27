"""Reproduce the four Git-safe P2 portfolio figures from aggregate JSON only."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


COLORS = {
    "direct-mlp": "#8c98a4",
    "p1-rule": "#e69f00",
    "numeric-learned": "#4c78a8",
    "numeric-verified": "#2ca25f",
    "minimind-learned": "#8b5cf6",
    "minimind-verified": "#0f9d76",
    "minimind-no-history": "#b794f4",
    "minimind-no-instruction": "#d6bcfa",
}
CORE_POLICIES = (
    "direct-mlp",
    "p1-rule",
    "numeric-learned",
    "numeric-verified",
    "minimind-learned",
    "minimind-verified",
)
SAFETY_LABEL_OFFSETS = {
    "numeric-learned": (8, -22),
    "numeric-verified": (-112, -22),
    "minimind-learned": (6, 8),
    "minimind-verified": (6, 8),
    "p1-rule": (6, 6),
    "direct-mlp": (6, 6),
}
PLANNER_ABLATION_POLICIES = (
    *CORE_POLICIES,
    "minimind-no-history",
    "minimind-no-instruction",
)


def _load(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("experiment") != "mousemind_seeded_closed_loop":
        raise ValueError(f"Not a closed-loop aggregate report: {path}")
    return payload


def _mean(summary: dict[str, Any], metric: str) -> float:
    value = summary[metric]
    return float(value["mean"] if isinstance(value, dict) else value)


def _interval(summary: dict[str, Any], metric: str) -> tuple[float, float, float]:
    value = summary[metric]
    return (
        float(value["mean"]),
        float(value["ci_low"]),
        float(value["ci_high"]),
    )


def _save(fig: Any, path: Path) -> None:
    fig.savefig(path, dpi=180, bbox_inches="tight", facecolor="white")
    fig.savefig(path.with_suffix(".svg"), bbox_inches="tight", facecolor="white")


def safety_frontier(
    report: dict[str, Any], frontier: list[tuple[float, dict[str, Any]]], output: Path
) -> None:
    import matplotlib.pyplot as plt

    fig, axis = plt.subplots(figsize=(8.6, 5.4))
    for policy, summary in report["policies"].items():
        if policy not in CORE_POLICIES:
            continue
        x, x_low, x_high = _interval(summary, "clean_success_rate")
        y, y_low, y_high = _interval(summary, "success_rate")
        x, x_low, x_high = 100 * x, 100 * x_low, 100 * x_high
        y, y_low, y_high = 100 * y, 100 * y_low, 100 * y_high
        axis.errorbar(
            x,
            y,
            xerr=[[x - x_low], [x_high - x]],
            yerr=[[y - y_low], [y_high - y]],
            fmt="o",
            markersize=9,
            capsize=3,
            linewidth=1.2,
            color=COLORS.get(policy, "#555555"),
            zorder=3,
        )
        axis.annotate(
            policy.replace("-", " "),
            (x, y),
            xytext=SAFETY_LABEL_OFFSETS[policy],
            textcoords="offset points",
            fontsize=9,
        )
    points = []
    for threshold, payload in frontier:
        summary = payload["policies"].get("numeric-verified") or payload["policies"].get("minimind-verified")
        if summary is None:
            continue
        points.append(
            (
                100 * _mean(summary, "clean_success_rate"),
                100 * _mean(summary, "success_rate"),
                threshold,
            )
        )
    if points:
        points.sort(key=lambda item: item[2])
        axis.plot(
            [item[0] for item in points],
            [item[1] for item in points],
            color="#2ca25f",
            alpha=0.65,
            linewidth=2,
            marker="o",
            label="verifier threshold sweep (development)",
        )
    axis.set_xlabel("Clean success (%)  → safer completion")
    axis.set_ylabel("Task success (%)")
    axis.set_title("Offline imitation did not predict closed-loop control")
    axis.grid(alpha=0.2)
    axis.set_xlim(left=0)
    axis.set_ylim(bottom=0)
    if points:
        axis.legend(frameon=False, loc="lower right")
    fig.tight_layout()
    _save(fig, output)
    plt.close(fig)


def ood_generalization(
    reports: list[tuple[str, dict[str, Any]]], output: Path
) -> None:
    import matplotlib.pyplot as plt

    common = set(reports[0][1]["policies"])
    for _, report in reports[1:]:
        common &= set(report["policies"])
    policies = [
        name for name in CORE_POLICIES if name in common and name != "direct-mlp"
    ]
    fig, axis = plt.subplots(figsize=(9.0, 5.2))
    x = list(range(len(reports)))
    for policy in policies:
        intervals = [
            _interval(report["policies"][policy], "clean_success_rate")
            for _, report in reports
        ]
        values = [100 * value[0] for value in intervals]
        low = [100 * value[1] for value in intervals]
        high = [100 * value[2] for value in intervals]
        axis.errorbar(
            x,
            values,
            yerr=[
                [value - lower for value, lower in zip(values, low, strict=True)],
                [upper - value for value, upper in zip(values, high, strict=True)],
            ],
            marker="o",
            linewidth=2.2,
            capsize=3,
            label=policy.replace("-", " "),
            color=COLORS.get(policy),
        )
    axis.set_xticks(x, [name.replace("_", "\n") for name, _ in reports])
    axis.set_ylabel("Clean success (%)")
    axis.set_title("Fresh ID and out-of-distribution robustness")
    axis.grid(axis="y", alpha=0.2)
    axis.set_ylim(bottom=0)
    axis.legend(frameon=False, ncol=2)
    fig.tight_layout()
    _save(fig, output)
    plt.close(fig)


def latency_tradeoff(report: dict[str, Any], output: Path) -> None:
    import matplotlib.pyplot as plt

    fig, axis = plt.subplots(figsize=(7.8, 5.0))
    for policy, summary in report["policies"].items():
        latency = 1000 * float(summary["latency_seconds"]["p95"])
        clean = 100 * _mean(summary, "clean_success_rate")
        axis.scatter(latency, clean, s=100, color=COLORS.get(policy, "#777777"))
        axis.annotate(policy.replace("-", " "), (latency, clean), xytext=(5, 5), textcoords="offset points", fontsize=8)
    axis.set_xlabel("p95 end-to-end policy latency (ms, log scale)")
    axis.set_ylabel("Clean success (%)")
    axis.set_xscale("log")
    axis.set_title("Control quality versus inference cost")
    axis.grid(alpha=0.2)
    fig.tight_layout()
    _save(fig, output)
    plt.close(fig)


def planner_ablation(report: dict[str, Any], output: Path) -> None:
    import matplotlib.pyplot as plt
    import numpy as np

    policies = [
        name for name in PLANNER_ABLATION_POLICIES if name in report["policies"]
    ]
    x = np.arange(len(policies))
    width = 0.25
    task = [100 * _mean(report["policies"][name], "success_rate") for name in policies]
    clean = [100 * _mean(report["policies"][name], "clean_success_rate") for name in policies]
    capture_free = [100 * (1 - _mean(report["policies"][name], "capture_rate")) for name in policies]
    fig, axis = plt.subplots(figsize=(9.2, 5.2))
    axis.bar(x - width, task, width, label="task success", color="#4c78a8")
    axis.bar(x, clean, width, label="clean success", color="#2ca25f")
    axis.bar(x + width, capture_free, width, label="capture-free", color="#f2a65a")
    axis.set_xticks(x, [name.replace("-", "\n") for name in policies])
    axis.set_ylabel("Episodes (%)")
    axis.set_title("Planner and verifier ablations on paired seeds")
    axis.legend(frameon=False, ncol=3)
    axis.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    _save(fig, output)
    plt.close(fig)


def _named_path(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("expected NAME=PATH")
    name, path = value.split("=", 1)
    return name, Path(path)


def _threshold_path(value: str) -> tuple[float, Path]:
    name, path = _named_path(value)
    return float(name), path


def main() -> None:
    parser = argparse.ArgumentParser(description="Render MouseMind P2 aggregate figures")
    parser.add_argument("--id-report", type=Path, required=True)
    parser.add_argument("--ood-report", action="append", type=_named_path, default=[])
    parser.add_argument("--frontier-report", action="append", type=_threshold_path, default=[])
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    identity = _load(args.id_report)
    frontier = [(threshold, _load(path)) for threshold, path in args.frontier_report]
    ood = [("ID", identity), *((name, _load(path)) for name, path in args.ood_report)]
    safety_frontier(identity, frontier, args.output_dir / "p2_safety_frontier.png")
    ood_generalization(ood, args.output_dir / "p2_ood_generalization.png")
    latency_tradeoff(identity, args.output_dir / "p2_latency_tradeoff.png")
    planner_ablation(identity, args.output_dir / "p2_planner_ablation.png")


if __name__ == "__main__":
    main()
