"""Render the frozen cross-task transfer result from aggregate JSON only."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


POLICY_ORDER = (
    "literal-direct-mlp",
    "literal-p1-rule",
    "literal-numeric",
    "literal-minimind",
    "aligned-goal-only",
    "aligned-p1-rule",
    "aligned-numeric",
    "aligned-minimind",
)
LABELS = {
    "literal-direct-mlp": "Direct MLP\n(literal)",
    "literal-p1-rule": "P1 rule\n(literal)",
    "literal-numeric": "Numeric\n(literal)",
    "literal-minimind": "MiniMind\n(literal)",
    "aligned-goal-only": "Goal only\n(aligned)",
    "aligned-p1-rule": "P1 rule\n(aligned)",
    "aligned-numeric": "Numeric\n(aligned)",
    "aligned-minimind": "MiniMind\n(aligned)",
}
COLORS = {
    "literal-direct-mlp": "#9CA3AF",
    "literal-p1-rule": "#F59E0B",
    "literal-numeric": "#60A5FA",
    "literal-minimind": "#A78BFA",
    "aligned-goal-only": "#64748B",
    "aligned-p1-rule": "#D97706",
    "aligned-numeric": "#2563EB",
    "aligned-minimind": "#7C3AED",
}


def _load(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if (
        payload.get("experiment") != "mousemind_frozen_cross_task_transfer"
        or payload.get("metadata", {}).get("research_evidence") is not True
        or payload.get("metadata", {}).get("seed_pool") != "final_test"
    ):
        raise ValueError(f"Not a complete final transfer report: {path}")
    return payload


def _interval(summary: dict[str, Any], metric: str) -> tuple[float, float, float]:
    value = summary[metric]
    return float(value["mean"]), float(value["ci_low"]), float(value["ci_high"])


def render(report: dict[str, Any], *, output: Path, title: str) -> None:
    import matplotlib.pyplot as plt
    import numpy as np

    policies = [name for name in POLICY_ORDER if name in report["policies"]]
    x = np.arange(len(policies))
    fig, (quality, captures_axis) = plt.subplots(
        1, 2, figsize=(13.2, 5.6), gridspec_kw={"width_ratios": (1.55, 1.0)}
    )

    width = 0.36
    success = [
        100 * _interval(report["policies"][name], "success_rate")[0]
        for name in policies
    ]
    completion = [
        100 * _interval(report["policies"][name], "goal_completion_rate")[0]
        for name in policies
    ]
    quality.bar(x - width / 2, success, width, color="#2563EB", label="task success")
    quality.bar(
        x + width / 2,
        completion,
        width,
        color="#14B8A6",
        label="objective completion",
    )
    quality.set_xticks(x, [LABELS[name] for name in policies], fontsize=8)
    quality.set_ylim(0, 108)
    quality.set_ylabel("Episodes / objectives (%)")
    quality.set_title("Completion exposes the interface boundary")
    quality.grid(axis="y", alpha=0.18)
    quality.legend(frameon=False, loc="upper left")

    capture_intervals = [
        _interval(report["policies"][name], "captures_per_episode")
        for name in policies
    ]
    capture_means = np.asarray([value[0] for value in capture_intervals])
    capture_low = np.asarray([value[1] for value in capture_intervals])
    capture_high = np.asarray([value[2] for value in capture_intervals])
    captures_axis.bar(
        x,
        capture_means,
        color=[COLORS[name] for name in policies],
        yerr=[capture_means - capture_low, capture_high - capture_means],
        capsize=3,
    )
    captures_axis.set_xticks(x, [LABELS[name] for name in policies], fontsize=8)
    captures_axis.set_ylabel("Captures per episode (log scale, lower is better)")
    captures_axis.set_yscale("symlog", linthresh=1.0)
    captures_axis.set_title("Task completion does not imply safe transfer")
    captures_axis.grid(axis="y", alpha=0.18)

    for axis in (quality, captures_axis):
        literal_count = sum(name.startswith("literal-") for name in policies)
        if literal_count and literal_count < len(policies):
            axis.axvline(literal_count - 0.5, color="#CBD5E1", linewidth=1.5)
    fig.suptitle(title, fontsize=16, fontweight="bold")
    fig.text(
        0.5,
        0.01,
        "Frozen planners · 100 untouched paired Oasis seeds · 95% bootstrap confidence intervals",
        ha="center",
        fontsize=9,
        color="#475569",
    )
    fig.tight_layout(rect=(0, 0.04, 1, 0.95))
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=190, bbox_inches="tight", facecolor="white")
    fig.savefig(output.with_suffix(".svg"), bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--title", default="MouseMind: Frozen Strategy Transfer Across Tasks"
    )
    args = parser.parse_args()
    render(_load(args.report), output=args.output, title=args.title)


if __name__ == "__main__":
    main()
