"""Write an English-only transfer study from verified aggregate reports."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


CORE_POLICIES = (
    "random",
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
    "random": "Random",
    "literal-direct-mlp": "Direct MLP (literal stack)",
    "literal-p1-rule": "P1 rule planner (literal stack)",
    "literal-numeric": "Numeric planner (literal stack)",
    "literal-minimind": "MiniMind planner (literal stack)",
    "aligned-goal-only": "Goal controller (aligned)",
    "aligned-p1-rule": "P1 rule planner (aligned)",
    "aligned-numeric": "Numeric planner (aligned)",
    "aligned-minimind": "MiniMind planner (aligned)",
    "aligned-minimind-no-history": "MiniMind without history (aligned)",
    "aligned-minimind-no-instruction": "MiniMind without instruction (aligned)",
}


def _load_final(path: Path, *, condition: str) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    metadata = payload.get("metadata", {})
    if (
        payload.get("experiment") != "mousemind_frozen_cross_task_transfer"
        or metadata.get("research_evidence") is not True
        or metadata.get("research_evidence_blockers")
        or metadata.get("seed_pool") != "final_test"
        or metadata.get("condition") != condition
        or int(metadata.get("episode_count", 0)) != 100
        or int(metadata.get("full_pool_episode_count", 0)) != 100
    ):
        raise ValueError(f"Not complete final {condition} evidence: {path}")
    return payload


def _mean(summary: dict[str, Any], metric: str) -> float:
    value = summary[metric]
    return float(value["mean"] if isinstance(value, dict) else value)


def _percent(summary: dict[str, Any], metric: str) -> str:
    return f"{100 * _mean(summary, metric):.1f}%"


def _number(summary: dict[str, Any], metric: str) -> str:
    return f"{_mean(summary, metric):.2f}"


def _delta_interval(
    report: dict[str, Any], candidate: str, reference: str, metric: str
) -> tuple[float, float, float]:
    key = f"{candidate}_minus_{reference}"
    value = report["within_mode_paired_comparisons"][key][metric]
    return float(value["mean"]), float(value["ci_low"]), float(value["ci_high"])


def _best_aligned(report: dict[str, Any]) -> str:
    candidates = [
        name
        for name in ("aligned-goal-only", "aligned-p1-rule", "aligned-numeric", "aligned-minimind")
        if name in report["policies"]
    ]
    if not candidates:
        raise ValueError("Final report has no aligned policies")
    return max(
        candidates,
        key=lambda name: (
            _mean(report["policies"][name], "clean_success_rate"),
            -_mean(report["policies"][name], "captures_per_episode"),
            _mean(report["policies"][name], "success_rate"),
        ),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seen-report", type=Path, required=True)
    parser.add_argument("--unseen-report", type=Path, required=True)
    parser.add_argument("--compatibility-audit", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--title", required=True)
    args = parser.parse_args()

    seen = _load_final(args.seen_report, condition="seen_instruction")
    unseen = _load_final(args.unseen_report, condition="unseen_instruction")
    compatibility = json.loads(args.compatibility_audit.read_text(encoding="utf-8"))
    if compatibility.get("research_evidence") is not True:
        raise ValueError("Compatibility audit is not verified")
    if seen["metadata"]["contract_sha256"] != compatibility["contract_sha256"]:
        raise ValueError("Final report and compatibility audit disagree")

    policies = [name for name in CORE_POLICIES if name in seen["policies"]]
    table = [
        "| Policy | Task success | Objective completion | Clean success | Captures / episode | Path efficiency |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for name in policies:
        summary = seen["policies"][name]
        table.append(
            f"| {LABELS[name]} | {_percent(summary, 'success_rate')} | "
            f"{_percent(summary, 'goal_completion_rate')} | "
            f"{_percent(summary, 'clean_success_rate')} | "
            f"{_number(summary, 'captures_per_episode')} | "
            f"{_percent(summary, 'path_efficiency')} |"
        )

    best = _best_aligned(seen)
    goal_only = seen["policies"]["aligned-goal-only"]
    best_summary = seen["policies"][best]
    literal = [name for name in policies if name.startswith("literal-")]
    best_literal_success = max(
        _mean(seen["policies"][name], "success_rate") for name in literal
    )
    learned_lines = []
    for name in ("aligned-p1-rule", "aligned-numeric", "aligned-minimind"):
        if name not in seen["policies"]:
            continue
        clean_mean, clean_low, clean_high = _delta_interval(
            seen, name, "aligned-goal-only", "clean_success_rate"
        )
        capture_mean, capture_low, capture_high = _delta_interval(
            seen, name, "aligned-goal-only", "captures"
        )
        learned_lines.append(
            f"- {LABELS[name]} changed clean success by "
            f"{100 * clean_mean:+.1f} points (95% CI {100 * clean_low:+.1f} to {100 * clean_high:+.1f}) "
            f"and captures by {capture_mean:+.2f} per episode "
            f"({capture_low:+.2f} to {capture_high:+.2f}) "
            "relative to the aligned goal-only controller."
        )

    unseen_lines = []
    for name in (
        "aligned-minimind",
        "aligned-minimind-no-history",
        "aligned-minimind-no-instruction",
    ):
        if name in unseen["policies"]:
            summary = unseen["policies"][name]
            unseen_lines.append(
                f"- {LABELS[name]}: {_percent(summary, 'success_rate')} task success, "
                f"{_percent(summary, 'clean_success_rate')} clean success, "
                f"{_number(summary, 'captures_per_episode')} captures per episode."
            )

    if best == "aligned-goal-only":
        decision = (
            "The target-task interface restored completion, but none of the frozen "
            "strategic planners improved the primary clean-success objective over "
            "the aligned goal-only controller."
        )
    else:
        decision = (
            f"{LABELS[best]} was the strongest aligned transfer system by clean "
            "success, with the goal-only controller retained as the paired reference."
        )

    lines = [
        f"# {args.title}",
        "",
        "## Research question",
        "",
        "Do high-level strategies learned in single-goal BotEvade transfer to the "
        "ordered multi-goal Oasis task, or are the P2 gains tied to the original "
        "low-level interface and task geometry?",
        "",
        "## Frozen evaluation contract",
        "",
        "- Source: BotEvade `21_05`; target: discrete-compatible Oasis `21_05`.",
        "- Final evidence: 100 untouched paired seeds, three sampled ordered goals, "
        "then return to start.",
        "- No target training, target adaptation, threshold selection, or final-seed tuning.",
        "- Literal transfer reuses the frozen P2 planner and 10D specialist unchanged.",
        "- Planner-isolation transfer freezes the high-level planner and gives every "
        "planner the same parameter-free active-goal controller.",
        "",
        "## Main result",
        "",
        decision,
        "",
        "![Frozen cross-task transfer](mouse_llm/reports/figures/transfer_boundary.png)",
        "",
        "![Predetermined final-seed rollout](mouse_llm/reports/figures/transfer_rollout.gif)",
        "",
        "The animation uses the predetermined first final seed (`42000`) as a qualitative "
        "illustration; all claims and confidence intervals come from the 100-seed aggregate.",
        "",
        *table,
        "",
        "## What the compatibility audit changed",
        "",
        f"- BotEvade and Oasis share all 295 action destinations exactly "
        f"(`{compatibility['action_contract']['coordinate_sha256'][:12]}…`).",
        "- The original Oasis wrapper did not provide deterministic seeded resets and "
        "conflated task success with zero-capture survival; both contracts are now explicit and tested.",
        "- One default Oasis goal was 0.046875 from its nearest discrete action, beyond "
        "the 0.025 completion threshold. The frozen discrete-compatible contract projects "
        "that goal once before evaluation and records the change.",
        "- The frozen BotEvade 10D specialist observes goal distance but not active goal "
        "coordinates. Literal full-stack transfer is therefore reported as an intentional "
        "fail-closed baseline, not as a valid planner-transfer test.",
        "",
        "## Transfer findings",
        "",
        f"- The best literal full-stack task success was {100 * best_literal_success:.1f}%.",
        f"- The strongest aligned system was {LABELS[best]} at "
        f"{_percent(best_summary, 'success_rate')} task success, "
        f"{_percent(best_summary, 'clean_success_rate')} clean success, and "
        f"{_number(best_summary, 'captures_per_episode')} captures per episode.",
        f"- The aligned goal-only reference reached {_percent(goal_only, 'success_rate')} "
        f"task success and {_percent(goal_only, 'clean_success_rate')} clean success.",
        *learned_lines,
        "",
        "## Unseen-instruction ablations",
        "",
        *unseen_lines,
        "",
        "## Interpretation and limitations",
        "",
        "This experiment separates interface compatibility from strategic transfer. "
        "Restoring access to the active target can recover task completion without proving "
        "that the learned safety strategy transferred. Task success and clean success remain "
        "separate, and no policy is described as safe from completion alone.",
        "",
        "The study uses one public Cellworld geometry with a new multi-goal task, not a "
        "new physical world. Cross-geometry transfer remains unsupported until its action "
        "and geometry contracts are verified independently.",
        "",
        "## Resume-ready summary",
        "",
        f"> Built a fail-closed BotEvade-to-Oasis transfer study over 100 untouched "
        f"paired seeds; identified and repaired seeded-reset, terminal-semantics, and "
        f"discrete-goal contract defects; separated literal full-stack transfer from "
        f"planner-isolation transfer; and showed that the best literal stack reached "
        f"{100 * best_literal_success:.1f}% task success while {LABELS[best]} reached "
        f"{_percent(best_summary, 'success_rate')} task / "
        f"{_percent(best_summary, 'clean_success_rate')} clean success under the "
        f"verified target interface.",
        "",
    ]
    args.output.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
