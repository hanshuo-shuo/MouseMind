"""Write P2_RESULTS.md directly from completed aggregate evidence artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _named(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("expected NAME=PATH")
    name, path = value.split("=", 1)
    return name, Path(path)


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _mean(summary: dict[str, Any], metric: str) -> float:
    value = summary[metric]
    return float(value["mean"] if isinstance(value, dict) else value)


def _percent(value: float) -> str:
    return f"{100 * value:.1f}%"


def _policy_row(name: str, summary: dict[str, Any]) -> str:
    return (
        f"| {name} | {_percent(_mean(summary, 'clean_success_rate'))} | "
        f"{_percent(_mean(summary, 'success_rate'))} | "
        f"{_percent(_mean(summary, 'capture_rate'))} | "
        f"{_mean(summary, 'captures'):.2f} |"
    )


def _paired_delta(
    report: dict[str, Any], candidate: str, reference: str, metric: str
) -> dict[str, float]:
    direct_key = f"{candidate}_minus_{reference}"
    reverse_key = f"{reference}_minus_{candidate}"
    comparisons = report["paired_comparisons"]
    if direct_key in comparisons:
        return comparisons[direct_key][metric]
    if reverse_key in comparisons:
        value = comparisons[reverse_key][metric]
        return {
            "mean": -float(value["mean"]),
            "ci_low": -float(value["ci_high"]),
            "ci_high": -float(value["ci_low"]),
        }
    raise ValueError(f"Missing paired comparison for {candidate} and {reference}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate evidence-backed P2_RESULTS.md")
    parser.add_argument("--id-report", type=Path, required=True)
    parser.add_argument("--ood-report", action="append", type=_named, default=[])
    parser.add_argument("--numeric-offline", type=Path, required=True)
    parser.add_argument("--minimind-offline", type=Path, required=True)
    parser.add_argument("--risk-report", type=Path, required=True)
    parser.add_argument("--counterfactual-report", type=Path, required=True)
    parser.add_argument("--corrective-report", type=Path)
    parser.add_argument("--operating-point", type=Path)
    parser.add_argument("--selected-policy", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    identity = _load(args.id_report)
    numeric = _load(args.numeric_offline)
    minimind = _load(args.minimind_offline)
    risk = _load(args.risk_report)
    counterfactual = _load(args.counterfactual_report)
    ood = [(name, _load(path)) for name, path in args.ood_report]
    if identity.get("metadata", {}).get("research_evidence") is not True:
        raise ValueError("ID report is not completed research evidence")
    for name, report in ood:
        if report.get("metadata", {}).get("research_evidence") is not True:
            raise ValueError(f"OOD report {name} is not completed research evidence")
    if args.selected_policy not in identity["policies"]:
        raise ValueError("Selected policy is missing from the ID report")
    selected = identity["policies"][args.selected_policy]
    p1 = identity["policies"]["p1-rule"]
    clean_delta = _mean(selected, "clean_success_rate") - _mean(
        p1, "clean_success_rate"
    )
    task_delta = _mean(selected, "success_rate") - _mean(p1, "success_rate")
    capture_delta = _mean(selected, "capture_rate") - _mean(p1, "capture_rate")
    risk_validation = risk["validation"]
    numeric_validation = numeric["validation"]
    minimind_models = minimind["models"]
    lora_unseen = minimind_models["minimind_skill_lora"]["unseen_paraphrase"]
    corrective_text = "Not run."
    if args.corrective_report is not None:
        corrective = _load(args.corrective_report)
        before = corrective["closed_loop_before"]
        after = corrective["closed_loop_after"]
        corrective_text = (
            f"Using {corrective['data']['selected_failure_episodes']} development "
            f"failures, P2.1 added {corrective['data']['corrective_anchors']} anchors "
            f"and {corrective['data']['verified_branches']} verified branches. "
            f"Clean success changed from {_percent(before['clean_success_rate'])} to "
            f"{_percent(after['clean_success_rate'])}, capture rate from "
            f"{_percent(before['capture_rate'])} to {_percent(after['capture_rate'])}, "
            f"and captures/episode from {before['captures_per_episode']:.2f} to "
            f"{after['captures_per_episode']:.2f}. The iteration was rejected for final."
        )
    operating_text = "Development operating point was not supplied."
    if args.operating_point is not None:
        operating = _load(args.operating_point)
        best_threshold = max(
            operating["threshold_sweep"],
            key=lambda row: row["clean_success_rate"],
        )
        unverified_clean = (
            best_threshold["clean_success_rate"]
            - best_threshold["clean_success_delta_vs_unverified"]
        )
        operating_text = (
            f"The best swept verifier threshold was {operating['selected_risk_threshold']:.1f}, "
            f"but development clean success changed from {_percent(unverified_clean)} "
            f"to {_percent(best_threshold['clean_success_rate'])} and capture rate moved "
            f"in the wrong direction. Status: {operating['verifier_deployment_status']}."
        )
    paired_clean = _paired_delta(
        identity, args.selected_policy, "p1-rule", "clean_success_rate"
    )
    paired_task = _paired_delta(
        identity, args.selected_policy, "p1-rule", "success_rate"
    )
    paired_capture = _paired_delta(
        identity, args.selected_policy, "p1-rule", "capture_rate"
    )
    lines = [
        "# MouseMind P2 results",
        "",
        "## 1. What was implemented",
        "",
        "A frozen fresh-seed contract, semantic temporal planner context, exact seeded replay, counterfactual branching for all three skills, outcome-grounded language targets, a numeric planner, MiniMind skill LoRA, a calibrated capture-risk critic, propose-verify control, development-only operating-point selection, OOD evaluation, and one corrective-data path.",
        "",
        "## 2. Exact experiments run",
        "",
        f"- Counterfactual anchors: {counterfactual['anchor_count']} anchors and {counterfactual['branch_count']} verified branches over horizons {counterfactual['horizons']}.",
        f"- Numeric planner validation: {numeric_validation['sample_count']} samples.",
        f"- MiniMind unseen-paraphrase test: {lora_unseen['sample_count']} samples.",
        f"- Fresh ID: {identity['metadata']['episode_count']} paired seeds from `{identity['metadata']['seed_pool']}`.",
        f"- OOD conditions: {', '.join(name for name, _ in ood) or 'none'}.",
        "",
        "## 3. Verified versus pending",
        "",
        "All numbers below come from aggregate artifacts marked `research_evidence=true`. Private states, trajectories, predictions, and episode CSV files remain outside Git. Historical P1 clean success is pending because its joint per-episode success/capture records were unavailable; no value was inferred.",
        "",
        "## 4. Main fresh-ID results",
        "",
        "| Policy | Clean success | Task success | Capture rate | Captures / episode |",
        "| --- | ---: | ---: | ---: | ---: |",
        *(_policy_row(name, summary) for name, summary in identity["policies"].items()),
        "",
        "## 5. Main OOD results",
        "",
        "| Condition | Clean success | Task success | Capture rate |",
        "| --- | ---: | ---: | ---: |",
        *(
            f"| {name} | {_percent(_mean(report['policies'][args.selected_policy], 'clean_success_rate'))} | {_percent(_mean(report['policies'][args.selected_policy], 'success_rate'))} | {_percent(_mean(report['policies'][args.selected_policy], 'capture_rate'))} |"
            for name, report in ood
        ),
        "",
        "## 6. Risk-model calibration",
        "",
        f"Validation AUROC: {risk_validation['auroc']}; AUPRC: {risk_validation['auprc']}; Brier: {risk_validation['brier_score']:.4f}; ECE: {risk_validation['ece']:.4f}. {operating_text}",
        "",
        "## 7. Ablation conclusions",
        "",
        f"Numeric planner validation accuracy / macro F1: {numeric_validation['accuracy']:.3f} / {numeric_validation['macro_f1']:.3f}. MiniMind unseen-paraphrase accuracy / macro F1: {lora_unseen['accuracy']:.3f} / {lora_unseen['macro_f1']:.3f}. Full MiniMind no-history and instruction-removed ablations are stored in the offline planner report.",
        "",
        "## 8. Failure modes that remain",
        "",
        f"Selected-policy aggregate taxonomy: `{json.dumps(selected['failure_taxonomy']['counts'], sort_keys=True)}`.",
        "",
        "## 9. Does P2 beat P1?",
        "",
        f"For `{args.selected_policy}`, candidate minus P1 is {paired_clean['mean']:+.3f} clean success (95% CI {paired_clean['ci_low']:+.3f} to {paired_clean['ci_high']:+.3f}), {paired_task['mean']:+.3f} task success ({paired_task['ci_low']:+.3f} to {paired_task['ci_high']:+.3f}), and {paired_capture['mean']:+.3f} capture rate ({paired_capture['ci_low']:+.3f} to {paired_capture['ci_high']:+.3f}). Interpret improvement under the metric named; do not call the policy safe from task success alone.",
        "",
        "## P2.1 corrective-data iteration",
        "",
        corrective_text,
        "",
        "## 10. Strongest supported resume bullet",
        "",
        f"Built a counterfactual learned control hierarchy and evaluated calibrated runtime risk overrides; on {identity['metadata']['episode_count']} untouched paired ID seeds, the selected `{args.selected_policy}` reached {_percent(_mean(selected, 'success_rate'))} task success and {_percent(_mean(selected, 'clean_success_rate'))} clean success ({clean_delta * 100:+.1f} points versus the P1 rule hierarchy), with fresh OOD evaluation across {len(ood)} shifts; verifier overrides were rejected when they worsened closed-loop clean success despite strong offline AUROC.",
        "",
    ]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
