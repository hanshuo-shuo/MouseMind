"""Write a compact paired SFT versus Verified DPO result from frozen P2 reports."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

from mouse_llm.evaluation.closed_loop import bootstrap_mean


METRICS = (
    ("task success", "success", "%"),
    ("clean success", "clean_success", "%"),
    ("capture rate", "captured", "%"),
    ("captures / episode", "captures", ""),
)


def load_run(directory: Path) -> tuple[dict, dict[int, dict]]:
    report = json.loads((directory / "closed_loop_metrics.json").read_text())
    with (directory / "closed_loop_episodes.csv").open(newline="") as handle:
        rows = [row for row in csv.DictReader(handle) if row["policy"] == "minimind-learned"]
    by_seed = {int(row["seed"]): row for row in rows}
    if len(by_seed) != 100 or len(rows) != 100:
        raise ValueError(f"Expected exactly 100 distinct final seeds: {directory}")
    return report, by_seed


def fmt(value: float, unit: str) -> str:
    return f"{value * 100:.1f}%" if unit == "%" else f"{value:.2f}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--dpo-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--baseline-label", default="Current SFT")
    args = parser.parse_args()
    runs = [(args.baseline_label, args.baseline)] + [
        (f"DPO β={beta}", args.dpo_root / f"beta_{beta}" / "final_id")
        for beta in ("0.1", "0.05", "0.2")
    ]
    loaded = [(label, *load_run(path)) for label, path in runs]
    baseline_report, baseline_rows = loaded[0][1:]
    baseline_meta = baseline_report["metadata"]
    sft_weight = args.baseline.parent / "minimind/lora_mousemind_skill_planner_768.pth"
    sft_sha = hashlib.sha256(sft_weight.read_bytes()).hexdigest() if sft_weight.exists() else "unavailable"
    for label, report, rows in loaded[1:]:
        meta = report["metadata"]
        for key in ("contract_sha256", "seed_sha256", "episode_count", "planner_horizon", "risk_threshold", "ood_condition", "preference", "instruction_split"):
            if meta[key] != baseline_meta[key]:
                raise ValueError(f"{label}: evaluation metadata differs at {key}")
        if set(rows) != set(baseline_rows):
            raise ValueError(f"{label}: final seeds differ")
    lines = [
        "# Verified DPO results", "",
        "## Configuration", "",
        "- Source: 320 verified exact-state P2 collection anchors, horizon 8; one best-versus-worst pair per anchor and preference when utility margin is positive.",
        "- The original anchor-ID split shared collection seeds across train/validation/test. For strict seed isolation, this experiment regrouped all 320 anchors by collection seed (69 train, 6 validation, 5 test seeds), removed zero-margin pairs, and retrained SFT on the same 1,072 chosen responses used by DPO. The original best SFT checkpoint was not used as initialization because it had seen the held-out collection seeds.",
        "- DPO: this seed-clean SFT LoRA initializes both the trainable policy and frozen reference; base MiniMind is frozen. β = 0.1, 0.05, 0.2; one epoch; 67 LoRA optimizer updates per run.",
        "- Evaluation: frozen P2 final ID contract, 100 untouched paired seeds, survival-first preference, planner horizon 8, the same specialist and constrained three-skill JSON output.",
        f"- Contract SHA-256: `{baseline_meta['contract_sha256']}`; seed SHA-256: `{baseline_meta['seed_sha256']}`; seed-clean SFT LoRA SHA-256: `{sft_sha}`.",
        "", "## Closed-loop results", "",
        "| Policy | Task success | Clean success | Capture rate | Captures / episode |", "| --- | ---: | ---: | ---: | ---: |",
    ]
    for label, report, _ in loaded:
        summary = report["policies"]["minimind-learned"]
        values = [summary[key]["mean"] for key in ("success_rate", "clean_success_rate", "capture_rate", "captures_per_episode")]
        lines.append(f"| {label} | {fmt(values[0], '%')} | {fmt(values[1], '%')} | {fmt(values[2], '%')} | {fmt(values[3], '')} |")
    lines += ["", "## Paired DPO − SFT", "", "Mean difference and 95% paired bootstrap CI across the same 100 seeds.", "", "| Policy | Metric | Delta | 95% CI |", "| --- | --- | ---: | ---: |"]
    for label, _, rows in loaded[1:]:
        for name, key, unit in METRICS:
            differences = [float(rows[seed][key]) - float(baseline_rows[seed][key]) for seed in sorted(rows)]
            estimate = bootstrap_mean(differences, seed=42)
            scale = 100 if unit == "%" else 1
            suffix = " pp" if unit == "%" else ""
            lines.append(f"| {label} | {name} | {estimate['mean'] * scale:+.2f}{suffix} | [{estimate['ci_low'] * scale:+.2f}, {estimate['ci_high'] * scale:+.2f}]{suffix} |")
    lines += ["", "## Offline skill accuracy", ""]
    for label, path in runs:
        offline = path.parent / "offline_metrics.json"
        if not offline.exists():
            continue
        model = json.loads(offline.read_text())["models"]["minimind_skill_lora"]
        lines.append(f"- {label}: seen instruction {model['seen_instruction']['accuracy']:.3f}; unseen paraphrase {model['unseen_paraphrase']['accuracy']:.3f}.")
    diagnostic = args.dpo_root / "beta_0.1" / "diagnostic.json"
    if diagnostic.exists():
        groups = json.loads(diagnostic.read_text())["groups"]
        lines += ["", "## Exact-state final-rollout diagnostic", ""]
        for group in ("all", "predator_visible", "near_occlusion"):
            values = groups.get(group, {})
            lines.append(f"- {group}: {values.get('changed', 0)} / {values.get('decisions', 0)} decisions changed; SFT wrong → DPO correct {values.get('sft_wrong_dpo_correct', 0)}; SFT correct → DPO wrong {values.get('sft_correct_dpo_wrong', 0)}; both wrong {values.get('both_wrong', 0)}; replay unverified {values.get('unverified_changed', 0)}.")
        all_changes = groups.get("all", {})
        lines.append(f"Only {all_changes.get('changed', 0) - all_changes.get('unverified_changed', 0)} / {all_changes.get('changed', 0)} changed decisions passed deterministic replay. The oracle-correction counts apply only to this verified subset and cannot explain the overall closed-loop improvement.")
    base_metrics = baseline_report["policies"]["minimind-learned"]
    dpo_metrics = loaded[1][1]["policies"]["minimind-learned"]
    lines += [
        "", "## Conclusion", "",
        f"On the same 100 untouched final seeds, β=0.1 changed task success from {fmt(base_metrics['success_rate']['mean'], '%')} to {fmt(dpo_metrics['success_rate']['mean'], '%')}, clean success from {fmt(base_metrics['clean_success_rate']['mean'], '%')} to {fmt(dpo_metrics['clean_success_rate']['mean'], '%')}, and captures per episode from {fmt(base_metrics['captures_per_episode']['mean'], '')} to {fmt(dpo_metrics['captures_per_episode']['mean'], '')}. This supports a closed-loop safety gain without sacrificing task success in this frozen ID protocol.",
        "The β=0.05 run shows a similar but smaller safety gain; β=0.2 is inconclusive by the paired intervals. β=0.1 has lower offline seen and unseen-paraphrase accuracy than seed-clean SFT, so offline skill accuracy did not predict the closed-loop safety change.",
        "This controlled comparison uses a retrained seed-clean SFT baseline. The previously published SFT LoRA is a historical reference, not the initialization here. The result has one fixed final seed pool and should not be read as a general safety guarantee.", "",
    ]
    args.output.write_text("\n".join(lines))
    print(args.output)


if __name__ == "__main__":
    main()
