"""Draw the public four-metric, fixed-seed MouseMind LoRA comparison."""

from __future__ import annotations

import argparse
import html
import json
from pathlib import Path


ROWS = (
    ("Published SFT LoRA", "published_sft", "#315a93"),
    ("Seed-clean SFT LoRA", "seed_clean_sft", "#8a9aad"),
    ("Verified DPO LoRA", "verified_dpo_beta_0_1", "#0b8c79"),
    ("Numeric upper reference", "numeric_upper", "#b77932"),
)
PANELS = (
    ("success_rate", "Task success", 100.0, "Higher is better", "%"),
    ("clean_success_rate", "Clean success", 40.0, "Higher is better", "%"),
    ("capture_rate", "Capture rate", 100.0, "Lower is better", "%"),
    ("captures_per_episode", "Captures / episode", 12.0, "Lower is better", ""),
)


def load_values(original: Path, verified: Path) -> tuple[dict[str, dict[str, float]], int]:
    baseline = json.loads(original.read_text(encoding="utf-8"))
    adaptation = json.loads(verified.read_text(encoding="utf-8"))
    left = baseline["metadata"]
    right = adaptation["evaluation"]
    for key in ("contract_sha256", "seed_sha256", "episode_count", "seed_pool", "planner_horizon", "preference", "ood_condition"):
        if left[key] != right[key]:
            raise ValueError(f"Evaluation mismatch at {key}")
    sources = {
        "published_sft": baseline["policies"]["minimind-learned"],
        "numeric_upper": baseline["policies"]["numeric-learned"],
        "seed_clean_sft": adaptation["policies"]["seed_clean_sft"],
        "verified_dpo_beta_0_1": adaptation["policies"]["verified_dpo_beta_0_1"],
    }
    return {
        key: {metric: float(source[metric]["mean"]) for metric, *_ in PANELS}
        for key, source in sources.items()
    }, int(right["episode_count"])


def draw_svg(values: dict[str, dict[str, float]], episode_count: int) -> str:
    width, height = 1400, 795
    svg = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" role="img" aria-labelledby="title desc">',
        '<title id="title">Verified DPO improves the MiniMind skill-planner LoRA</title>',
        '<desc id="desc">Four metrics on the same 100 final-ID seeds compare the published and seed-clean SFT LoRAs, the Verified DPO LoRA, and a separate numeric upper reference.</desc>',
        '<rect width="1400" height="795" rx="24" fill="#f7f9fc"/>',
        '<text x="48" y="60" font-family="Arial,sans-serif" font-size="34" font-weight="700" fill="#13233b">Same hierarchy, stronger skill-planner LoRA</text>',
        '<text x="48" y="94" font-family="Arial,sans-serif" font-size="18" fill="#46566c">Verified DPO fine-tunes the MiniMind LoRA; the three-skill policy and specialist are unchanged.</text>',
        f'<text x="48" y="125" font-family="Arial,sans-serif" font-size="15" fill="#64748b">{episode_count} paired final-ID seeds · survival-first preference · fixed evaluation protocol</text>',
    ]
    positions = ((40, 155), (710, 155), (40, 455), (710, 455))
    for (metric, title, maximum, direction, suffix), (left, top) in zip(PANELS, positions, strict=True):
        svg.extend([
            f'<rect x="{left}" y="{top}" width="650" height="280" rx="17" fill="#ffffff" stroke="#dbe3ec" stroke-width="1.5"/>',
            f'<text x="{left + 24}" y="{top + 38}" font-family="Arial,sans-serif" font-size="24" font-weight="700" fill="#13233b">{html.escape(title)}</text>',
            f'<text x="{left + 625}" y="{top + 37}" text-anchor="end" font-family="Arial,sans-serif" font-size="14" fill="#64748b">{html.escape(direction)}</text>',
            f'<line x1="{left + 24}" y1="{top + 54}" x2="{left + 626}" y2="{top + 54}" stroke="#e9eef4"/>',
        ])
        bar_left, bar_width = left + 257, 270
        for index, (label, policy, color) in enumerate(ROWS):
            y = top + 88 + index * 49
            if policy == "verified_dpo_beta_0_1":
                svg.append(f'<rect x="{left + 13}" y="{y - 22}" width="624" height="42" rx="9" fill="#eaf7f3"/>')
            value = values[policy][metric]
            scaled = value * 100 if suffix else value
            length = min(max(scaled / maximum, 0.0), 1.0) * bar_width
            shown = f"{scaled:.0f}%" if suffix else f"{scaled:.2f}"
            svg.extend([
                f'<text x="{left + 27}" y="{y + 5}" font-family="Arial,sans-serif" font-size="16" font-weight="{700 if policy == "verified_dpo_beta_0_1" else 400}" fill="#213047">{html.escape(label)}</text>',
                f'<rect x="{bar_left}" y="{y - 10}" width="{bar_width}" height="20" rx="7" fill="#edf1f5"/>',
                f'<rect x="{bar_left}" y="{y - 10}" width="{length:.1f}" height="20" rx="7" fill="{color}"/>',
                f'<text x="{left + 625}" y="{y + 5}" text-anchor="end" font-family="Arial,sans-serif" font-size="17" font-weight="700" fill="{color}">{shown}</text>',
            ])
        svg.append(f'<text x="{bar_left}" y="{top + 267}" font-family="Arial,sans-serif" font-size="12" fill="#7a8898">0</text>')
        svg.append(f'<text x="{bar_left + bar_width}" y="{top + 267}" text-anchor="end" font-family="Arial,sans-serif" font-size="12" fill="#7a8898">{maximum:g}{suffix}</text>')
    svg.extend([
        '<text x="48" y="766" font-family="Arial,sans-serif" font-size="14" fill="#526279">The numeric planner is a separate non-language upper reference. Bars show means; paired uncertainty and training-split caveats are in the results report.</text>',
        '</svg>',
    ])
    return "\n".join(svg) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--original", type=Path, required=True)
    parser.add_argument("--verified", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    values, count = load_values(args.original, args.verified)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(draw_svg(values, count), encoding="utf-8")


if __name__ == "__main__":
    main()
