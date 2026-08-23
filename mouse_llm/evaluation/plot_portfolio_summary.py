"""Render the Git-safe three-policy offline portfolio summary."""

from __future__ import annotations

import argparse
import html
import json
from pathlib import Path


POLICIES = (
    ("Base MiniMind", "#6B7280"),
    ("MiniMind LoRA", "#2563EB"),
    ("MLP BC", "#059669"),
)
PANELS = (
    ("exact_action_accuracy", "Exact action accuracy", "Higher is better"),
    (
        "normalized_destination_error",
        "Destination error",
        "Lower is better · % of arena diagonal",
    ),
)
TEXT = "#111827"
MUTED = "#4B5563"
GRID = "#D1D5DB"


def load_values(lora_path: Path, mlp_path: Path):
    lora = json.loads(lora_path.read_text(encoding="utf-8"))
    mlp = json.loads(mlp_path.read_text(encoding="utf-8"))
    if not lora.get("research_evidence") or not mlp.get("research_evidence"):
        raise ValueError("Portfolio figures require research_evidence=true inputs")
    if lora["evaluation"]["sample_count"] != mlp["evaluation"]["sample_count"]:
        raise ValueError("Policy metrics must use the same sample count")
    values = {}
    for key, _, _ in PANELS:
        mlp_metric = mlp["test"][key]
        values[key] = (
            lora["base"][key],
            lora["lora"][key],
            mlp_metric,
        )
    return values, int(lora["evaluation"]["sample_count"])


def _font(size: int, *, bold: bool = False):
    from PIL import ImageFont

    filename = "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"
    try:
        return ImageFont.truetype(filename, size=size)
    except OSError:
        return ImageFont.load_default()


def render_png(values, sample_count: int, output: Path) -> None:
    from PIL import Image, ImageDraw

    width, height = 1500, 720
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    draw.text(
        (width / 2, 35),
        "MouseMind Offline Policy Baselines",
        fill=TEXT,
        font=_font(34, bold=True),
        anchor="ma",
    )
    draw.text(
        (width / 2, 79),
        "Same deterministic held-out subset · 95% bootstrap confidence intervals",
        fill=MUTED,
        font=_font(17),
        anchor="ma",
    )
    panel_width = 620
    for panel_index, (key, title, subtitle) in enumerate(PANELS):
        left = 75 + panel_index * 720
        right = left + panel_width
        top, bottom = 190, 560
        draw.text(
            ((left + right) / 2, 125),
            title,
            fill=TEXT,
            font=_font(23, bold=True),
            anchor="ma",
        )
        draw.text(
            ((left + right) / 2, 158),
            subtitle,
            fill=MUTED,
            font=_font(15),
            anchor="ma",
        )
        for tick in range(5):
            y = bottom - tick / 4 * (bottom - top)
            draw.line((left + 55, y, right - 15, y), fill=GRID, width=1)
            draw.text(
                (left + 45, y),
                f"{tick * 25}",
                fill=MUTED,
                font=_font(13),
                anchor="rm",
            )
        draw.line((left + 55, top, left + 55, bottom), fill=TEXT, width=2)
        draw.line((left + 55, bottom, right - 15, bottom), fill=TEXT, width=2)
        centers = [left + 155, left + 335, left + 515]
        for center, metric, (name, color) in zip(
            centers, values[key], POLICIES, strict=True
        ):
            value = float(metric["mean"]) * 100
            low = float(metric["ci_low"]) * 100
            high = float(metric["ci_high"]) * 100
            bar_top = bottom - value / 100 * (bottom - top)
            draw.rectangle((center - 48, bar_top, center + 48, bottom), fill=color)
            ci_top = bottom - high / 100 * (bottom - top)
            ci_bottom = bottom - low / 100 * (bottom - top)
            draw.line((center, ci_top, center, ci_bottom), fill=TEXT, width=3)
            draw.line((center - 10, ci_top, center + 10, ci_top), fill=TEXT, width=3)
            draw.line(
                (center - 10, ci_bottom, center + 10, ci_bottom),
                fill=TEXT,
                width=3,
            )
            draw.text(
                (center, max(ci_top - 9, top + 19)),
                f"{value:.1f}%",
                fill=TEXT,
                font=_font(17, bold=True),
                anchor="mb",
            )
            draw.text(
                (center, bottom + 20),
                name,
                fill=TEXT,
                font=_font(14),
                anchor="ma",
            )
    draw.text(
        (width / 2, 660),
        f"N={sample_count} transitions from episode-isolated test data · MiniMind uses free decoding",
        fill=MUTED,
        font=_font(15),
        anchor="ms",
    )
    image.save(output, format="PNG", optimize=True)


def render_svg(values, sample_count: int, output: Path) -> None:
    width, height = 1500, 720
    elements = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="750" y="58" text-anchor="middle" font-family="Arial,sans-serif" font-size="34" font-weight="700" fill="{TEXT}">MouseMind Offline Policy Baselines</text>',
        f'<text x="750" y="91" text-anchor="middle" font-family="Arial,sans-serif" font-size="17" fill="{MUTED}">Same deterministic held-out subset · 95% bootstrap confidence intervals</text>',
    ]
    for panel_index, (key, title, subtitle) in enumerate(PANELS):
        left = 75 + panel_index * 720
        right = left + 620
        top, bottom = 190, 560
        center_panel = (left + right) / 2
        elements.extend(
            [
                f'<text x="{center_panel}" y="145" text-anchor="middle" font-family="Arial,sans-serif" font-size="23" font-weight="700" fill="{TEXT}">{html.escape(title)}</text>',
                f'<text x="{center_panel}" y="174" text-anchor="middle" font-family="Arial,sans-serif" font-size="15" fill="{MUTED}">{html.escape(subtitle)}</text>',
            ]
        )
        for tick in range(5):
            y = bottom - tick / 4 * (bottom - top)
            elements.extend(
                [
                    f'<line x1="{left + 55}" y1="{y}" x2="{right - 15}" y2="{y}" stroke="{GRID}"/>',
                    f'<text x="{left + 45}" y="{y + 5}" text-anchor="end" font-family="Arial,sans-serif" font-size="13" fill="{MUTED}">{tick * 25}</text>',
                ]
            )
        elements.append(
            f'<path d="M {left + 55} {top} V {bottom} H {right - 15}" fill="none" stroke="{TEXT}" stroke-width="2"/>'
        )
        centers = [left + 155, left + 335, left + 515]
        for center, metric, (name, color) in zip(
            centers, values[key], POLICIES, strict=True
        ):
            value = float(metric["mean"]) * 100
            low = float(metric["ci_low"]) * 100
            high = float(metric["ci_high"]) * 100
            bar_top = bottom - value / 100 * (bottom - top)
            ci_top = bottom - high / 100 * (bottom - top)
            ci_bottom = bottom - low / 100 * (bottom - top)
            elements.extend(
                [
                    f'<rect x="{center - 48}" y="{bar_top}" width="96" height="{bottom - bar_top}" fill="{color}"/>',
                    f'<line x1="{center}" y1="{ci_top}" x2="{center}" y2="{ci_bottom}" stroke="{TEXT}" stroke-width="3"/>',
                    f'<line x1="{center - 10}" y1="{ci_top}" x2="{center + 10}" y2="{ci_top}" stroke="{TEXT}" stroke-width="3"/>',
                    f'<line x1="{center - 10}" y1="{ci_bottom}" x2="{center + 10}" y2="{ci_bottom}" stroke="{TEXT}" stroke-width="3"/>',
                    f'<text x="{center}" y="{max(ci_top - 9, top + 19)}" text-anchor="middle" font-family="Arial,sans-serif" font-size="17" font-weight="700" fill="{TEXT}">{value:.1f}%</text>',
                    f'<text x="{center}" y="{bottom + 38}" text-anchor="middle" font-family="Arial,sans-serif" font-size="14" fill="{TEXT}">{html.escape(name)}</text>',
                ]
            )
    elements.extend(
        [
            f'<text x="750" y="660" text-anchor="middle" font-family="Arial,sans-serif" font-size="15" fill="{MUTED}">N={sample_count} transitions from episode-isolated test data · MiniMind uses free decoding</text>',
            "</svg>",
        ]
    )
    output.write_text("\n".join(elements) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lora-metrics", type=Path, required=True)
    parser.add_argument("--mlp-metrics", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    values, sample_count = load_values(args.lora_metrics, args.mlp_metrics)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    render_png(values, sample_count, args.output_dir / "mousemind_offline_baselines.png")
    render_svg(values, sample_count, args.output_dir / "mousemind_offline_baselines.svg")


if __name__ == "__main__":
    main()
