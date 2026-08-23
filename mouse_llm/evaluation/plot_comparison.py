"""Render Git-safe aggregate PNG/SVG reports without Matplotlib."""

from __future__ import annotations

import argparse
import html
import json
from pathlib import Path
from typing import Any


PANELS = (
    ("valid_output_rate", "Valid JSON output", 100.0, "%", True, 100.0),
    ("exact_action_accuracy", "Exact action accuracy", 100.0, "%", True, 100.0),
    ("task_nll", "Action-response NLL", 1.0, "nats/token", False, None),
    (
        "normalized_destination_error",
        "Destination error",
        100.0,
        "% of arena diagonal",
        False,
        100.0,
    ),
)

BASE_COLOR = "#6B7280"
LORA_COLOR = "#2563EB"
TEXT_COLOR = "#111827"
MUTED_COLOR = "#4B5563"
GRID_COLOR = "#D1D5DB"
BACKGROUND = "#FFFFFF"


def _panel_values(
    metrics: dict[str, Any], key: str, scale: float
) -> tuple[list[float], list[float], list[float]]:
    values = [metrics[model][key]["mean"] * scale for model in ("base", "lora")]
    lows = [metrics[model][key]["ci_low"] * scale for model in ("base", "lora")]
    highs = [metrics[model][key]["ci_high"] * scale for model in ("base", "lora")]
    return values, lows, highs


def _font(size: int, *, bold: bool = False):
    from PIL import ImageFont

    filename = "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"
    try:
        return ImageFont.truetype(filename, size=size)
    except OSError:
        return ImageFont.load_default()


def render_png(metrics: dict[str, Any], output: Path) -> None:
    from PIL import Image, ImageDraw

    width, height = 1800, 720
    image = Image.new("RGB", (width, height), BACKGROUND)
    draw = ImageDraw.Draw(image)
    title_font = _font(32, bold=True)
    panel_font = _font(19, bold=True)
    label_font = _font(16)
    value_font = _font(18, bold=True)
    small_font = _font(14)
    draw.text(
        (width / 2, 28),
        "MiniMind Mouse Policy - Held-out Episode Evaluation",
        fill=TEXT_COLOR,
        font=title_font,
        anchor="ma",
    )

    outer_left, outer_right = 68, 40
    panel_gap = 28
    panel_width = (width - outer_left - outer_right - 3 * panel_gap) / 4
    plot_top, plot_bottom = 150, 555
    for panel_index, (key, title, scale, unit, higher, fixed_max) in enumerate(PANELS):
        left = outer_left + panel_index * (panel_width + panel_gap)
        right = left + panel_width
        plot_left, plot_right = left + 55, right - 20
        values, lows, highs = _panel_values(metrics, key, scale)
        y_max = fixed_max or max(max(highs) * 1.25, 1e-6)
        draw.text(
            ((left + right) / 2, 92),
            title,
            fill=TEXT_COLOR,
            font=panel_font,
            anchor="ma",
        )
        draw.text(
            ((left + right) / 2, 119),
            f"{'Higher' if higher else 'Lower'} is better | {unit}",
            fill=MUTED_COLOR,
            font=small_font,
            anchor="ma",
        )
        for tick in range(5):
            fraction = tick / 4
            y = plot_bottom - fraction * (plot_bottom - plot_top)
            draw.line((plot_left, y, plot_right, y), fill=GRID_COLOR, width=1)
            tick_value = y_max * fraction
            tick_text = f"{tick_value:.1f}" if y_max < 20 else f"{tick_value:.0f}"
            draw.text(
                (plot_left - 8, y),
                tick_text,
                fill=MUTED_COLOR,
                font=small_font,
                anchor="rm",
            )
        draw.line((plot_left, plot_top, plot_left, plot_bottom), fill=TEXT_COLOR, width=2)
        draw.line((plot_left, plot_bottom, plot_right, plot_bottom), fill=TEXT_COLOR, width=2)
        centers = (
            plot_left + (plot_right - plot_left) * 0.32,
            plot_left + (plot_right - plot_left) * 0.72,
        )
        bar_width = min(92, (plot_right - plot_left) * 0.24)
        for center, value, low, high, color, name in zip(
            centers,
            values,
            lows,
            highs,
            (BASE_COLOR, LORA_COLOR),
            ("Base", "LoRA"),
            strict=True,
        ):
            top = plot_bottom - min(value / y_max, 1) * (plot_bottom - plot_top)
            draw.rectangle(
                (center - bar_width / 2, top, center + bar_width / 2, plot_bottom),
                fill=color,
            )
            ci_top = plot_bottom - min(high / y_max, 1) * (plot_bottom - plot_top)
            ci_bottom = plot_bottom - min(low / y_max, 1) * (plot_bottom - plot_top)
            draw.line((center, ci_top, center, ci_bottom), fill=TEXT_COLOR, width=3)
            draw.line((center - 10, ci_top, center + 10, ci_top), fill=TEXT_COLOR, width=3)
            draw.line((center - 10, ci_bottom, center + 10, ci_bottom), fill=TEXT_COLOR, width=3)
            shown = f"{value:.1f}" if scale == 100 else f"{value:.3f}"
            draw.text(
                (center, max(ci_top - 10, plot_top + 4)),
                shown,
                fill=TEXT_COLOR,
                font=value_font,
                anchor="mb",
            )
            draw.text(
                (center, plot_bottom + 13),
                name,
                fill=TEXT_COLOR,
                font=label_font,
                anchor="ma",
            )
        delta = values[1] - values[0]
        delta_text = f"LoRA change: {delta:+.1f}" if scale == 100 else f"LoRA change: {delta:+.3f}"
        draw.text(
            ((left + right) / 2, 601),
            delta_text,
            fill=TEXT_COLOR,
            font=label_font,
            anchor="ma",
        )

    legend_y = 645
    draw.rectangle((620, legend_y, 644, legend_y + 18), fill=BASE_COLOR)
    draw.text((654, legend_y + 9), "Before: base MiniMind", fill=TEXT_COLOR, font=label_font, anchor="lm")
    draw.rectangle((925, legend_y, 949, legend_y + 18), fill=LORA_COLOR)
    draw.text((959, legend_y + 9), "After: mouse-policy LoRA", fill=TEXT_COLOR, font=label_font, anchor="lm")
    sample_count = metrics["evaluation"]["sample_count"]
    footer = (
        f"N={sample_count} transitions from held-out episodes | 95% bootstrap CIs | "
        "invalid generations receive maximum spatial error"
    )
    draw.text((width / 2, 693), footer, fill=MUTED_COLOR, font=small_font, anchor="ms")
    image.save(output, format="PNG", optimize=True)


def render_svg(metrics: dict[str, Any], output: Path) -> None:
    width, height = 1800, 720
    elements = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        f'<rect width="100%" height="100%" fill="{BACKGROUND}"/>',
        f'<text x="900" y="52" text-anchor="middle" font-family="DejaVu Sans,Arial,sans-serif" font-size="32" font-weight="700" fill="{TEXT_COLOR}">MiniMind Mouse Policy - Held-out Episode Evaluation</text>',
    ]
    outer_left, outer_right, panel_gap = 68, 40, 28
    panel_width = (width - outer_left - outer_right - 3 * panel_gap) / 4
    plot_top, plot_bottom = 150, 555
    for panel_index, (key, title, scale, unit, higher, fixed_max) in enumerate(PANELS):
        left = outer_left + panel_index * (panel_width + panel_gap)
        right = left + panel_width
        plot_left, plot_right = left + 55, right - 20
        values, lows, highs = _panel_values(metrics, key, scale)
        y_max = fixed_max or max(max(highs) * 1.25, 1e-6)
        center_x = (left + right) / 2
        elements.append(
            f'<text x="{center_x:.1f}" y="106" text-anchor="middle" font-family="DejaVu Sans,Arial,sans-serif" font-size="19" font-weight="700" fill="{TEXT_COLOR}">{html.escape(title)}</text>'
        )
        direction = "Higher" if higher else "Lower"
        elements.append(
            f'<text x="{center_x:.1f}" y="133" text-anchor="middle" font-family="DejaVu Sans,Arial,sans-serif" font-size="14" fill="{MUTED_COLOR}">{direction} is better | {html.escape(unit)}</text>'
        )
        for tick in range(5):
            fraction = tick / 4
            y = plot_bottom - fraction * (plot_bottom - plot_top)
            tick_value = y_max * fraction
            tick_text = f"{tick_value:.1f}" if y_max < 20 else f"{tick_value:.0f}"
            elements.append(
                f'<line x1="{plot_left:.1f}" y1="{y:.1f}" x2="{plot_right:.1f}" y2="{y:.1f}" stroke="{GRID_COLOR}" stroke-width="1"/>'
            )
            elements.append(
                f'<text x="{plot_left - 8:.1f}" y="{y + 5:.1f}" text-anchor="end" font-family="DejaVu Sans,Arial,sans-serif" font-size="14" fill="{MUTED_COLOR}">{tick_text}</text>'
            )
        elements.append(
            f'<path d="M {plot_left:.1f} {plot_top} V {plot_bottom} H {plot_right:.1f}" fill="none" stroke="{TEXT_COLOR}" stroke-width="2"/>'
        )
        centers = (
            plot_left + (plot_right - plot_left) * 0.32,
            plot_left + (plot_right - plot_left) * 0.72,
        )
        bar_width = min(92, (plot_right - plot_left) * 0.24)
        for center, value, low, high, color, name in zip(
            centers,
            values,
            lows,
            highs,
            (BASE_COLOR, LORA_COLOR),
            ("Base", "LoRA"),
            strict=True,
        ):
            top = plot_bottom - min(value / y_max, 1) * (plot_bottom - plot_top)
            bar_height = plot_bottom - top
            ci_top = plot_bottom - min(high / y_max, 1) * (plot_bottom - plot_top)
            ci_bottom = plot_bottom - min(low / y_max, 1) * (plot_bottom - plot_top)
            shown = f"{value:.1f}" if scale == 100 else f"{value:.3f}"
            elements.extend(
                [
                    f'<rect x="{center - bar_width / 2:.1f}" y="{top:.1f}" width="{bar_width:.1f}" height="{bar_height:.1f}" fill="{color}"/>',
                    f'<line x1="{center:.1f}" y1="{ci_top:.1f}" x2="{center:.1f}" y2="{ci_bottom:.1f}" stroke="{TEXT_COLOR}" stroke-width="3"/>',
                    f'<line x1="{center - 10:.1f}" y1="{ci_top:.1f}" x2="{center + 10:.1f}" y2="{ci_top:.1f}" stroke="{TEXT_COLOR}" stroke-width="3"/>',
                    f'<line x1="{center - 10:.1f}" y1="{ci_bottom:.1f}" x2="{center + 10:.1f}" y2="{ci_bottom:.1f}" stroke="{TEXT_COLOR}" stroke-width="3"/>',
                    f'<text x="{center:.1f}" y="{max(ci_top - 10, plot_top + 18):.1f}" text-anchor="middle" font-family="DejaVu Sans,Arial,sans-serif" font-size="18" font-weight="700" fill="{TEXT_COLOR}">{shown}</text>',
                    f'<text x="{center:.1f}" y="{plot_bottom + 31}" text-anchor="middle" font-family="DejaVu Sans,Arial,sans-serif" font-size="16" fill="{TEXT_COLOR}">{name}</text>',
                ]
            )
        delta = values[1] - values[0]
        delta_text = f"LoRA change: {delta:+.1f}" if scale == 100 else f"LoRA change: {delta:+.3f}"
        elements.append(
            f'<text x="{center_x:.1f}" y="606" text-anchor="middle" font-family="DejaVu Sans,Arial,sans-serif" font-size="16" fill="{TEXT_COLOR}">{html.escape(delta_text)}</text>'
        )
    elements.extend(
        [
            f'<rect x="620" y="645" width="24" height="18" fill="{BASE_COLOR}"/>',
            f'<text x="654" y="660" font-family="DejaVu Sans,Arial,sans-serif" font-size="16" fill="{TEXT_COLOR}">Before: base MiniMind</text>',
            f'<rect x="925" y="645" width="24" height="18" fill="{LORA_COLOR}"/>',
            f'<text x="959" y="660" font-family="DejaVu Sans,Arial,sans-serif" font-size="16" fill="{TEXT_COLOR}">After: mouse-policy LoRA</text>',
        ]
    )
    sample_count = metrics["evaluation"]["sample_count"]
    footer = (
        f"N={sample_count} transitions from held-out episodes | 95% bootstrap CIs | "
        "invalid generations receive maximum spatial error"
    )
    elements.append(
        f'<text x="900" y="698" text-anchor="middle" font-family="DejaVu Sans,Arial,sans-serif" font-size="14" fill="{MUTED_COLOR}">{html.escape(footer)}</text>'
    )
    elements.append("</svg>")
    output.write_text("\n".join(elements) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--stem", default="mouse_policy_before_after")
    args = parser.parse_args()

    metrics = json.loads(args.metrics.read_text(encoding="utf-8"))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    png = args.output_dir / f"{args.stem}.png"
    svg = args.output_dir / f"{args.stem}.svg"
    render_png(metrics, png)
    render_svg(metrics, svg)
    print(png)
    print(svg)


if __name__ == "__main__":
    main()
