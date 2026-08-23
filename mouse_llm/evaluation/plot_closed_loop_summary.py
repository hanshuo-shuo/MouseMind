"""Render the three-policy 100-seed closed-loop portfolio summary."""

from __future__ import annotations

import argparse
import html
import json
from pathlib import Path


POLICIES = (
    ("random", "Random", "#6B7280"),
    ("mlp-bc", "Direct MLP", "#2563EB"),
    ("hierarchical-mlp", "Hierarchical", "#059669"),
)
PANELS = (
    ("success_rate", "Success rate", 100.0, 100.0, "Higher is better · %"),
    ("captures", "Captures per episode", 1.0, 100.0, "Lower is better"),
    ("path_efficiency", "Path efficiency", 100.0, 30.0, "Higher is better · %"),
)
TEXT = "#111827"
MUTED = "#4B5563"
GRID = "#D1D5DB"


def load_metrics(path: Path):
    report = json.loads(path.read_text(encoding="utf-8"))
    if report.get("metadata", {}).get("research_evidence") is not True:
        raise ValueError("Closed-loop figure requires research_evidence=true")
    values = {}
    for key, _, _, _, _ in PANELS:
        values[key] = tuple(report["policies"][policy][key] for policy, _, _ in POLICIES)
    return values, int(report["metadata"]["episode_count"])


def _font(size: int, *, bold: bool = False):
    from PIL import ImageFont

    filename = "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"
    try:
        return ImageFont.truetype(filename, size=size)
    except OSError:
        return ImageFont.load_default()


def render_png(values, episode_count: int, output: Path) -> None:
    from PIL import Image, ImageDraw

    width, height = 1800, 700
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    draw.text(
        (width / 2, 28),
        "MouseMind: Offline Winner, Closed-Loop Redesign",
        fill=TEXT,
        font=_font(32, bold=True),
        anchor="ma",
    )
    draw.text(
        (width / 2, 70),
        f"N={episode_count} paired BotEvade seeds · 95% bootstrap confidence intervals",
        fill=MUTED,
        font=_font(16),
        anchor="ma",
    )
    panel_width = 530
    for panel_index, (key, title, scale, y_max, subtitle) in enumerate(PANELS):
        left = 55 + panel_index * 585
        right = left + panel_width
        top, bottom = 180, 545
        draw.text(
            ((left + right) / 2, 112),
            title,
            fill=TEXT,
            font=_font(21, bold=True),
            anchor="ma",
        )
        draw.text(
            ((left + right) / 2, 143),
            subtitle,
            fill=MUTED,
            font=_font(14),
            anchor="ma",
        )
        for tick in range(5):
            y = bottom - tick / 4 * (bottom - top)
            value = tick / 4 * y_max
            draw.line((left + 48, y, right - 10, y), fill=GRID)
            draw.text(
                (left + 40, y),
                f"{value:.0f}",
                fill=MUTED,
                font=_font(12),
                anchor="rm",
            )
        draw.line((left + 48, top, left + 48, bottom), fill=TEXT, width=2)
        draw.line((left + 48, bottom, right - 10, bottom), fill=TEXT, width=2)
        centers = [left + 130, left + 285, left + 440]
        for center, metric, (_, label, color) in zip(
            centers, values[key], POLICIES, strict=True
        ):
            value = float(metric["mean"]) * scale
            low = float(metric["ci_low"]) * scale
            high = float(metric["ci_high"]) * scale
            bar_top = bottom - min(value / y_max, 1.0) * (bottom - top)
            ci_top = bottom - min(high / y_max, 1.0) * (bottom - top)
            ci_bottom = bottom - min(low / y_max, 1.0) * (bottom - top)
            draw.rectangle((center - 42, bar_top, center + 42, bottom), fill=color)
            draw.line((center, ci_top, center, ci_bottom), fill=TEXT, width=3)
            draw.line((center - 9, ci_top, center + 9, ci_top), fill=TEXT, width=3)
            draw.line((center - 9, ci_bottom, center + 9, ci_bottom), fill=TEXT, width=3)
            suffix = "%" if scale == 100 else ""
            shown = f"{value:.1f}{suffix}" if value else f"0{suffix}"
            draw.text(
                (center, max(ci_top - 8, top + 18)),
                shown,
                fill=TEXT,
                font=_font(16, bold=True),
                anchor="mb",
            )
            draw.text(
                (center, bottom + 18),
                label,
                fill=TEXT,
                font=_font(13),
                anchor="ma",
            )
    draw.text(
        (width / 2, 650),
        "Direct MLP wins offline accuracy; hierarchy keeps it as controller and adds instruction-conditioned safety planning",
        fill=MUTED,
        font=_font(15),
        anchor="ms",
    )
    image.save(output, format="PNG", optimize=True)


def render_svg(values, episode_count: int, output: Path) -> None:
    width, height = 1800, 700
    elements = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="900" y="52" text-anchor="middle" font-family="Arial,sans-serif" font-size="32" font-weight="700" fill="{TEXT}">MouseMind: Offline Winner, Closed-Loop Redesign</text>',
        f'<text x="900" y="83" text-anchor="middle" font-family="Arial,sans-serif" font-size="16" fill="{MUTED}">N={episode_count} paired BotEvade seeds · 95% bootstrap confidence intervals</text>',
    ]
    for panel_index, (key, title, scale, y_max, subtitle) in enumerate(PANELS):
        left = 55 + panel_index * 585
        right = left + 530
        top, bottom = 180, 545
        center_panel = (left + right) / 2
        elements.extend(
            [
                f'<text x="{center_panel}" y="127" text-anchor="middle" font-family="Arial,sans-serif" font-size="21" font-weight="700" fill="{TEXT}">{html.escape(title)}</text>',
                f'<text x="{center_panel}" y="154" text-anchor="middle" font-family="Arial,sans-serif" font-size="14" fill="{MUTED}">{html.escape(subtitle)}</text>',
            ]
        )
        for tick in range(5):
            y = bottom - tick / 4 * (bottom - top)
            tick_value = tick / 4 * y_max
            elements.extend(
                [
                    f'<line x1="{left + 48}" y1="{y}" x2="{right - 10}" y2="{y}" stroke="{GRID}"/>',
                    f'<text x="{left + 40}" y="{y + 5}" text-anchor="end" font-family="Arial,sans-serif" font-size="12" fill="{MUTED}">{tick_value:.0f}</text>',
                ]
            )
        elements.append(
            f'<path d="M {left + 48} {top} V {bottom} H {right - 10}" fill="none" stroke="{TEXT}" stroke-width="2"/>'
        )
        centers = [left + 130, left + 285, left + 440]
        for center, metric, (_, label, color) in zip(
            centers, values[key], POLICIES, strict=True
        ):
            value = float(metric["mean"]) * scale
            low = float(metric["ci_low"]) * scale
            high = float(metric["ci_high"]) * scale
            bar_top = bottom - min(value / y_max, 1.0) * (bottom - top)
            ci_top = bottom - min(high / y_max, 1.0) * (bottom - top)
            ci_bottom = bottom - min(low / y_max, 1.0) * (bottom - top)
            suffix = "%" if scale == 100 else ""
            shown = f"{value:.1f}{suffix}" if value else f"0{suffix}"
            elements.extend(
                [
                    f'<rect x="{center - 42}" y="{bar_top}" width="84" height="{bottom - bar_top}" fill="{color}"/>',
                    f'<line x1="{center}" y1="{ci_top}" x2="{center}" y2="{ci_bottom}" stroke="{TEXT}" stroke-width="3"/>',
                    f'<line x1="{center - 9}" y1="{ci_top}" x2="{center + 9}" y2="{ci_top}" stroke="{TEXT}" stroke-width="3"/>',
                    f'<line x1="{center - 9}" y1="{ci_bottom}" x2="{center + 9}" y2="{ci_bottom}" stroke="{TEXT}" stroke-width="3"/>',
                    f'<text x="{center}" y="{max(ci_top - 8, top + 18)}" text-anchor="middle" font-family="Arial,sans-serif" font-size="16" font-weight="700" fill="{TEXT}">{shown}</text>',
                    f'<text x="{center}" y="{bottom + 35}" text-anchor="middle" font-family="Arial,sans-serif" font-size="13" fill="{TEXT}">{html.escape(label)}</text>',
                ]
            )
    elements.extend(
        [
            f'<text x="900" y="650" text-anchor="middle" font-family="Arial,sans-serif" font-size="15" fill="{MUTED}">Direct MLP wins offline accuracy; hierarchy keeps it as controller and adds instruction-conditioned safety planning</text>',
            "</svg>",
        ]
    )
    output.write_text("\n".join(elements) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    values, episode_count = load_metrics(args.metrics)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    render_png(
        values,
        episode_count,
        args.output_dir / "mousemind_closed_loop_summary.png",
    )
    render_svg(
        values,
        episode_count,
        args.output_dir / "mousemind_closed_loop_summary.svg",
    )


if __name__ == "__main__":
    main()
