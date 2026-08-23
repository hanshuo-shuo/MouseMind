"""Render closed-loop episode outcomes and deterministic failure taxonomy."""

from __future__ import annotations

import argparse
import html
import json
from pathlib import Path
from typing import Any


COLORS = {
    "capture_free_success": "#059669",
    "late_predator_response": "#DC2626",
    "capture_near_occlusion": "#EA580C",
    "open_space_capture": "#F59E0B",
    "capture_other": "#F97316",
    "navigation_oscillation": "#7C3AED",
    "stuck_timeout": "#4F46E5",
    "goal_overshoot": "#0EA5E9",
    "wrong_way_navigation": "#0891B2",
    "timeout_other": "#6B7280",
}
TEXT = "#111827"
MUTED = "#4B5563"
GRID = "#D1D5DB"


def load_outcomes(path: Path) -> tuple[dict[str, dict[str, int]], list[str], int]:
    report = json.loads(path.read_text(encoding="utf-8"))
    if report.get("metadata", {}).get("research_evidence") is not True:
        raise ValueError("Failure figure requires research_evidence=true")
    outcomes: dict[str, dict[str, int]] = {}
    totals: dict[str, int] = {}
    episode_count: int | None = None
    for policy, summary in report["policies"].items():
        count = int(summary["episode_count"])
        if episode_count is None:
            episode_count = count
        elif count != episode_count:
            raise ValueError("Policies must have the same episode count")
        taxonomy = summary["failure_taxonomy"]
        policy_counts = {
            name: int(value) for name, value in taxonomy["counts"].items()
        }
        policy_counts["capture_free_success"] = count - int(
            taxonomy["failed_episode_count"]
        )
        outcomes[policy] = policy_counts
        for name, value in policy_counts.items():
            totals[name] = totals.get(name, 0) + value
    categories = ["capture_free_success", *sorted(
        (name for name in totals if name != "capture_free_success"),
        key=lambda name: (-totals[name], name),
    )]
    return outcomes, categories, int(episode_count or 0)


def _label(name: str) -> str:
    return name.replace("_", " ").title()


def _font(size: int, *, bold: bool = False):
    from PIL import ImageFont

    filename = "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"
    try:
        return ImageFont.truetype(filename, size=size)
    except OSError:
        return ImageFont.load_default()


def render_png(
    outcomes: dict[str, dict[str, int]],
    categories: list[str],
    episode_count: int,
    output: Path,
) -> None:
    from PIL import Image, ImageDraw

    width = 1500
    row_height = 100
    legend_rows = (len(categories) + 2) // 3
    height = 260 + len(outcomes) * row_height + legend_rows * 42
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    draw.text(
        (width / 2, 34),
        "MouseMind Closed-Loop Outcome Taxonomy",
        fill=TEXT,
        font=_font(32, bold=True),
        anchor="ma",
    )
    draw.text(
        (width / 2, 78),
        f"N={episode_count} paired seeds per policy · success means goal + zero captures",
        fill=MUTED,
        font=_font(17),
        anchor="ma",
    )
    plot_left, plot_right = 245, 1425
    for tick in range(5):
        x = plot_left + tick / 4 * (plot_right - plot_left)
        draw.line((x, 130, x, 150 + len(outcomes) * row_height), fill=GRID)
        draw.text(
            (x, 112),
            f"{tick * 25}%",
            fill=MUTED,
            font=_font(13),
            anchor="ma",
        )
    for row_index, (policy, counts) in enumerate(outcomes.items()):
        top = 145 + row_index * row_height
        bottom = top + 56
        draw.text(
            (plot_left - 20, (top + bottom) / 2),
            policy,
            fill=TEXT,
            font=_font(17, bold=True),
            anchor="rm",
        )
        cursor = plot_left
        for category in categories:
            count = counts.get(category, 0)
            if count == 0:
                continue
            segment = count / episode_count * (plot_right - plot_left)
            color = COLORS.get(category, "#9CA3AF")
            draw.rectangle((cursor, top, cursor + segment, bottom), fill=color)
            if count / episode_count >= 0.06:
                draw.text(
                    (cursor + segment / 2, (top + bottom) / 2),
                    f"{count / episode_count:.0%}",
                    fill="white",
                    font=_font(15, bold=True),
                    anchor="mm",
                )
            cursor += segment
    legend_top = 175 + len(outcomes) * row_height
    column_width = 440
    for index, category in enumerate(categories):
        column = index % 3
        row = index // 3
        x = 120 + column * column_width
        y = legend_top + row * 42
        draw.rectangle((x, y, x + 24, y + 20), fill=COLORS.get(category, "#9CA3AF"))
        draw.text(
            (x + 34, y + 10),
            _label(category),
            fill=TEXT,
            font=_font(14),
            anchor="lm",
        )
    image.save(output, format="PNG", optimize=True)


def render_svg(
    outcomes: dict[str, dict[str, int]],
    categories: list[str],
    episode_count: int,
    output: Path,
) -> None:
    width = 1500
    row_height = 100
    legend_rows = (len(categories) + 2) // 3
    height = 260 + len(outcomes) * row_height + legend_rows * 42
    plot_left, plot_right = 245, 1425
    elements = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="750" y="58" text-anchor="middle" font-family="Arial,sans-serif" font-size="32" font-weight="700" fill="{TEXT}">MouseMind Closed-Loop Outcome Taxonomy</text>',
        f'<text x="750" y="91" text-anchor="middle" font-family="Arial,sans-serif" font-size="17" fill="{MUTED}">N={episode_count} paired seeds per policy · success means goal + zero captures</text>',
    ]
    for tick in range(5):
        x = plot_left + tick / 4 * (plot_right - plot_left)
        elements.extend(
            [
                f'<line x1="{x}" y1="130" x2="{x}" y2="{150 + len(outcomes) * row_height}" stroke="{GRID}"/>',
                f'<text x="{x}" y="120" text-anchor="middle" font-family="Arial,sans-serif" font-size="13" fill="{MUTED}">{tick * 25}%</text>',
            ]
        )
    for row_index, (policy, counts) in enumerate(outcomes.items()):
        top = 145 + row_index * row_height
        bottom = top + 56
        elements.append(
            f'<text x="{plot_left - 20}" y="{(top + bottom) / 2 + 6}" text-anchor="end" font-family="Arial,sans-serif" font-size="17" font-weight="700" fill="{TEXT}">{html.escape(policy)}</text>'
        )
        cursor = plot_left
        for category in categories:
            count = counts.get(category, 0)
            if count == 0:
                continue
            segment = count / episode_count * (plot_right - plot_left)
            color = COLORS.get(category, "#9CA3AF")
            elements.append(
                f'<rect x="{cursor}" y="{top}" width="{segment}" height="56" fill="{color}"/>'
            )
            if count / episode_count >= 0.06:
                elements.append(
                    f'<text x="{cursor + segment / 2}" y="{(top + bottom) / 2 + 6}" text-anchor="middle" font-family="Arial,sans-serif" font-size="15" font-weight="700" fill="white">{count / episode_count:.0%}</text>'
                )
            cursor += segment
    legend_top = 175 + len(outcomes) * row_height
    for index, category in enumerate(categories):
        column = index % 3
        row = index // 3
        x = 120 + column * 440
        y = legend_top + row * 42
        elements.extend(
            [
                f'<rect x="{x}" y="{y}" width="24" height="20" fill="{COLORS.get(category, "#9CA3AF")}"/>',
                f'<text x="{x + 34}" y="{y + 15}" font-family="Arial,sans-serif" font-size="14" fill="{TEXT}">{html.escape(_label(category))}</text>',
            ]
        )
    elements.append("</svg>")
    output.write_text("\n".join(elements) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    outcomes, categories, episode_count = load_outcomes(args.metrics)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    render_png(
        outcomes,
        categories,
        episode_count,
        args.output_dir / "mousemind_failure_taxonomy.png",
    )
    render_svg(
        outcomes,
        categories,
        episode_count,
        args.output_dir / "mousemind_failure_taxonomy.svg",
    )


if __name__ == "__main__":
    main()
