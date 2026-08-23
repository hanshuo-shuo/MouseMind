from __future__ import annotations

from mouse_llm.evaluation.plot_comparison import render_png, render_svg


def _metrics():
    def interval(mean, width=0.01):
        return {"mean": mean, "ci_low": max(mean - width, 0), "ci_high": mean + width}

    return {
        "evaluation": {"sample_count": 32},
        "base": {
            "valid_output_rate": interval(0.1),
            "exact_action_accuracy": interval(0.02),
            "task_nll": interval(4.0, 0.1),
            "normalized_destination_error": interval(0.9),
        },
        "lora": {
            "valid_output_rate": interval(0.9),
            "exact_action_accuracy": interval(0.4),
            "task_nll": interval(1.0, 0.1),
            "normalized_destination_error": interval(0.2),
        },
    }


def test_plot_outputs_valid_png_and_svg(tmp_path):
    png = tmp_path / "comparison.png"
    svg = tmp_path / "comparison.svg"
    render_png(_metrics(), png)
    render_svg(_metrics(), svg)
    assert png.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
    assert svg.read_text(encoding="utf-8").startswith("<svg")
    assert "Held-out Episode Evaluation" in svg.read_text(encoding="utf-8")
