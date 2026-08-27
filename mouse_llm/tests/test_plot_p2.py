import pytest

from mouse_llm.evaluation.plot_p2 import (
    latency_tradeoff,
    ood_generalization,
    planner_ablation,
    safety_frontier,
)


def _report():
    policies = {}
    for index, name in enumerate(("direct-mlp", "p1-rule", "numeric-learned", "numeric-verified")):
        task = 0.2 + index * 0.1
        clean = 0.1 + index * 0.1
        capture = 0.9 - index * 0.1
        policies[name] = {
            "success_rate": {"mean": task, "ci_low": task - 0.05, "ci_high": task + 0.05},
            "clean_success_rate": {"mean": clean, "ci_low": max(clean - 0.05, 0), "ci_high": clean + 0.05},
            "capture_rate": {"mean": capture, "ci_low": capture - 0.05, "ci_high": min(capture + 0.05, 1)},
            "latency_seconds": {"p95": 0.001 * (index + 1)},
        }
    return {"experiment": "mousemind_seeded_closed_loop", "policies": policies}


def test_p2_figures_render_from_aggregate_json_only(tmp_path):
    pytest.importorskip("matplotlib")
    report = _report()
    outputs = (
        tmp_path / "frontier.png",
        tmp_path / "ood.png",
        tmp_path / "latency.png",
        tmp_path / "ablation.png",
    )
    safety_frontier(report, [(0.5, report)], outputs[0])
    ood_generalization([("ID", report), ("shift", report)], outputs[1])
    latency_tradeoff(report, outputs[2])
    planner_ablation(report, outputs[3])
    for output in outputs:
        assert output.read_bytes().startswith(b"\x89PNG")
        assert output.with_suffix(".svg").read_text(encoding="utf-8").lstrip().startswith("<?xml")
