from __future__ import annotations

import csv
import json
import math
from types import SimpleNamespace

import pytest

from mouse_llm.evaluation.audit_observation_contract import (
    _source_fields,
    audit_dataset_distribution,
    reference_legacy_observation,
)


def test_source_field_parser_recovers_legacy_order():
    source = """
class BotEvadeObservation:
    fields = ["prey_x", "prey_y", "prey_direction"]
"""
    assert _source_fields(source) == (
        "prey_x",
        "prey_y",
        "prey_direction",
    )


def test_distribution_audit_accepts_visible_and_hidden_rows(tmp_path):
    path = tmp_path / "legacy.csv"
    hidden = [0.1, 0.5, -math.pi, 0, 0, 0, 0.9, 0, 0, 0]
    visible = [0.2, 0.6, math.pi, 0.7, 0.4, -1.0, 0.8, 1, 0.5, 1]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=("obs", "action", "reward", "next_obs", "done")
        )
        writer.writeheader()
        for observation in (hidden, visible):
            writer.writerow(
                {
                    "obs": json.dumps(observation),
                    "action": 0,
                    "reward": 0,
                    "next_obs": json.dumps(observation),
                    "done": 0,
                }
            )
    report = audit_dataset_distribution(path)
    assert report["status"] == "PASS"
    assert report["predator_visibility"] == {
        "hidden_rows": 1,
        "visible_rows": 1,
    }


def test_reference_encoder_preserves_positive_pi():
    model = SimpleNamespace(
        use_predator=True,
        running=True,
        puff_cool_down=0.25,
        prey=SimpleNamespace(
            state=SimpleNamespace(location=(0.1, 0.5), direction=180)
        ),
        predator=SimpleNamespace(
            state=SimpleNamespace(location=(0.7, 0.4), direction=-180)
        ),
        prey_data=SimpleNamespace(
            predator_visible=True,
            prey_goal_distance=0.9,
            puffed=False,
        ),
    )
    encoded = reference_legacy_observation(model)
    assert encoded[2] == pytest.approx(math.pi)
    assert encoded[5] == pytest.approx(-math.pi)
