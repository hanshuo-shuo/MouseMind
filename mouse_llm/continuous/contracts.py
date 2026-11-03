"""Strict public contracts for continuous XY destinations."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from numbers import Real
from typing import Any, Sequence

ACTION_SCHEMA = "mousemind_xy_destination_v1"
DATA_SCHEMA = "mousemind_continuous_transition_v1"
LABEL_MODES = ("next_position", "commanded_destination")


def validate_xy(value: Sequence[float]) -> tuple[float, float]:
    """Validate an absolute arena destination; never silently clip or quantize."""
    try:
        if len(value) != 2:
            raise ValueError("Continuous destinations require exactly two values: x, y")
        result = []
        for coordinate in value:
            if isinstance(coordinate, bool) or not isinstance(coordinate, Real):
                raise ValueError("Destination coordinates must be real numbers")
            number = float(coordinate)
            if not math.isfinite(number) or not 0.0 <= number <= 1.0:
                raise ValueError("Destination coordinates must be finite and in [0, 1]")
            result.append(number)
        return result[0], result[1]
    except TypeError as exc:
        raise ValueError("Continuous destinations require a two-element sequence") from exc


@dataclass(frozen=True)
class ContinuousDecision:
    action: tuple[float, float]
    valid: bool = True
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "action", validate_xy(self.action))
