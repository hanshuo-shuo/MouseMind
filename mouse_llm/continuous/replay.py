"""Fingerprint physical/controller state for fail-closed prefix replay.

These diagnostics are privileged training/evaluation metadata. They must never
be added to planner prompts. This is not a general simulator snapshot API.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any

import numpy as np


def _plain(value: Any, depth: int = 0) -> Any:
    if depth > 5:
        return None
    if value is None or isinstance(value, (bool, str, int, float)):
        return value
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (list, tuple)):
        return [_plain(item, depth + 1) for item in value]
    if isinstance(value, dict):
        return {str(key): _plain(item, depth + 1) for key, item in value.items()}
    return None


def _scalars(obj: Any) -> dict[str, Any]:
    result = {}
    for key, value in getattr(obj, "__dict__", {}).items():
        if key in {"last_step", "episode_count"}:
            # Wall-clock telemetry and lifetime episode counters do not affect
            # the non-real-time dynamics and differ across fresh instances.
            continue
        if isinstance(value, (bool, str, int, float, np.generic)) or value is None:
            result[key] = _plain(value)
    return result


def _fingerprint_payload(env: Any) -> dict:
    base = getattr(env, "unwrapped", env)
    model = getattr(base, "model", None)
    payload = {"environment": _scalars(base), "model": _scalars(model), "agents": {},
               "navigation_contract": getattr(env, "navigation_contract", None),
               "reset_contract": getattr(env, "reset_contract", None),
               "destination_mode": getattr(env, "destination_mode", None),
               "projection_epsilon": getattr(env, "projection_epsilon", None)}
    for name in ("prey", "predator"):
        agent = getattr(model, name, None)
        if agent is None:
            continue
        payload["agents"][name] = {
            "scalars": _scalars(agent),
            "state": _plain(getattr(getattr(agent, "state", None), "__dict__", {})),
            "dynamics": _plain(getattr(getattr(agent, "dynamics", None), "__dict__", {})),
            "navigation": {key: _plain(getattr(agent, key, None)) for key in (
                "destination", "new_destination", "path", "destination_wait",
                "navigation_plan_update_wait", "active_navigation",
            )},
        }
    payload["prey_data"] = _plain(getattr(getattr(model, "prey_data", None), "__dict__", {}))
    rng = getattr(model, "rng", None)
    payload["model_rng"] = _plain(rng.getstate()) if hasattr(rng, "getstate") else None
    return payload


def replay_fingerprint(env: Any) -> str:
    payload = _fingerprint_payload(env)
    serialized = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(serialized.encode()).hexdigest()
