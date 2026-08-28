"""Versioned, immutable evaluation contracts for MouseMind experiments."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


DEFAULT_P2_CONTRACT = Path(__file__).with_name("contracts") / "p2_eval_v1.json"
DEFAULT_TRANSFER_CONTRACT = (
    Path(__file__).with_name("contracts") / "cross_task_transfer_v1.json"
)


def seed_values(specification: dict[str, Any]) -> tuple[int, ...]:
    start = int(specification["start"])
    stop = int(specification["stop_exclusive"])
    seeds = tuple(range(start, stop))
    if len(seeds) != int(specification["count"]):
        raise ValueError("Seed count does not match the declared range")
    digest = hashlib.sha256(",".join(map(str, seeds)).encode("ascii")).hexdigest()
    if digest != specification["sha256"]:
        raise ValueError("Seed range SHA256 does not match the frozen contract")
    return seeds


def load_p2_contract(path: Path = DEFAULT_P2_CONTRACT) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != 1:
        raise ValueError(f"Unsupported P2 evaluation contract: {path}")
    pools = {
        name: seed_values(specification)
        for name, specification in payload["seed_pools"].items()
    }
    names = tuple(pools)
    for index, left_name in enumerate(names):
        for right_name in names[index + 1 :]:
            overlap = set(pools[left_name]).intersection(pools[right_name])
            if overlap:
                raise ValueError(
                    f"P2 seed pools overlap: {left_name} and {right_name}"
                )
    return payload


def contract_seeds(pool: str, path: Path = DEFAULT_P2_CONTRACT) -> tuple[int, ...]:
    contract = load_p2_contract(path)
    try:
        specification = contract["seed_pools"][pool]
    except KeyError as exc:
        raise ValueError(f"Unknown P2 seed pool {pool!r}") from exc
    return seed_values(specification)


def load_transfer_contract(
    path: Path = DEFAULT_TRANSFER_CONTRACT,
) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if (
        payload.get("schema_version") != 1
        or payload.get("contract_name") != "mousemind_cross_task_transfer_v1"
    ):
        raise ValueError(f"Unsupported transfer evaluation contract: {path}")
    pools = {
        name: seed_values(specification)
        for name, specification in payload["seed_pools"].items()
    }
    names = tuple(pools)
    for index, left_name in enumerate(names):
        for right_name in names[index + 1 :]:
            if set(pools[left_name]).intersection(pools[right_name]):
                raise ValueError(
                    f"Transfer seed pools overlap: {left_name} and {right_name}"
                )
    if payload["observation_contract"]["literal_low_level_transfer"]["compatible"]:
        raise ValueError("The frozen 10D low-level transfer must remain fail-closed")
    return payload


def transfer_contract_seeds(
    pool: str, path: Path = DEFAULT_TRANSFER_CONTRACT
) -> tuple[int, ...]:
    contract = load_transfer_contract(path)
    try:
        specification = contract["seed_pools"][pool]
    except KeyError as exc:
        raise ValueError(f"Unknown transfer seed pool {pool!r}") from exc
    return seed_values(specification)
