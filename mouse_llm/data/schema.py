"""Versioned prompt contract for legacy mouse-policy transitions."""

from __future__ import annotations

import json
import math
from collections.abc import Sequence


SCHEMA_NAME = "legacy_mouse_vector_10d_v1"

# V1 was trained with position-indexed JSON keys because the exporter metadata
# had not yet been recovered. Those serialized keys remain immutable for
# checkpoint compatibility. The authoritative semantics were later recovered
# from hanshuo-shuo/Mice commit 67e769f and are verified by the checked-in
# source/data/state-replay audit.
FEATURE_NAMES = tuple(f"obs_{index:02d}" for index in range(10))
SEMANTIC_FEATURE_NAMES = (
    "prey_x",
    "prey_y",
    "prey_direction",
    "predator_x",
    "predator_y",
    "predator_direction",
    "prey_goal_distance",
    "puffed",
    "puff_cooled_down",
    "finished",
)

USER_PROMPT_PREFIX = "Current observation:\n"
USER_PROMPT_SUFFIX = "\nChoose the next action."

SYSTEM_PROMPT = (
    "You are the action policy for a mouse in a partially observable "
    "Cellworld arena. Select one discrete destination action from 0 through "
    "{max_action}. Return exactly one JSON object with an integer action, for "
    'example {{"action":24}}. Do not add prose.'
)


def _rounded(value: float, precision: int) -> int | float:
    if not math.isfinite(value):
        raise ValueError(f"Observation contains a non-finite value: {value!r}")
    rounded = round(float(value), precision)
    return 0 if rounded == 0 else rounded


def state_payload(
    observation: Sequence[float], *, precision: int = 4
) -> dict[str, int | float]:
    if len(observation) != len(FEATURE_NAMES):
        raise ValueError(
            f"{SCHEMA_NAME} expects {len(FEATURE_NAMES)} values, got "
            f"{len(observation)}"
        )
    return {
        name: _rounded(value, precision)
        for name, value in zip(FEATURE_NAMES, observation, strict=True)
    }


def user_prompt(observation: Sequence[float], *, precision: int = 4) -> str:
    payload = json.dumps(
        state_payload(observation, precision=precision),
        ensure_ascii=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    return f"{USER_PROMPT_PREFIX}{payload}{USER_PROMPT_SUFFIX}"


def parse_user_observation(content: str) -> tuple[float, ...]:
    """Recover the numeric policy input from a versioned user prompt.

    Keeping this parser beside the serializer gives non-language baselines the
    exact same input contract as MiniMind. It also fails closed when a future
    prompt schema changes instead of silently training on the wrong fields.
    """
    if not content.startswith(USER_PROMPT_PREFIX) or not content.endswith(
        USER_PROMPT_SUFFIX
    ):
        raise ValueError(f"User message does not match {SCHEMA_NAME}")
    raw_payload = content[
        len(USER_PROMPT_PREFIX) : len(content) - len(USER_PROMPT_SUFFIX)
    ]
    try:
        payload = json.loads(raw_payload)
    except json.JSONDecodeError as exc:
        raise ValueError(f"User message contains invalid {SCHEMA_NAME} JSON") from exc
    if not isinstance(payload, dict) or tuple(payload) != FEATURE_NAMES:
        raise ValueError(
            f"Expected ordered fields {FEATURE_NAMES}, got "
            f"{tuple(payload) if isinstance(payload, dict) else type(payload).__name__}"
        )
    observation: list[float] = []
    for name in FEATURE_NAMES:
        value = payload[name]
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"{name} must be a finite number")
        numeric = float(value)
        if not math.isfinite(numeric):
            raise ValueError(f"{name} must be a finite number")
        observation.append(numeric)
    return tuple(observation)


def conversation_observation(
    conversations: Sequence[dict[str, str]],
) -> tuple[float, ...]:
    """Read an observation from a MiniMind SFT conversation."""
    user_messages = [
        message for message in conversations if message.get("role") == "user"
    ]
    if len(user_messages) != 1 or not isinstance(user_messages[0].get("content"), str):
        raise ValueError("Expected exactly one user observation message")
    return parse_user_observation(user_messages[0]["content"])


def assistant_target(action: int) -> str:
    return json.dumps({"action": int(action)}, separators=(",", ":"))


def make_conversation(
    observation: Sequence[float],
    action: int,
    *,
    action_count: int,
    precision: int = 4,
) -> dict[str, list[dict[str, str]]]:
    if not 0 <= int(action) < action_count:
        raise ValueError(f"Action {action} is outside [0, {action_count - 1}]")
    return {
        "conversations": [
            {
                "role": "system",
                "content": SYSTEM_PROMPT.format(max_action=action_count - 1),
            },
            {
                "role": "user",
                "content": user_prompt(observation, precision=precision),
            },
            {"role": "assistant", "content": assistant_target(action)},
        ]
    }
