"""Render a privacy-safe side-by-side frozen transfer rollout GIF."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from mouse_llm.baselines.planner_mlp import NumericSkillPlanner
from mouse_llm.data.planner_schema import Preference
from mouse_llm.evaluation.closed_loop import (
    LEGACY_GYM_SOURCE_INDICES,
    MLPCheckpointPolicy,
    Policy,
    _policy_input,
)
from mouse_llm.evaluation.contracts import (
    DEFAULT_TRANSFER_CONTRACT,
    load_transfer_contract,
)
from mouse_llm.evaluation.evaluate_policy import load_action_catalog
from mouse_llm.evaluation.transfer_benchmark import (
    GoalCoordinatePolicy,
    _PlannerIsolationSpecialist,
)
from mouse_llm.hierarchical.verified_policy import ProposeVerifyPolicy


@dataclass(frozen=True)
class FrameState:
    step: int
    prey: tuple[float, float]
    predator: tuple[float, float]
    predator_visible: bool
    goal: tuple[float, float] | None
    captures: int
    goals_completed: int
    return_completed: int
    objectives_completed: int
    skill: str
    terminated: bool
    timed_out: bool


def _record(env_factory: Any, policy: Policy, *, seed: int) -> list[FrameState]:
    env = env_factory()
    states: list[FrameState] = []
    try:
        observation, _ = env.reset(seed=seed)
        policy.reset(seed)
        info: dict[str, Any] = {
            "captures": 0,
            "goals_completed": 0,
            "objectives_completed": 0,
            "predator_visible": 0,
        }
        skill = "initializing"
        while True:
            goal = (
                tuple(float(value) for value in env.model.goal_location)
                if env.model.goal_location is not None
                else None
            )
            states.append(
                FrameState(
                    step=int(env.step_count),
                    prey=tuple(float(value) for value in env.model.prey.state.location),
                    predator=tuple(
                        float(value) for value in env.model.predator.state.location
                    ),
                    predator_visible=bool(info.get("predator_visible", 0)),
                    goal=goal,
                    captures=int(info.get("captures", 0)),
                    goals_completed=int(info.get("goals_completed", 0)),
                    return_completed=int(info.get("return_completed", 0)),
                    objectives_completed=int(info.get("objectives_completed", 0)),
                    skill=skill,
                    terminated=not env.model.running,
                    timed_out=bool(
                        env.step_count >= env.max_step and env.model.running
                    ),
                )
            )
            if not env.model.running or env.step_count >= env.max_step:
                break
            decision = policy.act(_policy_input(env, policy, observation))
            if decision.metadata and isinstance(decision.metadata.get("skill"), str):
                skill = decision.metadata["skill"]
            observation, _, terminated, truncated, info = env.step(decision.action)
            if terminated or truncated:
                continue
        return states
    finally:
        env.close()


def _font(size: int, *, bold: bool = False):
    from PIL import ImageFont

    filename = "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"
    try:
        return ImageFont.truetype(filename, size=size)
    except OSError:
        return ImageFont.load_default()


def _xy(point: tuple[float, float], box: tuple[int, int, int, int]) -> tuple[int, int]:
    left, top, right, bottom = box
    return (
        int(left + point[0] * (right - left)),
        int(bottom - point[1] * (bottom - top)),
    )


def _render(
    trajectories: list[tuple[str, list[FrameState]]],
    *,
    destinations: np.ndarray,
    goals: list[tuple[float, float]],
    seed: int,
    output: Path,
) -> None:
    from PIL import Image, ImageDraw

    width, height = 1500, 540
    panel_width = width // len(trajectories)
    sample_stride = 3
    frame_count = max((len(states) + sample_stride - 1) // sample_stride for _, states in trajectories)
    frames = []
    path_history: list[list[tuple[float, float]]] = [[] for _ in trajectories]
    for frame_index in range(frame_count):
        image = Image.new("RGB", (width, height), "white")
        draw = ImageDraw.Draw(image)
        draw.text(
            (width // 2, 18),
            f"Frozen BotEvade → Oasis transfer · evaluation seed {seed}",
            fill="#111827",
            font=_font(24, bold=True),
            anchor="ma",
        )
        legend = (
            (width // 2 - 170, "#2563EB", "prey"),
            (width // 2, "#DC2626", "predator"),
            (width // 2 + 190, "#10B981", "active goal"),
        )
        for x, color, label in legend:
            draw.ellipse((x - 7, 44, x + 7, 58), fill=color)
            draw.text((x + 14, 51), label, fill="#475569", font=_font(12), anchor="lm")
        for panel_index, (label, states) in enumerate(trajectories):
            state_index = min(frame_index * sample_stride, len(states) - 1)
            state = states[state_index]
            path_history[panel_index] = [
                item.prey for item in states[: state_index + 1]
            ]
            panel_left = panel_index * panel_width
            box = (panel_left + 35, 104, panel_left + panel_width - 35, 470)
            draw.rounded_rectangle(box, radius=12, fill="#F8FAFC", outline="#CBD5E1", width=2)
            for destination in destinations:
                x, y = _xy((float(destination[0]), float(destination[1])), box)
                draw.ellipse((x - 1, y - 1, x + 1, y + 1), fill="#CBD5E1")
            for goal in goals:
                x, y = _xy(goal, box)
                draw.ellipse((x - 5, y - 5, x + 5, y + 5), outline="#94A3B8", width=2)
            path = [_xy(point, box) for point in path_history[panel_index]]
            if len(path) >= 2:
                draw.line(path, fill="#BFDBFE", width=2)
                draw.line(path[-80:], fill="#2563EB", width=3)
            if state.goal is not None:
                gx, gy = _xy(state.goal, box)
                draw.ellipse((gx - 9, gy - 9, gx + 9, gy + 9), outline="#10B981", width=4)
            px, py = _xy(state.prey, box)
            dx, dy = _xy(state.predator, box)
            draw.ellipse((px - 8, py - 8, px + 8, py + 8), fill="#2563EB", outline="white", width=2)
            draw.ellipse(
                (dx - 9, dy - 9, dx + 9, dy + 9),
                fill="#DC2626" if state.predator_visible else "#FCA5A5",
                outline="white",
                width=2,
            )
            draw.text(
                (panel_left + panel_width // 2, 78),
                label,
                fill="#111827",
                font=_font(18, bold=True),
                anchor="ma",
            )
            status = (
                "complete"
                if state.terminated
                else "timeout"
                if state.timed_out
                else f"step {state.step}"
            )
            draw.text(
                (panel_left + panel_width // 2, 492),
                f"{status} · goals {state.goals_completed}/3 · return {state.return_completed}/1 · captures {state.captures}",
                fill="#334155",
                font=_font(14),
                anchor="ma",
            )
            draw.text(
                (panel_left + panel_width // 2, 516),
                f"skill: {state.skill.replace('_', ' ')}",
                fill="#64748B",
                font=_font(13),
                anchor="ma",
            )
        frames.append(image)
    output.parent.mkdir(parents=True, exist_ok=True)
    frames[0].save(
        output,
        save_all=True,
        append_images=frames[1:],
        duration=100,
        loop=0,
        optimize=False,
        disposal=2,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, default=DEFAULT_TRANSFER_CONTRACT)
    parser.add_argument("--action-catalog", type=Path, default=Path("mouse_llm/envs/mice/assets/action_catalog_21_05.json"))
    parser.add_argument("--mlp-checkpoint", type=Path, required=True)
    parser.add_argument("--planner-checkpoint", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42000)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    contract = load_transfer_contract(args.contract)
    destinations = load_action_catalog(args.action_catalog)
    goals = [tuple(values) for values in contract["target"]["goal_locations"]]
    from mouse_llm.envs.mice import OasisEnv, oasis_reward

    def env_factory():
        return OasisEnv(
            world_name=contract["target"]["world"],
            goal_locations=goals,
            goal_threshold=contract["target"]["goal_threshold"],
            use_lppos=False,
            use_predator=True,
            frame_stack_k=1,
            max_step=contract["max_steps"],
            time_step=contract["time_step"],
            reward_function=oasis_reward(),
        )

    def specialist(name: str):
        return MLPCheckpointPolicy(
            args.mlp_checkpoint,
            device=args.device,
            observation_indices=LEGACY_GYM_SOURCE_INDICES,
            name=name,
        )

    literal = ProposeVerifyPolicy(
        specialist=specialist("literal-numeric-specialist"),
        destinations=destinations,
        planner=NumericSkillPlanner(args.planner_checkpoint, device=args.device),
        preference=Preference.SURVIVAL_FIRST,
        planner_horizon=contract["selection"]["planner_horizon"],
        evade_distance=contract["selection"]["evade_distance"],
        name="literal-numeric",
        observation_mode="transfer",
    )
    goal_only = GoalCoordinatePolicy(destinations, name="aligned-goal-only")
    aligned = ProposeVerifyPolicy(
        specialist=_PlannerIsolationSpecialist(),
        destinations=destinations,
        planner=NumericSkillPlanner(args.planner_checkpoint, device=args.device),
        preference=Preference.SURVIVAL_FIRST,
        planner_horizon=contract["selection"]["planner_horizon"],
        evade_distance=contract["selection"]["evade_distance"],
        name="aligned-numeric",
        observation_mode="transfer",
        goal_destination_indices=(15, 16),
    )
    trajectories = [
        ("Literal frozen stack", _record(env_factory, literal, seed=args.seed)),
        ("Aligned goal controller", _record(env_factory, goal_only, seed=args.seed)),
        ("Frozen numeric planner", _record(env_factory, aligned, seed=args.seed)),
    ]
    _render(
        trajectories,
        destinations=destinations,
        goals=goals,
        seed=args.seed,
        output=args.output,
    )


if __name__ == "__main__":
    main()
