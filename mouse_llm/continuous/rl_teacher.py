"""Train a continuous SAC whole-policy teacher with optional BC actor warmstart.

Offline next-position labels are not executed actions and must never enter SAC
replay. Warmstart uses current observations to distill a frozen BC policy's
predictions into the actor. SAC replay then contains online environment data
only. This policy is an independent teacher/baseline, not a GO skill.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
import random
import tempfile
from dataclasses import dataclass
from numbers import Real
from pathlib import Path
from typing import Any, Callable, Sequence

from .contracts import ACTION_SCHEMA, DATA_SCHEMA, LABEL_MODES, ContinuousDecision, validate_xy

TEACHER_SCHEMA = "mousemind_continuous_sac_teacher_v1"
OBSERVATION_SCHEMA = "legacy_mouse_vector_10d_v1"
POLICY_ROLE = "whole_policy_teacher"
OBSERVATION_DIM = 10
SAFE_INFO_FIELDS = {"captures", "is_success", "survived", "termination_reason", "TimeLimit.truncated"}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _observation(value: Sequence[float]) -> tuple[float, ...]:
    if len(value) != OBSERVATION_DIM:
        raise ValueError("SAC teacher requires the ten-dimensional legacy observation")
    result = []
    for entry in value:
        if isinstance(entry, bool) or not isinstance(entry, Real) or not math.isfinite(entry):
            raise ValueError("SAC observation must contain ten finite real numbers")
        result.append(float(entry))
    return tuple(result)


def _seed(value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or not 0 <= value < 2**32:
        raise ValueError("seed must be an integer in [0, 2**32)")
    return value


def _load_sac_class():
    try:
        from stable_baselines3 import SAC
    except ImportError as exc:
        raise RuntimeError("SAC training/loading requires stable-baselines3 and its Gymnasium/PyTorch dependencies") from exc
    return SAC


def _validate_spaces(model_or_env: Any) -> None:
    import numpy as np

    action_space = model_or_env.action_space
    if getattr(action_space, "shape", None) != (2,) or not np.array_equal(action_space.low, np.zeros(2)) or not np.array_equal(action_space.high, np.ones(2)):
        raise ValueError("SAC requires the native XY Box(0, 1, shape=(2,)) action contract")
    if getattr(model_or_env.observation_space, "shape", None) != (OBSERVATION_DIM,):
        raise ValueError("SAC requires the legacy ten-dimensional observation contract")


def wrap_legacy_observation(env):
    """Keep native XY dynamics and expose only the audited public 10D encoder."""
    import gymnasium as gym
    import numpy as np

    if not callable(getattr(env, "legacy_policy_observation", None)):
        raise ValueError("Training environment must expose legacy_policy_observation()")

    class LegacyObservationWrapper(gym.Wrapper):
        def __init__(self, base):
            super().__init__(base)
            self.observation_space = gym.spaces.Box(-np.inf, np.inf, shape=(OBSERVATION_DIM,), dtype=np.float32)
            _validate_spaces(self)

        def _public_observation(self):
            result = np.asarray(_observation(self.env.legacy_policy_observation()), dtype=np.float32)
            if not np.isfinite(result).all():
                raise ValueError("Legacy observation overflows float32")
            return result

        def reset(self, *, seed=None, options=None):
            _, info = self.env.reset(seed=seed, options=options)
            return self._public_observation(), {key: value for key, value in info.items() if key in SAFE_INFO_FIELDS}

        def step(self, action):
            destination = validate_xy(action)
            _, reward, terminated, truncated, info = self.env.step(destination)
            return self._public_observation(), reward, terminated, truncated, {key: value for key, value in info.items() if key in SAFE_INFO_FIELDS}

    return LegacyObservationWrapper(env)


def make_training_env(*, world: str = "21_05", max_steps: int = 300,
                      time_step: float = 0.25, predator_speed_ratio: float = 0.15,
                      destination_mode: str = "project", projection_epsilon: float = 1e-5):
    from .environment import ContinuousBotEvadeEnv

    # ContinuousBotEvadeEnv supplies the original custom_reward and simulator.
    return wrap_legacy_observation(ContinuousBotEvadeEnv(
        world_name=world, max_step=max_steps, time_step=time_step,
        predator_prey_forward_speed_ratio=predator_speed_ratio,
        destination_mode=destination_mode, projection_epsilon=projection_epsilon,
        render=False, real_time=False,
    ))


@dataclass(frozen=True)
class WarmstartInputs:
    observations: tuple[tuple[float, ...], ...]
    provenance: dict[str, Any]


def load_warmstart_inputs(path: Path, *, max_samples: int = 60000, seed: int = 42) -> WarmstartInputs:
    """Read current observations only; never consume action/reward/next-state labels."""
    path = Path(path).resolve()
    if max_samples < 0:
        raise ValueError("max_samples must be nonnegative")
    _seed(seed)
    digest = _sha256(path)
    source_manifest = path.parent / "manifest.json"
    split_verification = "manifest_unavailable"
    if source_manifest.is_file():
        manifest = json.loads(source_manifest.read_text())
        training = manifest.get("artifacts", {}).get("train", {})
        if training.get("sha256") != digest:
            raise ValueError("Warmstart data must match the prepared manifest's train split")
        split_verification = "prepared_train_sha256_matched"
    observations, modes = [], set()
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
                if row.get("schema_version") != DATA_SCHEMA or row.get("action_schema") != ACTION_SCHEMA:
                    raise ValueError("Wrong continuous data/action schema")
                if row.get("label_mode") not in LABEL_MODES:
                    raise ValueError("Missing or unsupported label mode")
                modes.add(row["label_mode"])
                observations.append(_observation(row["observation"]))
            except (KeyError, TypeError, ValueError, AttributeError) as exc:
                raise ValueError(f"Invalid warmstart observation record at line {line_number}") from exc
    if not observations or len(modes) != 1:
        raise ValueError("Warmstart input must be nonempty with exactly one label mode")
    source_rows = len(observations)
    if max_samples and source_rows > max_samples:
        indices = sorted(random.Random(seed).sample(range(source_rows), max_samples))
        observations = [observations[index] for index in indices]
    return WarmstartInputs(tuple(observations), {
        "filename": path.name, "sha256": digest, "source_rows": source_rows,
        "selected_rows": len(observations), "source_label_mode": next(iter(modes)),
        "fields_read_for_supervision": ["observation"],
        "target_source": "frozen_BC_policy_predictions", "split_verification": split_verification,
        "offline_replay_ingestion": False,
    })


def warmstart_actor(model, teacher, observations: Sequence[Sequence[float]], *,
                    steps: int, batch_size: int = 256, learning_rate: float = 3e-4,
                    seed: int = 42) -> dict[str, Any]:
    """Fit SAC's deterministic squashed mean to BC predictions; leave critics/replay alone."""
    import numpy as np
    import torch

    if steps < 1 or batch_size < 1 or not math.isfinite(learning_rate) or learning_rate <= 0:
        raise ValueError("Warmstart steps, batch size, and learning rate must be positive")
    _validate_spaces(model)
    values = np.asarray([_observation(row) for row in observations], dtype=np.float32)
    if not len(values) or not np.isfinite(values).all():
        raise ValueError("Warmstart needs nonempty finite float32 observations")
    rng = np.random.default_rng(_seed(seed))
    teacher.reset(seed)
    actor = model.actor
    device = next(actor.parameters()).device
    optimizer = torch.optim.Adam(actor.parameters(), lr=learning_rate)
    buffer_before = int(model.replay_buffer.size())
    was_training = actor.training
    actor.train()
    losses = []
    try:
        for _ in range(steps):
            batch = values[rng.integers(0, len(values), size=min(batch_size, len(values)))]
            # The next-position labels in the source file are deliberately unused.
            targets = [validate_xy(teacher.act(row.copy()).action) for row in batch]
            obs_tensor = torch.as_tensor(batch, dtype=torch.float32, device=device)
            # SB3 Actor.forward emits tanh-squashed [-1, 1] actions before
            # policy.predict unscales them to the environment's [0, 1] Box.
            target_tensor = 2.0 * torch.as_tensor(targets, dtype=torch.float32, device=device) - 1.0
            prediction = actor(obs_tensor, deterministic=True)
            loss = torch.nn.functional.mse_loss(prediction, target_tensor)
            if not torch.isfinite(loss):
                raise RuntimeError("Non-finite SAC actor warmstart loss")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(actor.parameters(), max_norm=5.0, error_if_nonfinite=True)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
    finally:
        actor.train(was_training)
    if int(model.replay_buffer.size()) != buffer_before:
        raise RuntimeError("Actor warmstart must not modify SAC replay")
    return {
        "optimizer_steps": steps, "rows": len(values), "batch_size": batch_size,
        "learning_rate": learning_rate, "first_batch_scaled_mse": losses[0],
        "last_batch_scaled_mse": losses[-1], "mean_scaled_mse": math.fsum(losses) / len(losses),
        "target_source": "frozen_BC_policy_predictions", "actor_prediction": "deterministic_squashed_mean",
        "critic_warmstart": False, "offline_replay_ingestion": False,
        "replay_size_before": buffer_before, "replay_size_after": int(model.replay_buffer.size()),
        "scope": "offline_initialization_diagnostic_not_environment_performance",
    }


def _private_output(path: Path) -> Path:
    path = Path(path).expanduser().resolve()
    if any((parent / ".git").exists() for parent in (path, *path.parents)):
        raise ValueError("Private SAC outputs must be outside every Git checkout")
    if path.exists() and (not path.is_dir() or any(path.iterdir())):
        raise ValueError("SAC output directory must be new or empty")
    return path


def _save_private_model(model, path: Path) -> None:
    descriptor, temporary = tempfile.mkstemp(prefix=".sac-", suffix=".zip", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            model.save(handle)
        os.replace(temporary, path)
        path.chmod(0o600)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _save_private_json(path: Path, payload: dict) -> None:
    descriptor, temporary = tempfile.mkstemp(prefix=".metadata-", suffix=".json", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
        os.replace(temporary, path)
        path.chmod(0o600)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def train_teacher(args: argparse.Namespace, *, env_factory: Callable | None = None,
                  sac_class=None, teacher_factory: Callable | None = None) -> dict[str, Any]:
    """Run real online SAC when invoked; injection points support offline contract tests."""
    output = _private_output(args.output_dir)
    _seed(args.seed)
    for name in ("total_timesteps", "max_steps", "buffer_size", "batch_size"):
        if getattr(args, name) < 1:
            raise ValueError(f"{name} must be positive")
    if args.learning_starts < 0:
        raise ValueError("learning_starts must be nonnegative")
    for name in ("learning_rate", "time_step", "predator_speed_ratio"):
        if not math.isfinite(getattr(args, name)) or getattr(args, name) <= 0:
            raise ValueError(f"{name} must be finite and positive")
    if bool(args.bc_checkpoint) != bool(args.warmstart_data):
        raise ValueError("BC warmstart requires both --bc-checkpoint and --warmstart-data")
    if args.bc_checkpoint and args.warmstart_steps < 1:
        raise ValueError("BC warmstart requires positive --warmstart-steps")
    inputs = None
    teacher = None
    bc_source = None
    if args.bc_checkpoint:
        checkpoint = Path(args.bc_checkpoint).resolve()
        bc_source = {"filename": checkpoint.name, "sha256": _sha256(checkpoint)}
        inputs = load_warmstart_inputs(args.warmstart_data, max_samples=args.warmstart_max_samples, seed=args.seed)
        if teacher_factory is None:
            from .specialist import ContinuousCheckpointPolicy
            teacher_factory = ContinuousCheckpointPolicy
        teacher = teacher_factory(checkpoint, device=args.device)
        bc_source["label_mode"] = getattr(teacher, "label_mode", None)
    injected = sac_class is not None or env_factory is not None
    SAC = sac_class or _load_sac_class()
    environment_config = {"world": args.world, "max_steps": args.max_steps, "time_step": args.time_step, "predator_speed_ratio": args.predator_speed_ratio,
                          "destination_mode": args.destination_mode, "projection_epsilon": args.projection_epsilon}
    env = (env_factory or make_training_env)(**environment_config)
    try:
        _validate_spaces(env)
        model = SAC(
            "MlpPolicy", env, learning_rate=args.learning_rate,
            buffer_size=args.buffer_size, learning_starts=args.learning_starts,
            batch_size=args.batch_size, train_freq=1, gradient_steps=1,
            policy_kwargs={"net_arch": [256, 256], "share_features_extractor": False},
            seed=args.seed, device=args.device, verbose=0,
        )
        warmstart = {"optimizer_steps": 0, "offline_replay_ingestion": False}
        if teacher is not None and inputs is not None:
            warmstart = warmstart_actor(model, teacher, inputs.observations,
                steps=args.warmstart_steps, batch_size=args.warmstart_batch_size,
                learning_rate=args.warmstart_learning_rate, seed=args.seed)
            warmstart["input_data"] = inputs.provenance
        if int(model.replay_buffer.size()) != 0:
            raise RuntimeError("Online SAC must start with an empty replay buffer")
        model.learn(total_timesteps=args.total_timesteps, progress_bar=False)
        output = _private_output(output)
        output.mkdir(parents=True, exist_ok=True, mode=0o700)
        output.chmod(0o700)
        checkpoint = output / "sac_teacher.zip"
        _save_private_model(model, checkpoint)
        try:
            version = importlib.metadata.version("stable-baselines3")
        except importlib.metadata.PackageNotFoundError:
            version = "unavailable_in_injected_test"
        metadata = {
            "schema_version": TEACHER_SCHEMA, "algorithm": "stable_baselines3.SAC",
            "stable_baselines3_version": version, "action_schema": ACTION_SCHEMA,
            "observation_schema": OBSERVATION_SCHEMA, "observation_dim": OBSERVATION_DIM,
            "action_dim": 2, "action_bounds": [0.0, 1.0], "policy_role": POLICY_ROLE,
            "observation_permissions": "public_legacy_10d_only; hidden_predator_coordinates_remain_masked",
            "reward": "unchanged_environment_custom_reward", "environment": environment_config,
            "navigation_contract": getattr(env, "navigation_contract", "injected_test_environment"),
            "reset_contract": getattr(env, "reset_contract", "injected_test_environment"),
            "seed": args.seed, "requested_timesteps": args.total_timesteps,
            "completed_timesteps": int(model.num_timesteps),
            "training_completed": True, "closed_loop_evaluated": False,
            "research_evidence": False, "injected_test_dependencies": injected,
            "replay_source": "online_executed_environment_transitions_only",
            "offline_replay_ingestion": False, "source_bc_checkpoint": bc_source,
            "warmstart": warmstart, "checkpoint_filename": checkpoint.name,
            "checkpoint_sha256": _sha256(checkpoint),
            "training_config": {key: value if not isinstance(value, Path) else str(value) for key, value in vars(args).items()},
        }
        _save_private_json(checkpoint.with_suffix(".metadata.json"), metadata)
        return metadata
    finally:
        env.close()


class SACCheckpointPolicy:
    """Deterministic full-policy XY baseline with a verified metadata sidecar."""

    observation_mode = "legacy"
    action_schema = ACTION_SCHEMA

    def __init__(self, checkpoint: str | Path, *, device: str = "cpu",
                 name: str = "sac-teacher", metadata_path: Path | None = None, _sac_class=None):
        checkpoint = Path(checkpoint).expanduser().resolve()
        if checkpoint.suffix != ".zip":
            checkpoint = checkpoint.with_suffix(".zip")
        sidecar = Path(metadata_path) if metadata_path else checkpoint.with_suffix(".metadata.json")
        metadata = json.loads(sidecar.read_text(encoding="utf-8"))
        required = {"schema_version": TEACHER_SCHEMA, "action_schema": ACTION_SCHEMA,
                    "observation_schema": OBSERVATION_SCHEMA, "observation_dim": OBSERVATION_DIM,
                    "action_dim": 2, "policy_role": POLICY_ROLE}
        if any(metadata.get(key) != value for key, value in required.items()):
            raise ValueError("SAC checkpoint metadata does not match the continuous teacher contract")
        if metadata.get("checkpoint_sha256") != _sha256(checkpoint):
            raise ValueError("SAC checkpoint hash does not match its metadata")
        self.model = (_sac_class or _load_sac_class()).load(checkpoint, device=device)
        _validate_spaces(self.model)
        self.name, self.metadata = name, metadata

    def reset(self, seed: int) -> None:
        self.model.set_random_seed(_seed(seed))

    def act(self, observation: Sequence[float]) -> ContinuousDecision:
        import numpy as np

        values = np.asarray(_observation(observation), dtype=np.float32)
        if not np.isfinite(values).all():
            raise ValueError("Observation overflows float32")
        action, _ = self.model.predict(values, deterministic=True)
        return ContinuousDecision(validate_xy(action), metadata={
            "action_schema": ACTION_SCHEMA, "policy_role": POLICY_ROLE,
            "prediction_rule": "deterministic_SAC_squashed_mean",
        })


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    train = commands.add_parser("train", help="Train an independent continuous SAC teacher")
    train.add_argument("--output-dir", type=Path, required=True)
    train.add_argument("--total-timesteps", type=int, default=100000)
    train.add_argument("--seed", type=int, default=42)
    train.add_argument("--bc-checkpoint", type=Path)
    train.add_argument("--warmstart-data", type=Path)
    train.add_argument("--warmstart-steps", type=int, default=1000)
    train.add_argument("--warmstart-batch-size", type=int, default=256)
    train.add_argument("--warmstart-learning-rate", type=float, default=3e-4)
    train.add_argument("--warmstart-max-samples", type=int, default=60000)
    train.add_argument("--world", default="21_05")
    train.add_argument("--max-steps", type=int, default=300)
    train.add_argument("--time-step", type=float, default=0.25)
    train.add_argument("--predator-speed-ratio", type=float, default=0.15)
    train.add_argument("--destination-mode", choices=("project", "reject"), default="project")
    train.add_argument("--projection-epsilon", type=float, default=1e-5)
    train.add_argument("--learning-rate", type=float, default=3e-4)
    train.add_argument("--learning-starts", type=int, default=1000)
    train.add_argument("--buffer-size", type=int, default=100000)
    train.add_argument("--batch-size", type=int, default=256)
    train.add_argument("--device", default="cpu")
    return parser


def main() -> None:
    metadata = train_teacher(build_parser().parse_args())
    print(json.dumps(metadata, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
