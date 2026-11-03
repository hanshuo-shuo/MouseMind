from __future__ import annotations

import json
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from mouse_llm.continuous.contracts import ACTION_SCHEMA, DATA_SCHEMA, ContinuousDecision
from mouse_llm.continuous.rl_teacher import (
    OBSERVATION_SCHEMA,
    POLICY_ROLE,
    TEACHER_SCHEMA,
    SACCheckpointPolicy,
    _sha256,
    build_parser,
    load_warmstart_inputs,
    train_teacher,
    warmstart_actor,
    wrap_legacy_observation,
)


def spaces():
    gym = pytest.importorskip("gymnasium")
    return gym.spaces.Box(-np.inf, np.inf, shape=(10,), dtype=np.float32), gym.spaces.Box(0.0, 1.0, shape=(2,), dtype=np.float32)


def test_module_and_help_do_not_import_training_dependencies():
    code = "import sys; import mouse_llm.continuous.rl_teacher; assert not any(x in sys.modules for x in ['stable_baselines3', 'torch', 'gymnasium'])"
    result = subprocess.run([sys.executable, "-B", "-c", code], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    result = subprocess.run([sys.executable, "-B", "-m", "mouse_llm.continuous.rl_teacher", "train", "--help"], capture_output=True, text=True)
    assert result.returncode == 0
    assert "--warmstart-data" in result.stdout


def test_wrapper_uses_public_encoder_and_preserves_action_reward_and_termination():
    gym = pytest.importorskip("gymnasium")

    class NativeXYMock(gym.Env):
        def __init__(self):
            self.observation_space = gym.spaces.Box(-np.inf, np.inf, shape=(15,), dtype=np.float32)
            self.action_space = gym.spaces.Box(0.0, 1.0, shape=(2,), dtype=np.float32)
            self.value = 0.2

        def legacy_policy_observation(self):
            return [self.value, 0.4, -2.5, 0, 0, 0, 0.7, 0, 1, 0]

        def reset(self, *, seed=None, options=None):
            self.received_seed = seed
            return np.full(15, 999), {"predator_x": 0.9, "captures": 0}

        def step(self, action):
            self.last_action = action
            self.value = 0.3
            return np.full(15, 999), -2.0, False, True, {"predator_x": 0.9, "captures": 1}

    base = NativeXYMock()
    env = wrap_legacy_observation(base)
    obs, info = env.reset(seed=123)
    assert base.received_seed == 123
    assert obs.shape == (10,) and obs[2] == -2.5
    assert info == {"captures": 0}
    action = (0.314159, 0.271828)
    obs, reward, terminated, truncated, info = env.step(action)
    assert base.last_action == action
    assert reward == -2.0 and not terminated and truncated
    assert info == {"captures": 1}
    assert obs[0] == pytest.approx(0.3)


def test_warmstart_reads_only_current_observations_and_rejects_test_manifest(tmp_path):
    source = tmp_path / "train.jsonl"
    row = {"schema_version": DATA_SCHEMA, "action_schema": ACTION_SCHEMA,
           "label_mode": "next_position", "observation": [0.2] * 10,
           "action_xy": [999, 999], "reward": "unused", "next_observation": "unused"}
    source.write_text(json.dumps(row) + "\n")
    inputs = load_warmstart_inputs(source)
    assert inputs.observations == (tuple([0.2] * 10),)
    assert inputs.provenance["offline_replay_ingestion"] is False
    assert inputs.provenance["fields_read_for_supervision"] == ["observation"]
    (tmp_path / "manifest.json").write_text(json.dumps({"artifacts": {"train": {"sha256": "different-train-hash"}}}))
    with pytest.raises(ValueError, match="train split"):
        load_warmstart_inputs(source)


def test_actor_warmstart_matches_bc_predictions_without_touching_critic_or_replay():
    torch = pytest.importorskip("torch")

    class Actor(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.mean = torch.nn.Parameter(torch.zeros(2))

        def forward(self, observations, deterministic=False):
            assert deterministic
            return self.mean.tanh().expand(len(observations), 2)

    class Teacher:
        def reset(self, seed):
            self.seed = seed

        def act(self, observation):
            assert len(observation) == 10
            return ContinuousDecision((0.8, 0.2))

    obs_space, action_space = spaces()
    critic = torch.nn.Linear(10, 1)
    original_critic = {name: value.clone() for name, value in critic.state_dict().items()}
    model = SimpleNamespace(actor=Actor(), critic=critic,
                            observation_space=obs_space, action_space=action_space,
                            replay_buffer=SimpleNamespace(size=lambda: 0))
    summary = warmstart_actor(model, Teacher(), [[0.2] * 10] * 3,
                             steps=30, batch_size=3, learning_rate=0.05)
    assert summary["last_batch_scaled_mse"] < summary["first_batch_scaled_mse"]
    assert summary["replay_size_after"] == summary["replay_size_before"] == 0
    assert all(torch.equal(critic.state_dict()[name], value) for name, value in original_critic.items())
    action = (model.actor(torch.zeros((1, 10)), deterministic=True).detach()[0].numpy() + 1) / 2
    assert action[0] > 0.7 and action[1] < 0.3


class FakeSAC:
    def __init__(self, policy, env, **kwargs):
        self.observation_space, self.action_space = env.observation_space, env.action_space
        self.replay_buffer = SimpleNamespace(size=lambda: 0)
        self.num_timesteps = 0
        self.kwargs = kwargs

    def learn(self, *, total_timesteps, progress_bar):
        assert self.replay_buffer.size() == 0
        self.num_timesteps = total_timesteps
        return self

    def save(self, file):
        file.write(b"mock-saved-model")

    @classmethod
    def load(cls, path, *, device):
        assert path.read_bytes() == b"mock-saved-model"
        obs_space, action_space = spaces()
        return cls("MlpPolicy", SimpleNamespace(observation_space=obs_space, action_space=action_space))

    def predict(self, observation, *, deterministic):
        assert observation.shape == (10,)
        assert deterministic
        return np.asarray([0.2, 0.8], dtype=np.float32), None

    def set_random_seed(self, seed):
        self.seed = seed


def test_training_contract_and_checkpoint_roundtrip_with_offline_mocks(tmp_path):
    obs_space, action_space = spaces()
    events = {}

    def factory(**config):
        events["environment"] = config
        return SimpleNamespace(observation_space=obs_space, action_space=action_space,
                               close=lambda: events.update(closed=True))

    output = tmp_path / "sac-run"
    args = build_parser().parse_args(["train", "--output-dir", str(output), "--total-timesteps", "20", "--seed", "7"])
    metadata = train_teacher(args, env_factory=factory, sac_class=FakeSAC)
    assert events["environment"] == {"world": "21_05", "max_steps": 300, "time_step": 0.25, "predator_speed_ratio": 0.15}
    assert events["closed"]
    assert metadata["completed_timesteps"] == 20
    assert metadata["policy_role"] == POLICY_ROLE
    assert metadata["source_bc_checkpoint"] is None
    assert metadata["offline_replay_ingestion"] is False
    assert metadata["closed_loop_evaluated"] is False
    assert metadata["injected_test_dependencies"] is True
    assert output.stat().st_mode & 0o777 == 0o700
    assert all(path.stat().st_mode & 0o777 == 0o600 for path in output.iterdir())
    policy = SACCheckpointPolicy(output / "sac_teacher.zip", _sac_class=FakeSAC)
    policy.reset(42)
    assert policy.model.seed == 42
    decision = policy.act([0.1] * 10)
    assert decision.action == pytest.approx((0.2, 0.8))
    assert "skill" not in decision.metadata


def test_checkpoint_contract_or_hash_mismatch_fails_before_sb3_import(tmp_path):
    checkpoint = tmp_path / "teacher.zip"
    checkpoint.write_bytes(b"mock-saved-model")
    metadata = {"schema_version": TEACHER_SCHEMA, "action_schema": ACTION_SCHEMA,
                "observation_schema": OBSERVATION_SCHEMA, "observation_dim": 10,
                "action_dim": 2, "policy_role": POLICY_ROLE, "checkpoint_sha256": "wrong"}
    sidecar = checkpoint.with_suffix(".metadata.json")
    sidecar.write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match="hash"):
        SACCheckpointPolicy(checkpoint)
    metadata.update(checkpoint_sha256=_sha256(checkpoint), policy_role="go_to_goal_skill")
    sidecar.write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match="contract"):
        SACCheckpointPolicy(checkpoint)


def test_training_refuses_incomplete_warmstart_and_git_outputs_before_dependencies(tmp_path):
    args = build_parser().parse_args(["train", "--output-dir", str(tmp_path / "new"), "--bc-checkpoint", str(tmp_path / "missing.pt")])
    with pytest.raises(ValueError, match="both"):
        train_teacher(args)
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / ".git").write_text("gitdir: /elsewhere")
    args = build_parser().parse_args(["train", "--output-dir", str(repo / "private")])
    with pytest.raises(ValueError, match="Git checkout"):
        train_teacher(args)


def test_optional_real_sb3_actor_api_without_environment_rollouts(tmp_path):
    sb3 = pytest.importorskip("stable_baselines3")
    gym = pytest.importorskip("gymnasium")

    class NeverSteppedEnv(gym.Env):
        observation_space, action_space = spaces()

        def reset(self, **kwargs):
            raise AssertionError("Offline API validation must not reset an environment")

        def step(self, action):
            raise AssertionError("Offline API validation must not simulate")

    model = sb3.SAC("MlpPolicy", NeverSteppedEnv(), buffer_size=16,
                    policy_kwargs={"net_arch": [8]}, device="cpu", seed=42)
    teacher = SimpleNamespace(reset=lambda seed: None, act=lambda obs: ContinuousDecision((0.7, 0.3)))
    summary = warmstart_actor(model, teacher, [[0.1] * 10], steps=2, batch_size=1)
    assert summary["replay_size_before"] == summary["replay_size_after"] == 0
    assert model.num_timesteps == 0
