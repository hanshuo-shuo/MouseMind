"""One-config full-parameter verified GRPO from seed-clean full SFT."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
from collections import Counter
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from mouse_llm.data.build_verified_dpo import seed_split
from mouse_llm.data.planner_schema import Preference, instruction_for, planner_messages
from mouse_llm.evaluation.closed_loop import MLPCheckpointPolicy
from mouse_llm.evaluation.contracts import DEFAULT_P2_CONTRACT, contract_seeds
from mouse_llm.evaluation.evaluate_policy import load_action_catalog
from mouse_llm.verified_rl.reward import RL_PREFERENCES, reward_components
from mouse_llm.verified_rl.rollout import ReplayVerificationError, RolloutVerifier
from mouse_llm.verified_rl.sampler import SKILLS, sample_skills, score_skill_distributions


def anchor_id(anchor: dict[str, Any]) -> str:
    key = json.dumps(
        {
            "seed": int(anchor["seed"]),
            "step_index": int(anchor["step_index"]),
            "prefix_actions": list(anchor["prefix_actions"]),
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(key.encode("utf-8")).hexdigest()[:20]


def load_train_anchors(
    anchors_path: Path,
    verified_path: Path,
    contract_path: Path,
    *,
    source_pool: str,
    horizon: int,
    max_anchors: int = 0,
) -> list[dict[str, Any]]:
    allowed = set(contract_seeds(source_pool, contract_path))
    final = set(contract_seeds("final_id_test", contract_path))
    records = [json.loads(line) for line in verified_path.read_text().splitlines() if line]
    verified: dict[str, dict[str, Any]] = {}
    for row in records:
        key = row["anchor_id"]
        if key in verified:
            raise ValueError(f"Duplicate verified anchor {key}")
        if not row.get("replay_verification", {}).get("verified", False):
            raise ValueError(f"Counterfactual anchor {key} was not verified")
        if str(horizon) not in row.get("branches_by_horizon", {}):
            raise ValueError(f"Missing verified horizon {horizon} for {key}")
        verified[key] = row
    anchors = [json.loads(line) for line in anchors_path.read_text().splitlines() if line]
    selected: list[dict[str, Any]] = []
    seen: set[str] = set()
    for anchor in anchors:
        seed = int(anchor["seed"])
        if seed in final or seed not in allowed:
            raise ValueError(f"Anchor seed {seed} is outside permitted {source_pool} pool")
        key = anchor_id(anchor)
        if key in seen:
            raise ValueError(f"Duplicate replay anchor {key}")
        seen.add(key)
        record = verified.get(key)
        if record is None or int(record["seed"]) != seed or int(record["step_index"]) != int(anchor["step_index"]):
            raise ValueError(f"Replay anchor {key} lacks matching verified counterfactual")
        if seed_split(seed) == "train":
            selected.append(anchor)
    if not selected:
        raise ValueError("No verified training anchors in the requested pool")
    selected.sort(key=lambda row: anchor_id(row))
    return selected[:max_anchors] if max_anchors else selected


def make_model(config: Any, sft_full_weight: Path, device: torch.device, *, trainable: bool):
    from model.model_minimind import MiniMindForCausalLM

    model = MiniMindForCausalLM(config)
    model.load_state_dict(torch.load(sft_full_weight, map_location="cpu", weights_only=True), strict=True)
    model.to(device)
    for parameter in model.parameters():
        parameter.requires_grad_(trainable)
    model.eval()  # deterministic likelihoods with gradients for the policy
    return model


def group_advantages(rewards: list[float], *, threshold: float = 1e-6) -> torch.Tensor | None:
    values = torch.tensor(rewards, dtype=torch.float32)
    std = values.std(unbiased=False)
    if float(std) <= threshold:
        return None
    return (values - values.mean()) / (std + 1e-6)


def grpo_loss(
    current_logps: torch.Tensor,
    old_logps: torch.Tensor,
    advantages: torch.Tensor,
    current_distribution: torch.Tensor,
    reference_distribution: torch.Tensor,
    *,
    clip_epsilon: float,
    kl_coefficient: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    ratio = torch.exp(current_logps - old_logps)
    objective = torch.minimum(
        ratio * advantages,
        ratio.clamp(1.0 - clip_epsilon, 1.0 + clip_epsilon) * advantages,
    )
    # Exact categorical KL is available because the valid action set has size 3.
    kl = (current_distribution.exp() * (current_distribution - reference_distribution)).sum()
    return -objective.mean() + kl_coefficient * kl, kl


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def adapt_kl_coefficient(
    coefficient: float, *,
    observed_kl: float,
    target_kl: float,
    initial_coefficient: float,
) -> float:
    """Keep the exact three-skill KL near a development-set-free target."""
    if target_kl <= 0:
        return coefficient
    if observed_kl > 1.5 * target_kl:
        return min(coefficient * 1.5, 2.0)
    if observed_kl < target_kl / 1.5:
        return max(coefficient / 1.5, initial_coefficient)
    return coefficient


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--anchors", type=Path, required=True, help="Raw anchors with prefix_actions")
    parser.add_argument("--verified-counterfactuals", type=Path, required=True)
    parser.add_argument("--contract", type=Path, default=DEFAULT_P2_CONTRACT)
    parser.add_argument("--source-pool", choices=("data_collection", "development"), default="data_collection")
    parser.add_argument("--sft-full-weight", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--mlp-checkpoint", type=Path, required=True)
    parser.add_argument("--action-catalog", type=Path, default=Path(__file__).resolve().parents[1] / "envs/mice/assets/action_catalog_21_05.json")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--group-size", type=int, default=6)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--rollout-horizon", type=int, default=8)
    parser.add_argument("--clip-epsilon", type=float, default=0.2)
    parser.add_argument("--kl-coefficient", type=float, default=0.01)
    parser.add_argument("--target-kl", type=float, default=0.0, help="0 disables adaptive KL control")
    parser.add_argument("--policy-update-epochs", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=2e-6)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--max-anchors", type=int, default=0)
    parser.add_argument("--max-seq-len", type=int, default=1024)
    parser.add_argument("--replay-tolerance", type=float, default=1e-7)
    parser.add_argument("--evade-distance", type=float, default=0.35)
    parser.add_argument("--world", default="21_05")
    parser.add_argument("--max-steps", type=int, default=300)
    parser.add_argument("--time-step", type=float, default=0.25)
    parser.add_argument("--predator-speed-ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    if args.group_size < 2 or args.temperature <= 0 or args.rollout_horizon <= 0 or args.epochs < 1 or args.max_anchors < 0 or args.max_seq_len < 32 or not 0 < args.clip_epsilon < 1 or args.kl_coefficient < 0 or args.target_kl < 0 or args.policy_update_epochs < 1 or args.learning_rate <= 0:
        parser.error("Invalid GRPO optimization parameter")
    from transformers import AutoTokenizer
    from model.model_minimind import MiniMindConfig
    from mouse_llm.envs.mice import BotEvadeEnv, custom_reward

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    anchors = load_train_anchors(
        args.anchors, args.verified_counterfactuals, args.contract,
        source_pool=args.source_pool, horizon=args.rollout_horizon,
        max_anchors=args.max_anchors,
    )
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    config = MiniMindConfig(hidden_size=768, num_hidden_layers=8)
    policy = make_model(config, args.sft_full_weight, device, trainable=True)
    reference = make_model(config, args.sft_full_weight, device, trainable=False)
    parameters = [parameter for parameter in policy.parameters() if parameter.requires_grad]
    if not parameters:
        raise RuntimeError("No trainable model parameters")
    optimizer = torch.optim.AdamW(parameters, lr=args.learning_rate)
    specialist = MLPCheckpointPolicy(args.mlp_checkpoint, device=args.device, observation_indices=tuple(range(10)), name="verified-grpo-specialist")
    destinations = load_action_catalog(args.action_catalog)

    def env_factory():
        return BotEvadeEnv(
            world_name=args.world,
            use_lppos=False,
            use_predator=True,
            max_step=args.max_steps,
            reward_function=custom_reward,
            time_step=args.time_step,
            frame_stack_k=1,
            predator_prey_forward_speed_ratio=args.predator_speed_ratio,
        )

    verifier = RolloutVerifier(
        env_factory, specialist, destinations,
        horizon=args.rollout_horizon, evade_distance=args.evade_distance,
        replay_tolerance=args.replay_tolerance,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
    log_path = args.output_dir / "groups.jsonl"
    total = skipped = steps = 0
    skipped_replay_anchors: list[dict[str, Any]] = []
    all_rewards: list[float] = []
    all_stds: list[float] = []
    all_capture: list[float] = []
    all_goal: list[float] = []
    all_kl: list[float] = []
    all_clip_fraction: list[float] = []
    kl_coefficient = args.kl_coefficient
    kl_ema: float | None = None
    counts: Counter[str] = Counter()
    probabilities: dict[str, list[float]] = {skill.value: [] for skill in SKILLS}
    with log_path.open("w", encoding="utf-8") as log:
        for epoch in range(args.epochs):
            order = list(anchors)
            random.shuffle(order)
            for anchor in order:
                try:
                    # No policy sample from this anchor is used unless an exact
                    # fresh replay passes the original 1e-7 state check.
                    verifier.verify_anchor(anchor)
                except ReplayVerificationError as exc:
                    failure = {
                        "epoch": epoch + 1, "anchor_id": anchor_id(anchor),
                        "seed": int(anchor["seed"]),
                        "step_index": int(anchor["step_index"]),
                        "error": str(exc),
                    }
                    skipped_replay_anchors.append(failure)
                    print(f"skipping unverified anchor {failure}", flush=True)
                    continue
                preferences = list(RL_PREFERENCES)
                random.shuffle(preferences)
                for preference in preferences:
                    key = anchor_id(anchor)
                    instruction = instruction_for(preference, split="train", stable_key=key)
                    prompt = planner_messages(anchor["context"], preference=preference, instruction=instruction)
                    samples, probs = sample_skills(
                        policy, tokenizer, prompt, group_size=args.group_size,
                        temperature=args.temperature, max_seq_len=args.max_seq_len,
                    )
                    try:
                        outcomes, rewards = verifier.evaluate_group(
                            anchor, [sample.skill for sample in samples], preference.value
                        )
                    except ReplayVerificationError as exc:
                        failure = {
                            "epoch": epoch + 1, "anchor_id": key,
                            "seed": int(anchor["seed"]),
                            "step_index": int(anchor["step_index"]),
                            "preference": preference.value,
                            "error": str(exc),
                        }
                        skipped_replay_anchors.append(failure)
                        print(f"skipping unverified anchor {failure}", flush=True)
                        break
                    advantages = group_advantages(rewards)
                    total += 1
                    counts.update(sample.skill.value for sample in samples)
                    for skill in SKILLS:
                        probabilities[skill.value].append(probs[skill.value])
                    components = [reward_components(outcome, preference) for outcome in outcomes]
                    reward_tensor = torch.tensor(rewards, dtype=torch.float32)
                    all_rewards.extend(rewards)
                    all_stds.append(float(reward_tensor.std(unbiased=False)))
                    all_capture.extend(item["capture"] for item in components)
                    all_goal.extend(item["goal_progress"] for item in components)
                    log_row: dict[str, Any] = {
                        "epoch": epoch + 1, "anchor_id": key, "seed": int(anchor["seed"]),
                        "preference": preference.value, "skills": [sample.skill.value for sample in samples],
                        "sequence_log_probabilities": [sample.sequence_log_probability for sample in samples],
                        "rewards": rewards, "reward_mean": _mean(rewards),
                        "reward_std": all_stds[-1], "skill_probabilities": probs,
                        "capture_component_mean": _mean([item["capture"] for item in components]),
                        "goal_progress_component_mean": _mean([item["goal_progress"] for item in components]),
                        "unique_rollouts": len({sample.skill for sample in samples}),
                        "replay_attempts": verifier.last_replay_attempts,
                        "skipped": advantages is None,
                    }
                    if advantages is None:
                        skipped += 1
                        log_row["kl_to_sft"] = None
                    else:
                        indices = torch.tensor([SKILLS.index(sample.skill) for sample in samples], device=device)
                        old = torch.tensor([sample.constrained_log_probability for sample in samples], device=device)
                        advantage_device = advantages.to(device)
                        with torch.no_grad():
                            _, ref_branch_logps = score_skill_distributions(reference, tokenizer, prompt, max_seq_len=args.max_seq_len)
                            ref_distribution = F.log_softmax(ref_branch_logps / args.temperature, dim=0)
                        clip_fractions = []
                        for _ in range(args.policy_update_epochs):
                            _, branch_logps = score_skill_distributions(policy, tokenizer, prompt, max_seq_len=args.max_seq_len)
                            distribution = F.log_softmax(branch_logps / args.temperature, dim=0)
                            ratios = torch.exp(distribution[indices] - old)
                            clip_fractions.append(float(((ratios < 1.0 - args.clip_epsilon) | (ratios > 1.0 + args.clip_epsilon)).float().mean().detach()))
                            loss, kl = grpo_loss(
                                distribution[indices], old, advantage_device,
                                distribution, ref_distribution,
                                clip_epsilon=args.clip_epsilon, kl_coefficient=kl_coefficient,
                            )
                            optimizer.zero_grad(set_to_none=True)
                            loss.backward()
                            torch.nn.utils.clip_grad_norm_(parameters, 1.0)
                            optimizer.step()
                            steps += 1
                        log_row["loss"] = float(loss.detach())
                        log_row["kl_to_sft"] = float(kl.detach())
                        log_row["clip_fraction"] = _mean(clip_fractions)
                        log_row["kl_coefficient"] = kl_coefficient
                        all_kl.append(log_row["kl_to_sft"])
                        all_clip_fraction.append(log_row["clip_fraction"])
                        kl_ema = log_row["kl_to_sft"] if kl_ema is None else 0.9 * kl_ema + 0.1 * log_row["kl_to_sft"]
                        if args.target_kl > 0 and steps % (25 * args.policy_update_epochs) == 0:
                            kl_coefficient = adapt_kl_coefficient(
                                kl_coefficient,
                                observed_kl=kl_ema,
                                target_kl=args.target_kl,
                                initial_coefficient=args.kl_coefficient,
                            )
                    log.write(json.dumps(log_row, sort_keys=True) + "\n")
                    log.flush()
                    if total % 25 == 0:
                        print(f"groups={total} skipped={skipped / total:.3f} reward={_mean(all_rewards):.3f} KL={_mean(all_kl):.4f} skills={dict(counts)}", flush=True)
    weight = args.output_dir / "full_verified_grpo_768.pth"
    torch.save({key: value.detach().cpu() for key, value in policy.state_dict().items()}, weight)
    os.chmod(weight, 0o600)
    summary = {
        "initial_policy_and_reference": str(args.sft_full_weight), "method": "full_grpo",
        "train_anchors": len(anchors), "source_pool": args.source_pool,
        "skipped_replay_anchors": skipped_replay_anchors,
        "skipped_replay_anchor_count": len(skipped_replay_anchors),
        "epochs": args.epochs, "groups": total, "optimizer_steps": steps,
        "fraction_groups_skipped": skipped / total,
        "mean_verified_reward": _mean(all_rewards),
        "mean_group_reward_std": _mean(all_stds),
        "mean_capture_reward_component": _mean(all_capture),
        "mean_goal_progress_component": _mean(all_goal),
        "mean_kl_to_full_sft_reference": _mean(all_kl),
        "mean_clip_fraction": _mean(all_clip_fraction),
        "final_kl_coefficient": kl_coefficient,
        "target_kl": args.target_kl,
        "policy_update_epochs": args.policy_update_epochs,
        "sampled_skill_distribution": {skill.value: counts[skill.value] / (total * args.group_size) for skill in SKILLS},
        "mean_skill_probabilities": {skill.value: _mean(probabilities[skill.value]) for skill in SKILLS},
        "group_size": args.group_size, "temperature": args.temperature,
        "rollout_horizon": args.rollout_horizon, "clip_epsilon": args.clip_epsilon,
        "kl_coefficient": args.kl_coefficient, "learning_rate": args.learning_rate,
        "seed": args.seed, "output_weight": str(weight),
    }
    (args.output_dir / "training.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    os.chmod(args.output_dir / "training.json", 0o600)
    os.chmod(log_path, 0o600)
    print(json.dumps(summary, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
