"""Offline constrained evaluation for MiniMind base and skill-planner LoRA."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from mouse_llm.baselines.planner_mlp import SKILLS, classification_metrics
from mouse_llm.data.planner_schema import parse_skill
from mouse_llm.evaluation.evaluate_policy import _chat_text, generate_response, task_nll
from mouse_llm.hierarchical.minimind_planner import SkillTokenConstraint
from mouse_llm.hierarchical.policy import Skill


def load_rows(path: Path, *, max_samples: int, seed: int) -> list[dict[str, Any]]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    rows.sort(key=lambda row: hashlib.sha256(f"{seed}:{row['anchor_id']}:{row['preference']}".encode()).digest())
    return rows[:max_samples] if max_samples > 0 else rows


def load_model(
    *,
    base_weight: Path,
    lora_weight: Path | None,
    tokenizer_path: Path,
    device: str,
    hidden_size: int,
    num_hidden_layers: int,
):
    from transformers import AutoTokenizer

    from model.model_lora import apply_lora, load_lora
    from model.model_minimind import MiniMindConfig, MiniMindForCausalLM

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    model = MiniMindForCausalLM(
        MiniMindConfig(hidden_size=hidden_size, num_hidden_layers=num_hidden_layers)
    )
    model.load_state_dict(
        torch.load(base_weight, map_location="cpu", weights_only=True), strict=True
    )
    if lora_weight is not None:
        apply_lora(model)
        load_lora(model, lora_weight)
    torch_device = torch.device(device)
    model = model.half().to(torch_device) if torch_device.type == "cuda" else model.float().to(torch_device)
    model.eval()
    return model, tokenizer


def evaluate(
    model: Any,
    tokenizer: Any,
    rows: list[dict[str, Any]],
    *,
    device: str,
    max_seq_len: int,
    max_new_tokens: int,
) -> dict[str, Any]:
    constraint = SkillTokenConstraint(tokenizer)
    targets = []
    predictions = []
    valid = []
    regrets = []
    latencies = []
    nlls = []
    for row in rows:
        conversations = row["conversations"]
        prompt_text = _chat_text(tokenizer, conversations[:-1], generate=True)
        full_text = _chat_text(tokenizer, conversations, generate=False)
        prompt = tokenizer(
            prompt_text,
            return_tensors="pt",
            add_special_tokens=True,
            truncation=True,
            max_length=max_seq_len,
        )
        full = tokenizer(
            full_text,
            return_tensors="pt",
            add_special_tokens=True,
            truncation=True,
            max_length=max_seq_len,
        )
        prompt_length = prompt["input_ids"].shape[1]
        labels = full["input_ids"].clone()
        labels[:, :prompt_length] = -100
        inputs = {
            "input_ids": prompt["input_ids"].to(device),
            "attention_mask": prompt["attention_mask"].to(device),
        }
        nlls.append(
            task_nll(
                model,
                full["input_ids"].to(device),
                labels.to(device),
            )
        )
        response, latency = generate_response(
            model,
            tokenizer,
            inputs,
            max_new_tokens=max_new_tokens,
            action_constraint=constraint,
        )
        parsed = parse_skill(response)
        target = Skill(row["target_skill"])
        prediction = parsed or Skill.GO_TO_GOAL
        targets.append(SKILLS.index(target))
        predictions.append(SKILLS.index(prediction))
        valid.append(float(parsed is not None))
        utilities = row["utilities"]
        regrets.append(float(utilities[prediction.value]) - max(float(value) for value in utilities.values()))
        latencies.append(latency)
    metrics = classification_metrics(np.asarray(targets), np.asarray(predictions))
    metrics.update(
        {
            "sample_count": len(rows),
            "valid_skill_rate": float(np.mean(valid)),
            "counterfactual_regret_chosen_minus_oracle": float(np.mean(regrets)),
            "task_nll": float(np.mean(nlls)),
            "latency_seconds": {
                "p50": float(np.quantile(latencies, 0.50)),
                "p95": float(np.quantile(latencies, 0.95)),
                "p99": float(np.quantile(latencies, 0.99)),
            },
        }
    )
    return metrics


def ablate_rows(rows: list[dict[str, Any]], mode: str) -> list[dict[str, Any]]:
    ablated = copy.deepcopy(rows)
    for row in ablated:
        user = json.loads(row["conversations"][1]["content"])
        if mode == "no_temporal_history":
            history = user["context"]["history"]
            for key in history:
                history[key] = []
        elif mode == "instruction_removed":
            user["instruction"] = ""
            user["preference"] = "unspecified"
        else:
            raise ValueError(f"Unknown planner ablation {mode!r}")
        row["conversations"][1]["content"] = json.dumps(
            user, sort_keys=True, separators=(",", ":")
        )
    return ablated


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate MiniMind skill planners")
    parser.add_argument("--base-weight", type=Path, required=True)
    parser.add_argument("--lora-weight", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--seen-test-data", type=Path, required=True)
    parser.add_argument("--unseen-test-data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--hidden-size", type=int, default=768)
    parser.add_argument("--num-hidden-layers", type=int, default=8)
    parser.add_argument("--max-seq-len", type=int, default=1024)
    parser.add_argument("--max-new-tokens", type=int, default=24)
    parser.add_argument("--max-samples", type=int, default=512)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--research-evidence", action="store_true")
    args = parser.parse_args()
    datasets = {
        "seen_instruction": load_rows(args.seen_test_data, max_samples=args.max_samples, seed=args.seed),
        "unseen_paraphrase": load_rows(args.unseen_test_data, max_samples=args.max_samples, seed=args.seed),
    }
    report: dict[str, Any] = {
        "schema_version": 1,
        "artifact": "p2_offline_minimind_skill_planner",
        "research_evidence": args.research_evidence,
        "research_evidence_blockers": (
            [] if args.research_evidence else ["smoke or incomplete dataset"]
        ),
        "decode_mode": "skill-json-constrained",
        "models": {},
    }
    for name, lora in (("minimind_base", None), ("minimind_skill_lora", args.lora_weight)):
        model, tokenizer = load_model(
            base_weight=args.base_weight,
            lora_weight=lora,
            tokenizer_path=args.tokenizer,
            device=args.device,
            hidden_size=args.hidden_size,
            num_hidden_layers=args.num_hidden_layers,
        )
        report["models"][name] = {
            split: evaluate(
                model,
                tokenizer,
                rows,
                device=args.device,
                max_seq_len=args.max_seq_len,
                max_new_tokens=args.max_new_tokens,
            )
            for split, rows in datasets.items()
        }
        if name == "minimind_skill_lora":
            report["models"][name]["ablations"] = {
                mode: evaluate(
                    model,
                    tokenizer,
                    ablate_rows(datasets["seen_instruction"], mode),
                    device=args.device,
                    max_seq_len=args.max_seq_len,
                    max_new_tokens=args.max_new_tokens,
                )
                for mode in ("no_temporal_history", "instruction_removed")
            }
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, args.output)
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
