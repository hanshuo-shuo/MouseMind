"""Paired baseline-versus-LoRA evaluation on held-out mouse episodes."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


class ActionTokenConstraint:
    """A tokenizer-level trie that permits only valid action JSON outputs."""

    def __init__(self, tokenizer: Any, *, action_count: int):
        if tokenizer.eos_token_id is None:
            raise ValueError("Tokenizer must define eos_token_id for constrained decoding")
        self.eos_token_id = int(tokenizer.eos_token_id)
        self.root: dict[int, dict] = {}
        for action in range(action_count):
            text = json.dumps({"action": action}, separators=(",", ":"))
            tokens = tokenizer.encode(text, add_special_tokens=False)
            if not tokens:
                raise ValueError(f"Tokenizer produced no tokens for {text}")
            node = self.root
            for token in [*tokens, self.eos_token_id]:
                node = node.setdefault(int(token), {})

    def prefix_allowed_tokens_fn(self, *, prompt_length: int):
        """Build the callback expected by ``transformers.generate``."""

        def allowed(_batch_id: int, input_ids: Any) -> list[int]:
            suffix = input_ids.tolist()[prompt_length:]
            node = self.root
            for token in suffix:
                if int(token) not in node:
                    return [self.eos_token_id]
                node = node[int(token)]
            return sorted(node) if node else [self.eos_token_id]

        return allowed


@dataclass(frozen=True)
class EvalSample:
    sample_id: str
    conversations: list[dict[str, str]]
    target_action: int


def parse_action(response: str, *, action_count: int) -> int | None:
    try:
        payload = json.loads(response.strip())
    except json.JSONDecodeError:
        return None
    if not isinstance(payload, dict) or set(payload) != {"action"}:
        return None
    action = payload["action"]
    if isinstance(action, bool) or not isinstance(action, int):
        return None
    return action if 0 <= action < action_count else None


def load_samples(path: Path, *, max_samples: int, seed: int) -> list[EvalSample]:
    samples: list[EvalSample] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            payload = json.loads(line)
            conversations = payload["conversations"]
            if len(conversations) < 3 or conversations[-1].get("role") != "assistant":
                raise ValueError(f"Line {line_number}: expected a final assistant target")
            target_payload = json.loads(conversations[-1]["content"])
            target_action = target_payload.get("action")
            if isinstance(target_action, bool) or not isinstance(target_action, int):
                raise ValueError(f"Line {line_number}: invalid target action")
            canonical = json.dumps(conversations, sort_keys=True, separators=(",", ":"))
            sample_id = hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]
            samples.append(
                EvalSample(
                    sample_id=sample_id,
                    conversations=conversations,
                    target_action=target_action,
                )
            )
    if not samples:
        raise ValueError("No evaluation samples found")
    if max_samples > 0 and len(samples) > max_samples:
        samples.sort(
            key=lambda sample: hashlib.sha256(
                f"{seed}:{sample.sample_id}".encode("utf-8")
            ).digest()
        )
        samples = samples[:max_samples]
    return sorted(samples, key=lambda sample: sample.sample_id)


def load_action_catalog(path: Path) -> np.ndarray:
    payload = json.loads(path.read_text(encoding="utf-8"))
    actions = payload["actions"]
    destinations = np.asarray([item["destination"] for item in actions], dtype=np.float64)
    expected_ids = list(range(len(actions)))
    actual_ids = [int(item["action"]) for item in actions]
    if actual_ids != expected_ids or destinations.shape != (len(actions), 2):
        raise ValueError("Action catalog must contain contiguous 2D destinations")
    return destinations


def _chat_text(tokenizer: Any, messages: list[dict[str, str]], *, generate: bool) -> str:
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=generate,
        open_thinking=False,
    )


def encode_sample(
    tokenizer: Any, sample: EvalSample, *, device: str, max_seq_len: int
) -> tuple[dict[str, torch.Tensor], torch.Tensor, torch.Tensor]:
    import torch

    prompt_text = _chat_text(tokenizer, sample.conversations[:-1], generate=True)
    full_text = _chat_text(tokenizer, sample.conversations, generate=False)
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
    prompt_ids = prompt["input_ids"][0]
    full_ids = full["input_ids"][0]
    if len(full_ids) <= len(prompt_ids) or not torch.equal(
        full_ids[: len(prompt_ids)], prompt_ids
    ):
        raise ValueError(
            "Tokenizer chat template is not prefix-consistent for evaluation sample "
            f"{sample.sample_id}"
        )
    labels = full_ids.clone()
    labels[: len(prompt_ids)] = -100
    prompt_inputs = {
        "input_ids": prompt["input_ids"].to(device),
        "attention_mask": prompt["attention_mask"].to(device),
    }
    return prompt_inputs, full_ids.unsqueeze(0).to(device), labels.unsqueeze(0).to(device)


def generate_response(
    model: MiniMindForCausalLM,
    tokenizer: Any,
    prompt_inputs: dict[str, torch.Tensor],
    *,
    max_new_tokens: int,
    action_constraint: ActionTokenConstraint | None = None,
) -> tuple[str, float]:
    import torch

    start = time.perf_counter()
    with torch.inference_mode():
        if action_constraint is not None:
            # MiniMind's custom GenerationMixin path has changed across
            # Transformers releases and has previously ignored
            # prefix_allowed_tokens_fn. Apply the trie mask explicitly so
            # "constrained" always means constrained, independent of that
            # integration detail.
            output = prompt_inputs["input_ids"]
            attention_mask = prompt_inputs["attention_mask"]
            prompt_length = output.shape[1]
            allowed_tokens = action_constraint.prefix_allowed_tokens_fn(
                prompt_length=prompt_length
            )
            for _ in range(max_new_tokens):
                logits = model(
                    output,
                    attention_mask=attention_mask,
                ).logits[:, -1, :]
                allowed = allowed_tokens(0, output[0])
                allowed_tensor = torch.as_tensor(allowed, device=logits.device)
                next_token = allowed_tensor[
                    logits[0, allowed_tensor].argmax()
                ].reshape(1, 1)
                output = torch.cat((output, next_token), dim=1)
                attention_mask = torch.cat(
                    (
                        attention_mask,
                        torch.ones(
                            (attention_mask.shape[0], 1),
                            dtype=attention_mask.dtype,
                            device=attention_mask.device,
                        ),
                    ),
                    dim=1,
                )
                if int(next_token.item()) == int(tokenizer.eos_token_id):
                    break
        else:
            output = model.generate(
                inputs=prompt_inputs["input_ids"],
                attention_mask=prompt_inputs["attention_mask"],
                max_new_tokens=max_new_tokens,
                do_sample=False,
                temperature=1.0,
                top_p=1.0,
                top_k=0,
                repetition_penalty=1.0,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
    elapsed = time.perf_counter() - start
    generated = output[0, prompt_inputs["input_ids"].shape[1] :]
    return tokenizer.decode(generated, skip_special_tokens=True), elapsed


def task_nll(
    model: MiniMindForCausalLM, full_ids: torch.Tensor, labels: torch.Tensor
) -> float:
    import torch

    attention_mask = torch.ones_like(full_ids)
    with torch.inference_mode():
        output = model(full_ids, attention_mask=attention_mask, labels=labels)
    return float(output.loss.detach().float().cpu())


def _bootstrap_mean(
    values: list[float], *, seed: int, iterations: int = 2000
) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0:
        return {"mean": float("nan"), "ci_low": float("nan"), "ci_high": float("nan")}
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, array.size, size=(iterations, array.size))
    means = array[indices].mean(axis=1)
    return {
        "mean": float(array.mean()),
        "ci_low": float(np.quantile(means, 0.025)),
        "ci_high": float(np.quantile(means, 0.975)),
    }


def summarize(rows: list[dict[str, Any]], *, seed: int) -> dict[str, Any]:
    targets = [int(row["target_action"]) for row in rows]
    predictions = [row["predicted_action"] for row in rows]
    per_class: list[float] = []
    for target in sorted(set(targets)):
        indices = [index for index, value in enumerate(targets) if value == target]
        if len(indices) >= 2:
            per_class.append(
                sum(predictions[index] == target for index in indices) / len(indices)
            )
    latencies = np.asarray(
        [float(row["latency_seconds"]) for row in rows], dtype=np.float64
    )
    return {
        "sample_count": len(rows),
        "valid_output_rate": _bootstrap_mean(
            [float(row["valid_output"]) for row in rows], seed=seed + 1
        ),
        "exact_action_accuracy": _bootstrap_mean(
            [float(row["correct_action"]) for row in rows], seed=seed + 2
        ),
        "task_nll": _bootstrap_mean(
            [float(row["task_nll"]) for row in rows], seed=seed + 3
        ),
        "normalized_destination_error": _bootstrap_mean(
            [float(row["normalized_destination_error"]) for row in rows],
            seed=seed + 4,
        ),
        "macro_action_recall": float(np.mean(per_class)) if per_class else 0.0,
        "latency_seconds": {
            "p50": float(np.quantile(latencies, 0.50)),
            "p95": float(np.quantile(latencies, 0.95)),
            "p99": float(np.quantile(latencies, 0.99)),
            "mean": float(latencies.mean()),
        },
        # Kept for compatibility with the first verified Quest report.
        "median_latency_seconds": float(np.median(latencies)),
    }


def evaluate_model(
    model: MiniMindForCausalLM,
    tokenizer: Any,
    samples: list[EvalSample],
    destinations: np.ndarray,
    *,
    device: str,
    max_seq_len: int,
    max_new_tokens: int,
    action_constraint: ActionTokenConstraint | None = None,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    action_count = len(destinations)
    max_distance = math.sqrt(2.0)
    for index, sample in enumerate(samples, start=1):
        prompt_inputs, full_ids, labels = encode_sample(
            tokenizer, sample, device=device, max_seq_len=max_seq_len
        )
        nll = task_nll(model, full_ids, labels)
        response, latency = generate_response(
            model,
            tokenizer,
            prompt_inputs,
            max_new_tokens=max_new_tokens,
            action_constraint=action_constraint,
        )
        predicted = parse_action(response, action_count=action_count)
        valid = predicted is not None
        correct = predicted == sample.target_action
        if predicted is None:
            normalized_error = 1.0
        else:
            distance = np.linalg.norm(
                destinations[predicted] - destinations[sample.target_action]
            )
            normalized_error = min(float(distance / max_distance), 1.0)
        rows.append(
            {
                "sample_id": sample.sample_id,
                "target_action": sample.target_action,
                "predicted_action": predicted,
                "valid_output": valid,
                "correct_action": correct,
                "normalized_destination_error": normalized_error,
                "task_nll": nll,
                "latency_seconds": latency,
                "response": response[:512],
            }
        )
        if index % 25 == 0 or index == len(samples):
            print(f"evaluated {index}/{len(samples)}", flush=True)
    return rows


def _write_json(path: Path, payload: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _write_predictions(
    path: Path, base_rows: list[dict[str, Any]], lora_rows: list[dict[str, Any]]
) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for base, lora in zip(base_rows, lora_rows, strict=True):
            handle.write(
                json.dumps(
                    {"sample_id": base["sample_id"], "base": base, "lora": lora},
                    ensure_ascii=False,
                    separators=(",", ":"),
                )
                + "\n"
            )
    os.replace(temporary, path)


def evidence_metadata(test_data: Path) -> dict[str, Any]:
    manifest_path = test_data.parent / "manifest.json"
    if not manifest_path.is_file():
        return {
            "synthetic": None,
            "research_evidence": False,
            "evidence_note": "dataset manifest not found",
        }
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    synthetic = bool(manifest.get("synthetic", False))
    return {
        "synthetic": synthetic,
        "research_evidence": bool(
            manifest.get("research_evidence", not synthetic)
        ),
        "evidence_note": manifest.get("purpose", "held-out dataset evaluation"),
    }


def main() -> None:
    import torch
    from transformers import AutoTokenizer

    from model.model_lora import apply_lora, load_lora
    from model.model_minimind import MiniMindConfig, MiniMindForCausalLM

    parser = argparse.ArgumentParser()
    parser.add_argument("--base-weight", type=Path, required=True)
    parser.add_argument("--lora-weight", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--test-data", type=Path, required=True)
    parser.add_argument("--action-catalog", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--hidden-size", type=int, default=768)
    parser.add_argument("--num-hidden-layers", type=int, default=8)
    parser.add_argument("--max-seq-len", type=int, default=256)
    parser.add_argument("--max-new-tokens", type=int, default=24)
    parser.add_argument("--max-samples", type=int, default=512)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--decode-mode",
        choices=("free", "json-constrained"),
        default="free",
        help=(
            "Use token-level constrained decoding to remove JSON formatting as a "
            "comparison confound. 'free' reproduces the original reported run."
        ),
    )
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    samples = load_samples(
        args.test_data, max_samples=args.max_samples, seed=args.seed
    )
    destinations = load_action_catalog(args.action_catalog)
    action_constraint = (
        ActionTokenConstraint(tokenizer, action_count=len(destinations))
        if args.decode_mode == "json-constrained"
        else None
    )
    if any(not 0 <= sample.target_action < len(destinations) for sample in samples):
        raise ValueError("Test data contains an action outside the catalog")

    config = MiniMindConfig(
        hidden_size=args.hidden_size,
        num_hidden_layers=args.num_hidden_layers,
    )
    model = MiniMindForCausalLM(config)
    state = torch.load(args.base_weight, map_location="cpu")
    model.load_state_dict(state, strict=True)
    model = model.half().to(device) if device == "cuda" else model.float().to(device)
    model.eval()

    print("evaluating baseline", flush=True)
    base_rows = evaluate_model(
        model,
        tokenizer,
        samples,
        destinations,
        device=device,
        max_seq_len=args.max_seq_len,
        max_new_tokens=args.max_new_tokens,
        action_constraint=action_constraint,
    )

    apply_lora(model)
    load_lora(model, args.lora_weight)
    model = model.half().to(device) if device == "cuda" else model.float().to(device)
    model.eval()
    print("evaluating LoRA", flush=True)
    lora_rows = evaluate_model(
        model,
        tokenizer,
        samples,
        destinations,
        device=device,
        max_seq_len=args.max_seq_len,
        max_new_tokens=args.max_new_tokens,
        action_constraint=action_constraint,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
    args.output_dir.chmod(0o700)
    metrics = {
        "schema_version": 1,
        "experiment": "minimind_mouse_policy_lora",
        **evidence_metadata(args.test_data),
        "evaluation": {
            "split": "held-out episodes",
            "sample_count": len(samples),
            "seed": args.seed,
            "invalid_destination_error": 1.0,
            "decode_mode": args.decode_mode,
        },
        "base": summarize(base_rows, seed=args.seed + 100),
        "lora": summarize(lora_rows, seed=args.seed + 200),
    }
    metrics_path = args.output_dir / "comparison_metrics.json"
    predictions_path = args.output_dir / "predictions_private.jsonl"
    _write_json(metrics_path, metrics)
    _write_predictions(predictions_path, base_rows, lora_rows)
    metrics_path.chmod(0o600)
    predictions_path.chmod(0o600)
    print(json.dumps(metrics, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
