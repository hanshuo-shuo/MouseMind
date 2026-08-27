"""Constrained MiniMind high-level skill planner."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch

from mouse_llm.data.planner_schema import (
    Preference,
    instruction_for,
    parse_skill,
    planner_messages,
)
from mouse_llm.evaluation.evaluate_policy import _chat_text, generate_response
from mouse_llm.hierarchical.context import PlannerContext
from mouse_llm.hierarchical.policy import PlannerDecision, Skill


class SkillTokenConstraint:
    """Tokenizer trie permitting only the three canonical skill JSON strings."""

    def __init__(self, tokenizer: Any):
        if tokenizer.eos_token_id is None:
            raise ValueError("Tokenizer must define eos_token_id")
        self.eos_token_id = int(tokenizer.eos_token_id)
        self.root: dict[int, dict] = {}
        for skill in Skill:
            text = json.dumps({"skill": skill.value}, separators=(",", ":"))
            tokens = tokenizer.encode(text, add_special_tokens=False)
            if not tokens:
                raise ValueError(f"Tokenizer produced no tokens for {text}")
            node = self.root
            for token in [*tokens, self.eos_token_id]:
                node = node.setdefault(int(token), {})

    def prefix_allowed_tokens_fn(self, *, prompt_length: int):
        def allowed(_batch_id: int, input_ids: Any) -> list[int]:
            suffix = input_ids.tolist()[prompt_length:]
            node = self.root
            for token in suffix:
                if int(token) not in node:
                    return [self.eos_token_id]
                node = node[int(token)]
            return sorted(node) if node else [self.eos_token_id]

        return allowed


class MiniMindSkillPlanner:
    def __init__(
        self,
        *,
        base_weight: Path,
        tokenizer_path: Path,
        device: str,
        lora_weight: Path | None = None,
        hidden_size: int = 768,
        num_hidden_layers: int = 8,
        max_seq_len: int = 1024,
        max_new_tokens: int = 24,
        instruction_split: str = "train",
        ablation: str | None = None,
    ):
        from transformers import AutoTokenizer

        from model.model_lora import apply_lora, load_lora
        from model.model_minimind import MiniMindConfig, MiniMindForCausalLM

        self.device = torch.device(device)
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        config = MiniMindConfig(
            hidden_size=hidden_size, num_hidden_layers=num_hidden_layers
        )
        model = MiniMindForCausalLM(config)
        state = torch.load(base_weight, map_location="cpu", weights_only=True)
        model.load_state_dict(state, strict=True)
        if lora_weight is not None:
            apply_lora(model)
            load_lora(model, lora_weight)
        self.model = (
            model.half().to(self.device)
            if self.device.type == "cuda"
            else model.float().to(self.device)
        )
        self.model.eval()
        self.constraint = SkillTokenConstraint(self.tokenizer)
        self.max_seq_len = max_seq_len
        self.max_new_tokens = max_new_tokens
        self.instruction_split = instruction_split
        if ablation not in {None, "no_temporal_history", "instruction_removed"}:
            raise ValueError(f"Unknown MiniMind planner ablation {ablation!r}")
        self.ablation = ablation
        self.last_response: str | None = None
        self.last_valid = True

    def reset(self, seed: int) -> None:
        torch.manual_seed(seed)

    def plan_context(
        self, context: PlannerContext, preference: Preference
    ) -> PlannerDecision:
        instruction = instruction_for(
            preference,
            split=self.instruction_split,
            stable_key=context.serialize(),
        )
        messages = planner_messages(
            context,
            preference=preference,
            instruction=instruction,
        )
        if self.ablation is not None:
            user = json.loads(messages[1]["content"])
            if self.ablation == "no_temporal_history":
                for key in user["context"]["history"]:
                    user["context"]["history"][key] = []
            else:
                user["instruction"] = ""
                user["preference"] = "unspecified"
            messages[1]["content"] = json.dumps(
                user, sort_keys=True, separators=(",", ":")
            )
        prompt = _chat_text(self.tokenizer, messages, generate=True)
        encoded = self.tokenizer(
            prompt,
            return_tensors="pt",
            add_special_tokens=True,
            truncation=True,
            max_length=self.max_seq_len,
        )
        inputs = {
            "input_ids": encoded["input_ids"].to(self.device),
            "attention_mask": encoded["attention_mask"].to(self.device),
        }
        response, _ = generate_response(
            self.model,
            self.tokenizer,
            inputs,
            max_new_tokens=self.max_new_tokens,
            action_constraint=self.constraint,
        )
        skill = parse_skill(response)
        self.last_response = response
        self.last_valid = skill is not None
        return PlannerDecision(
            skill or Skill.GO_TO_GOAL,
            "minimind_counterfactual_skill_planner"
            if skill is not None
            else "invalid_minimind_skill_fallback",
        )
