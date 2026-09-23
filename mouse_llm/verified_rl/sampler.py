"""Sample only canonical skill JSON responses and score their full token sequences."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F

from mouse_llm.data.planner_schema import skill_target
from mouse_llm.evaluation.evaluate_policy import _chat_text
from mouse_llm.hierarchical.minimind_planner import SkillTokenConstraint
from mouse_llm.hierarchical.policy import Skill


SKILLS = tuple(Skill)


@dataclass(frozen=True)
class SkillSample:
    skill: Skill
    sequence_log_probability: float
    constrained_log_probability: float


def _prompt_ids(tokenizer: Any, prompt: str | list[dict[str, str]]) -> list[int]:
    text = _chat_text(tokenizer, prompt, generate=True) if isinstance(prompt, list) else prompt
    return list(tokenizer.encode(text, add_special_tokens=True))


def score_skill_distributions(
    model: Any,
    tokenizer: Any,
    prompt: str | list[dict[str, str]],
    *,
    max_seq_len: int = 1024,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Raw full-response logps and trie-constrained skill logps."""
    if tokenizer.eos_token_id is None:
        raise ValueError("Tokenizer must have an EOS token")
    prefix = _prompt_ids(tokenizer, prompt)
    responses = [
        [*tokenizer.encode(skill_target(skill), add_special_tokens=False),
         int(tokenizer.eos_token_id)]
        for skill in SKILLS
    ]
    if not prefix or len(prefix) + max(map(len, responses)) > max_seq_len:
        raise ValueError("Planner prompt or skill response exceeds max_seq_len")
    device = next(model.parameters()).device
    pad = tokenizer.pad_token_id
    pad = int(pad) if pad is not None else int(tokenizer.eos_token_id)
    length = len(prefix) + max(map(len, responses))
    sequences = [prefix + suffix + [pad] * (length - len(prefix) - len(suffix)) for suffix in responses]
    ids = torch.tensor(sequences, dtype=torch.long, device=device)
    logits = model(ids).logits.float()
    scores = []
    constrained_scores = []
    trie = SkillTokenConstraint(tokenizer).root
    for index, suffix in enumerate(responses):
        start = len(prefix) - 1
        response_logits = logits[index, start:start + len(suffix)]
        response_logps = F.log_softmax(response_logits, dim=-1)
        targets = torch.tensor(suffix, dtype=torch.long, device=device)
        scores.append(response_logps.gather(-1, targets.unsqueeze(-1)).sum())
        node = trie
        constrained = logits.new_zeros(())
        for offset, token in enumerate(suffix):
            allowed = sorted(node)
            if token not in node:
                raise ValueError("Canonical skill response diverges from token trie")
            local = F.log_softmax(logits[index, start + offset, allowed], dim=-1)
            constrained = constrained + local[allowed.index(token)]
            node = node[token]
        constrained_scores.append(constrained)
    return torch.stack(scores), torch.stack(constrained_scores)


def score_skills(
    model: Any,
    tokenizer: Any,
    prompt: str | list[dict[str, str]],
    *,
    max_seq_len: int = 1024,
) -> torch.Tensor:
    """Unnormalized MiniMind log P(full JSON + EOS | prompt)."""
    return score_skill_distributions(model, tokenizer, prompt, max_seq_len=max_seq_len)[0]


def sample_skills(
    model: Any,
    tokenizer: Any,
    prompt: str | list[dict[str, str]],
    *,
    group_size: int = 6,
    temperature: float = 1.0,
    max_seq_len: int = 1024,
) -> tuple[list[SkillSample], dict[str, float]]:
    """Categorical sampling over the three complete, valid JSON responses."""
    if group_size < 2 or temperature <= 0:
        raise ValueError("group_size must be >=2 and temperature must be positive")
    with torch.no_grad():
        sequence_logps, branch_logps = score_skill_distributions(
            model, tokenizer, prompt, max_seq_len=max_seq_len
        )
        constrained_logps = F.log_softmax(branch_logps / temperature, dim=0)
        probabilities = constrained_logps.exp()
        indices = torch.multinomial(probabilities, group_size, replacement=True)
    samples = [
        SkillSample(
            SKILLS[int(index)],
            float(sequence_logps[index].item()),
            float(constrained_logps[index].item()),
        )
        for index in indices
    ]
    return samples, {
        skill.value: float(probabilities[index].item())
        for index, skill in enumerate(SKILLS)
    }
