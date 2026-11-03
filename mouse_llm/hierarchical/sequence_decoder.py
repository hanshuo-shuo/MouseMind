"""Score complete canonical skills under the unchanged language model.

This is an opt-in inference comparison to tokenwise greedy trie decoding.
Scores are raw log P(JSON + EOS | actual encoded public prompt), summed over
the response tokens. There is no length normalization or trie renormalization.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F

from mouse_llm.data.planner_schema import skill_target
from mouse_llm.hierarchical.policy import Skill


@torch.inference_mode()
def score_canonical_skill_sequences(
    model: Any,
    tokenizer: Any,
    prompt_inputs: dict[str, torch.Tensor],
    *,
    prompt_text: str,
    max_new_tokens: int = 24,
) -> dict[str, float]:
    """Batch-score the three exact JSON+EOS responses for one public prompt.

    The encoded prompt is used verbatim, including the caller's existing
    truncation. We check that its IDs match that truncation and that appending
    a canonical response does not retokenize the text across the prompt/answer
    boundary. Padding after each response is excluded from the scoring mask.
    Non-finite response logits or scores fail explicitly, never via a skill
    fallback. Prompt token limits and response token budgets stay separate,
    matching the existing autoregressive inference contract.
    """
    if isinstance(max_new_tokens, bool) or not isinstance(max_new_tokens, int) or max_new_tokens < 1:
        raise ValueError("max_new_tokens must be a positive integer")
    if not isinstance(prompt_text, str):
        raise ValueError("Sequence scoring requires the actual serialized prompt text")
    ids = prompt_inputs.get("input_ids")
    attention = prompt_inputs.get("attention_mask")
    if (not isinstance(ids, torch.Tensor) or ids.ndim != 2 or ids.shape[0] != 1 or ids.shape[1] < 1
            or ids.dtype not in (torch.int32, torch.int64)):
        raise ValueError("Sequence scoring requires input_ids with shape [1, positive_length]")
    if (not isinstance(attention, torch.Tensor) or attention.shape != ids.shape
            or not bool((attention == 1).all().item())):
        raise ValueError("Sequence scoring requires a single unpadded public prompt")
    eos = tokenizer.eos_token_id
    eos_text = getattr(tokenizer, "eos_token", None)
    if isinstance(eos, bool) or not isinstance(eos, int) or eos < 0 or not isinstance(eos_text, str) or not eos_text:
        raise ValueError("Canonical skill scoring requires an explicit tokenizer EOS token")
    full_prefix = list(tokenizer.encode(prompt_text, add_special_tokens=True))
    prefix = ids[0].detach().cpu().tolist()
    if len(prefix) > len(full_prefix):
        raise ValueError("Encoded inference prompt is longer than its serialized token prefix")
    side = getattr(tokenizer, "truncation_side", "right")
    if side not in ("right", "left"):
        raise ValueError("Unsupported tokenizer truncation side")
    expected_prefix = full_prefix[:len(prefix)] if side == "right" else full_prefix[-len(prefix):]
    if prefix != expected_prefix:
        raise ValueError("Encoded inference prompt does not match the tokenizer's prompt prefix/truncation contract")
    skills = tuple(Skill)
    responses = []
    for skill in skills:
        response_text = skill_target(skill)
        response = [*tokenizer.encode(response_text, add_special_tokens=False), eos]
        if len(response) < 2 or len(response) > max_new_tokens:
            raise ValueError(f"Canonical {skill.value} response including EOS exceeds the existing response budget")
        # Verify before truncation so truncating the public prompt cannot hide
        # a training/inference boundary-tokenization mismatch.
        combined = list(tokenizer.encode(prompt_text + response_text + eos_text, add_special_tokens=True))
        if combined != full_prefix + response:
            raise ValueError(f"Tokenizer prompt/response prefix contract differs for {skill.value}")
        responses.append(response)
    if len({tuple(response) for response in responses}) != len(skills):
        raise ValueError("Canonical skills must have distinct token sequences")
    pad = tokenizer.pad_token_id
    pad = eos if pad is None else int(pad)
    length = len(prefix) + max(map(len, responses))
    batch_ids = torch.tensor([
        prefix + response + [pad] * (length - len(prefix) - len(response))
        for response in responses
    ], dtype=torch.long, device=ids.device)
    batch_mask = torch.tensor([
        [1] * (len(prefix) + len(response)) + [0] * (length - len(prefix) - len(response))
        for response in responses
    ], dtype=torch.long, device=ids.device)
    logits = model(batch_ids, attention_mask=batch_mask).logits
    if not isinstance(logits, torch.Tensor) or logits.ndim != 3 or tuple(logits.shape[:2]) != tuple(batch_ids.shape):
        raise ValueError("Sequence scoring requires full-length next-token logits for all three candidates")
    scores: dict[str, float] = {}
    for index, (skill, response) in enumerate(zip(skills, responses)):
        # The first response token is predicted at the final prompt position;
        # the EOS token is included, while later padding positions are not.
        response_logits = logits[index, len(prefix) - 1:len(prefix) - 1 + len(response)].float()
        if not bool(torch.isfinite(response_logits).all().item()):
            raise FloatingPointError(f"Non-finite response logits while scoring {skill.value}")
        targets = torch.tensor(response, dtype=torch.long, device=logits.device)
        if bool(((targets < 0) | (targets >= logits.shape[-1])).any().item()):
            raise ValueError(f"Canonical {skill.value} contains a token outside the model vocabulary")
        token_logps = F.log_softmax(response_logits, dim=-1).gather(-1, targets.unsqueeze(-1)).squeeze(-1)
        total = token_logps.sum()
        if not bool(torch.isfinite(token_logps).all().item()) or not bool(torch.isfinite(total).item()):
            raise FloatingPointError(f"Non-finite complete-sequence log probability for {skill.value}")
        scores[skill.value] = float(total.item())
    return scores
