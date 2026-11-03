"""Train continuous-context MiniMind skill LoRAs with SFT or verified DPO.

The LLM retains its three-skill output; the specialist executes continuous XY.
Model/data/executor hashes bind every new adapter to its supervision contract.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from .context import CONTEXT_SCHEMA
from .contracts import ACTION_SCHEMA
from .posttraining import DATASET_SCHEMA, private_directory, private_json, sha256, seed_split


def validate_dataset(directory: Path, specialist_checkpoint: Path) -> dict:
    from mouse_llm.data.planner_schema import parse_skill
    manifest = json.loads((directory / "manifest.json").read_text())
    if (manifest.get("schema_version") != DATASET_SCHEMA
            or manifest.get("action_schema") != ACTION_SCHEMA
            or manifest.get("context_schema") != CONTEXT_SCHEMA):
        raise ValueError("Expected a freshly generated continuous planner dataset")
    if manifest.get("specialist_sha256") != sha256(specialist_checkpoint):
        raise ValueError("Skill executor changed: regenerate counterfactual labels")
    seeds = {}
    for split in ("train", "validation"):
        path = directory / f"planner_{split}.jsonl"
        if sha256(path) != manifest["splits"][split]["sha256"]:
            raise ValueError(f"{split} data no longer matches its manifest")
        rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        if not rows or len(rows) != manifest["splits"][split]["rows"]:
            raise ValueError(f"Missing or incorrect {split} sample count")
        seeds[split] = set()
        for row in rows:
            if (row.get("schema_version") != DATASET_SCHEMA or row.get("split") != split
                    or row.get("action_schema") != ACTION_SCHEMA or row.get("context_schema") != CONTEXT_SCHEMA):
                raise ValueError("Wrong sample schema or split")
            seed = int(row["seed"])
            if seed_split(seed) != split or not 50000 <= seed < 50200:
                raise ValueError("Non-training collection seed entered planner training")
            seeds[split].add(seed)
            payload = json.loads(row["prompt"][1]["content"])
            if (payload["context"].get("schema_version") != CONTEXT_SCHEMA
                    or payload["context"].get("action_schema") != ACTION_SCHEMA):
                raise ValueError("Legacy context cannot train a new continuous planner")
            chosen, rejected = parse_skill(row["chosen"]), parse_skill(row["rejected"])
            if chosen is None or rejected is None or chosen == rejected:
                raise ValueError("Invalid preference responses")
            if chosen.value != row.get("target_skill"):
                raise ValueError("SFT target and DPO chosen response disagree")
            utilities = row.get("utilities", {})
            if (set(utilities) != {"go_to_goal", "evade_predator", "hold_position"}
                    or not all(math.isfinite(float(value)) for value in utilities.values())
                    or utilities[chosen.value] != max(utilities.values())
                    or utilities[chosen.value] <= utilities[rejected.value]):
                raise ValueError("Chosen/rejected utility ordering is invalid")
            expected_instruction_split = "train" if split == "train" else "validation"
            if row.get("instruction_split") != expected_instruction_split:
                raise ValueError("Instruction split mismatch")
    if seeds["train"] & seeds["validation"]:
        raise ValueError("Planner seed split leakage")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("sft", "dpo"))
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--specialist-checkpoint", type=Path, required=True)
    parser.add_argument("--base-weight", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, default=Path(__file__).resolve().parents[2] / "model")
    parser.add_argument("--sft-lora-weight", type=Path)
    parser.add_argument("--sft-manifest", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--gradient-accumulation", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--beta", type=float, default=0.1)
    parser.add_argument("--max-seq-len", type=int, default=1536)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    if min(args.epochs, args.batch_size, args.gradient_accumulation) < 1 or args.learning_rate <= 0 or args.beta <= 0 or args.max_seq_len < 64:
        parser.error("Invalid training parameters")
    dataset_manifest = validate_dataset(args.data_dir, args.specialist_checkpoint)
    if args.stage == "dpo":
        if args.sft_lora_weight is None or args.sft_manifest is None:
            parser.error("DPO requires its SFT initialization and --sft-manifest")
        source = json.loads(args.sft_manifest.read_text())
        if source.get("schema_version") != "mousemind_continuous_planner_checkpoint_v1" or source.get("stage") != "sft":
            raise ValueError("DPO initialization must be an identified continuous SFT checkpoint")
        for key, expected in (("skill_lora_sha256", sha256(args.sft_lora_weight)),
                              ("base_weight_sha256", sha256(args.base_weight)),
                              ("specialist_sha256", sha256(args.specialist_checkpoint)),
                              ("context_schema", CONTEXT_SCHEMA), ("action_schema", ACTION_SCHEMA)):
            if source.get(key) != expected:
                raise ValueError(f"SFT initialization mismatch: {key}")
    if args.validate_only:
        print(json.dumps({"validated": True, "stage": args.stage, "splits": dataset_manifest["splits"]}))
        return
    private_directory(args.output_dir)
    if any(args.output_dir.iterdir()):
        raise ValueError("Use an empty output directory to preserve checkpoint lineage")
    import torch
    import torch.nn.functional as F
    from torch.utils.data import DataLoader
    from transformers import AutoTokenizer
    from model.model_lora import apply_lora, save_lora
    from model.model_minimind import MiniMindConfig, MiniMindForCausalLM
    from mouse_llm.training.train_verified_dpo import PairDataset, completion_logps, dpo_loss, make_model

    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    train = PairDataset(args.data_dir / "planner_train.jsonl", tokenizer, args.max_seq_len)
    validation = PairDataset(args.data_dir / "planner_validation.jsonl", tokenizer, args.max_seq_len)
    # Preflight all labels so truncation cannot silently remove the response.
    for dataset in (train, validation):
        for index in range(len(dataset)):
            row = dataset.rows[index]
            for response in (row["chosen"], row["rejected"]):
                text = tokenizer.apply_chat_template([*row["prompt"], {"role": "assistant", "content": response}], tokenize=False, add_generation_prompt=False)
                if len(tokenizer(text).input_ids) > args.max_seq_len:
                    raise ValueError("A complete planner response exceeds max_seq_len; refuse partial-target training")
            dataset[index]
    loader = DataLoader(train, batch_size=args.batch_size, shuffle=True,
                        generator=torch.Generator().manual_seed(args.seed))
    validation_loader = DataLoader(validation, batch_size=args.batch_size, shuffle=False)
    config = MiniMindConfig(hidden_size=768, num_hidden_layers=8)
    reference = None
    if args.stage == "dpo":
        policy = make_model(config, args.base_weight, args.sft_lora_weight, device, trainable=True)
        reference = make_model(config, args.base_weight, args.sft_lora_weight, device, trainable=False).eval()
    else:
        policy = MiniMindForCausalLM(config)
        policy.load_state_dict(torch.load(args.base_weight, map_location="cpu", weights_only=True), strict=True)
        apply_lora(policy)
        for name, parameter in policy.named_parameters():
            parameter.requires_grad_(".lora." in name)
        policy.to(device)
    parameters = [parameter for parameter in policy.parameters() if parameter.requires_grad]
    if not parameters:
        raise ValueError("No trainable LoRA parameters")
    optimizer = torch.optim.AdamW(parameters, lr=args.learning_rate)

    def objective(batch):
        c_ids, c_labels, c_mask = (tensor.to(device) for tensor in batch["chosen"])
        if args.stage == "sft":
            logits = policy(c_ids).logits.float()
            token_losses = F.cross_entropy(logits.flatten(0, 1), c_labels.flatten(), reduction="none").view_as(c_labels)
            return (token_losses * c_mask).sum() / c_mask.sum()
        r_ids, r_labels, r_mask = (tensor.to(device) for tensor in batch["rejected"])
        with torch.no_grad():
            ref_c = completion_logps(reference, c_ids, c_labels, c_mask)
            ref_r = completion_logps(reference, r_ids, r_labels, r_mask)
        pi_c = completion_logps(policy, c_ids, c_labels, c_mask)
        pi_r = completion_logps(policy, r_ids, r_labels, r_mask)
        return dpo_loss(pi_c, pi_r, ref_c, ref_r, args.beta)

    best = float("inf")
    history, updates = [], 0
    weight = args.output_dir / "skill_lora.pth"
    for epoch in range(args.epochs):
        policy.train()
        optimizer.zero_grad(set_to_none=True)
        losses = []
        for index, batch in enumerate(loader):
            loss = objective(batch)
            group_start = (index // args.gradient_accumulation) * args.gradient_accumulation
            group_size = min(args.gradient_accumulation, len(loader) - group_start)
            (loss / group_size).backward()
            if (index + 1) % args.gradient_accumulation == 0 or index + 1 == len(loader):
                torch.nn.utils.clip_grad_norm_(parameters, 1.0)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                updates += 1
            losses.append(float(loss.detach()))
        policy.eval()
        with torch.inference_mode():
            total, count = 0.0, 0
            for batch in validation_loader:
                size = len(batch["chosen"][0])
                total += float(objective(batch)) * size
                count += size
            val_loss = total / count
        row = {"epoch": epoch + 1, "train_loss": sum(losses) / len(losses), "validation_loss": val_loss}
        history.append(row)
        print(json.dumps(row), flush=True)
        if val_loss < best:
            best = val_loss
            save_lora(policy, str(weight))
            weight.chmod(0o600)
    manifest = {
        "schema_version": "mousemind_continuous_planner_checkpoint_v1", "stage": args.stage,
        "action_schema": ACTION_SCHEMA, "context_schema": CONTEXT_SCHEMA,
        "skill_lora_sha256": sha256(weight), "base_weight_sha256": sha256(args.base_weight),
        "specialist_sha256": sha256(args.specialist_checkpoint),
        "data_manifest_sha256": sha256(args.data_dir / "manifest.json"),
        "sft_reference_sha256": None if reference is None else sha256(args.sft_lora_weight),
        "seed": args.seed, "optimizer_updates": updates, "max_seq_len": args.max_seq_len,
        "execution_contract": {"planner_horizon": dataset_manifest["candidate_steps"],
                               "memory_steps": dataset_manifest["memory_steps"],
                               "evade_distance": dataset_manifest["evade_distance"]},
        "training_environment": dataset_manifest["environment_config"],
        "checkpoint_selection": "validation_loss; closed-loop development evaluation still required",
        "history": history, "research_evidence": False,
    }
    private_json(args.output_dir / "planner_manifest.json", manifest)
    print(json.dumps(manifest, sort_keys=True))


if __name__ == "__main__":
    main()
