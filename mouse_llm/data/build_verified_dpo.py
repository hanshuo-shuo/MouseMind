"""Build seed-disjoint best-versus-worst pairs from verified P2 branches."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

from mouse_llm.data.planner_schema import Preference, instruction_for, planner_messages, select_skill, skill_target
from mouse_llm.evaluation.contracts import DEFAULT_P2_CONTRACT, contract_seeds
from mouse_llm.hierarchical.policy import Skill


def seed_split(seed: int) -> str:
    bucket = int.from_bytes(hashlib.sha256(f"verified-dpo:{seed}".encode()).digest()[:4], "big") % 100
    return "train" if bucket < 80 else "validation" if bucket < 90 else "test"


def build_pairs(records: list[dict], *, final_seeds: set[int], label_horizon: int) -> dict[str, list[dict]]:
    outputs: dict[str, list[dict]] = defaultdict(list)
    seen_ids: set[str] = set()
    seed_groups: dict[str, set[int]] = defaultdict(set)
    for record in records:
        anchor_id = record["anchor_id"]
        seed = int(record["seed"])
        if anchor_id in seen_ids:
            raise ValueError("Duplicate anchor_id in verified counterfactuals")
        seen_ids.add(anchor_id)
        if seed in final_seeds:
            raise ValueError("Final evaluation seed leaked into DPO source")
        split = seed_split(seed)
        seed_groups[split].add(seed)
        branches = record["branches_by_horizon"][str(label_horizon)]
        if set(branches) != {skill.value for skill in Skill}:
            raise ValueError("Anchor does not have all three verified skills")
        if not record.get("replay_verification", {}).get("verified", False):
            raise ValueError("Unverified counterfactual anchor")
        for preference in Preference:
            best, utilities = select_skill(branches, preference)
            worst = min(Skill, key=lambda skill: (utilities[skill.value], skill.value))
            if utilities[best.value] <= utilities[worst.value]:
                continue
            destinations = [(split, "train" if split == "train" else "validation")]
            if split == "test":
                destinations = [("seen_test", "train"), ("unseen_test", "unseen_test")]
            for output_split, instruction_split in destinations:
                instruction = instruction_for(preference, split=instruction_split, stable_key=anchor_id)
                prompt = planner_messages(record["context"], preference=preference, instruction=instruction)
                outputs[output_split].append({
                    "anchor_id": anchor_id,
                    "seed": seed,
                    "preference": preference.value,
                    "instruction_split": instruction_split,
                    "prompt": prompt,
                    "chosen": skill_target(best),
                    "rejected": skill_target(worst),
                    "chosen_utility": float(utilities[best.value]),
                    "rejected_utility": float(utilities[worst.value]),
                    "margin": float(utilities[best.value] - utilities[worst.value]),
                    "target_skill": best.value,
                    "utilities": utilities,
                    "conversations": [*prompt, {"role": "assistant", "content": skill_target(best)}],
                })
    if seed_groups["train"] & seed_groups["validation"] or seed_groups["train"] & seed_groups["test"] or seed_groups["validation"] & seed_groups["test"]:
        raise ValueError("DPO seed partition overlap")
    return outputs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--counterfactuals-private", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--contract", type=Path, default=DEFAULT_P2_CONTRACT)
    parser.add_argument("--label-horizon", type=int, default=8)
    args = parser.parse_args()
    records = [json.loads(line) for line in args.counterfactuals_private.read_text().splitlines() if line]
    final_seeds = set(contract_seeds("final_id_test", args.contract))
    outputs = build_pairs(records, final_seeds=final_seeds, label_horizon=args.label_horizon)
    args.output_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
    summary = {"source_anchors": len(records), "label_horizon": args.label_horizon, "split_method": "sha256(verified-dpo:seed) 80/10/10", "splits": {}}
    for split in ("train", "validation", "seen_test", "unseen_test"):
        rows = outputs[split]
        path = args.output_dir / f"dpo_{split}.jsonl"
        with path.open("w") as handle:
            for row in rows:
                handle.write(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n")
        path.chmod(0o600)
        summary["splits"][split] = {"rows": len(rows), "seeds": len({row["seed"] for row in rows}), "chosen_skills": dict(Counter(row["target_skill"] for row in rows))}
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
