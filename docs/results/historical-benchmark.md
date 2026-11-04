# Historical 100-seed benchmark

All rows use the same 100 paired BotEvade final-ID seeds, excluded from training. Multiple variants were evaluated on this fixed pool, so the comparison is exploratory rather than a one-shot blind test. Task success means reaching the goal; clean success additionally requires no capture; capture rate is the share of episodes with at least one capture. The numeric planner is a **non-language upper reference**. The published SFT LoRA is the main SFT result; Verified DPO was initialized from a separately retrained seed-isolated SFT LoRA, detailed in the [DPO report](verified-dpo.md).

| Policy | Planner training | Task success ↑ | Clean success ↑ | Capture rate ↓ | Captures / episode ↓ |
| --- | --- | ---: | ---: | ---: | ---: |
| Direct MiniMind | Flat action LoRA | 26% | 1% | 99% | 98.60 |
| Direct MLP BC | Flat action imitation | 20% | 5% | 94% | 87.90 |
| P1 rule hierarchy | Rule baseline | 79% | 14% | 86% | 11.20 |
| MiniMind without history | Hierarchical LoRA ablation | 80% | 1% | 99% | 13.15 |
| MiniMind without instruction | Hierarchical LoRA ablation | 80% | 1% | 99% | 13.15 |
| **Published MiniMind hierarchy** | **SFT LoRA · lower resource** | **97%** | **12%** | **88%** | **7.37** |
| Full-parameter MiniMind hierarchy | Full SFT, seed-isolated data | 95% | 18% | 82% | 5.41 |
| **Verified DPO skill planner, β=0.1** | **SFT LoRA → DPO LoRA · best language planner** | **100%** | **22%** | **78%** | **4.94** |
| Verified DPO skill planner, β=0.1 | Full SFT → full DPO | 80% | 1% | 99% | 13.09 |
| Verified GRPO skill planner | SFT LoRA → GRPO LoRA | 80% | 1% | 99% | 13.15 |
| Verified GRPO skill planner | Full SFT → full GRPO | 85% | 1% | 99% | 11.62 |
| Numeric planner | Non-language upper reference | 100% | 38% | 62% | 2.66 |

**Why use LoRA DPO for the main result?** It keeps the same MiniMind → skill → specialist hierarchy and continues training a seed-isolated SFT LoRA with exact-state verified preference pairs. Against its own SFT initialization (92% task success, 8% clean success, 92% capture rate, 8.94 captures per episode), task success improves by 8 percentage points, clean success by 14 points, capture rate falls by 14 points, and captures per episode fall by 4.00. The paired 95% confidence intervals for all four changes exclude zero. The published SFT LoRA above is a matched-evaluation historical reference, not DPO's initialization. LoRA DPO reaches the best tested combination of task completion and capture reduction among the language planners; the numeric reference remains stronger on clean success and captures.

**Why keep published LoRA SFT as the resource-conscious option?** It needs only the SFT stage and trains 393,216 skill-planner parameters (0.62% of MiniMind's 63,912,192). Its adapter checkpoint is about 0.8 MB versus about 275 MB for a full checkpoint; fewer trainable parameters also reduce optimizer-state storage. It achieves 97% task success, 12% clean success, and 7.37 captures per episode, while LoRA DPO provides the stronger measured safety result after an additional preference-training stage. The separately retrained seed-isolated SFT baseline remains in the detailed reports for a valid controlled DPO comparison.

Full-parameter DPO and GRPO both drift toward `evade_predator`: it occupies 99.8% and 97.4% of final-ID executed skill steps, respectively, versus 86.7% for full SFT and 68.0% for LoRA DPO. That observed shift accompanies lower task success and more captures; it is a plausible explanation, not proof of a unique cause. See [Verified DPO results](verified-dpo.md) for paired LoRA estimates, [full-parameter experiment](full-finetuning.md) for the new run, [P2 results](planner-benchmarks.md) for original ablations and OOD, and [Verified RLVR summary](verified-rlvr.md) for the earlier GRPO and longer-training runs.
