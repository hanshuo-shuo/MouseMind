# Implementation and ablations

## Skill execution

The goal executor maps observations to a continuous `(x, y)` request using a BC specialist. EVADE and HOLD use the existing task-specific controllers. The shared environment layer projects or rejects infeasible destinations. For compatibility with the planner's existing action-history format, the requested GO destination is mapped to its nearest catalog ID for history only; execution retains the continuous XY request.

## Executor ablation with a frozen DPO planner

To evaluate the execution layer, we kept the DPO weights, three-skill JSON interface, semantic context, planning cadence, and EVADE/HOLD controllers fixed, and replaced only the catalog-based GO MLP with continuous BC and its history adapter. The high-level planner was not retrained. All three systems below ran on the same 200 paired final seeds, sequentially on one H100 with a common simulator, navigation/reset contract, and software environment.

| System | Task success ↑ | Capture-free success ↑ | Captures / episode ↓ |
| --- | ---: | ---: | ---: |
| DPO planner + original discrete executor | 92.5% (185/200) | 18.5% (37/200) | 7.535 |
| Continuous BC alone | 96.0% (192/200) | 11.0% (22/200) | 3.780 |
| **DPO planner + continuous BC GO executor** | **97.5% (195/200)** | **41.5% (83/200)** | **5.275** |

The executor change produced paired 95% bootstrap intervals of **+23 pp [14, 31.5]** for capture-free success, **+5 pp [1, 9.5]** for task success, and **−2.26 [−4.20, −0.285]** for captures per episode relative to the original executor. The candidate passed all three predeclared criteria: improved capture-free completion, task noninferiority within 2 percentage points, and no increase in captures. These deltas describe execution refinement under fixed DPO weights, not a new DPO training gain.

Compared with direct BC, the hierarchy achieves more capture-free completions but has a higher mean capture count across all episodes: **5.275 versus 3.780**. This comparison adds the planner, skill switching, and EVADE/HOLD controllers together, so it does not isolate the effect of language reasoning. The intervals cover environment-seed variation for fixed checkpoints, not training-run variability. See the [development comparison](../results/continuous-development.md) for candidate selection and the [final report](../results/final-system.md) for the complete evidence.

## Instruction and history ablations

The historical skill-planner study measured **25% offline skill accuracy** without task LoRA and **62.2% on unseen paraphrases** after task adaptation. Removing instruction or temporal history reduced the historical hierarchy's closed-loop task success from **97% to 80%**. These ablations support the role of task adaptation and temporal context under that study's contract; they have not been rerun with the latest continuous executor.
