# Verified preference learning and RLVR

## Research question

Can environment-verifiable skill rewards improve capture avoidance beyond a supervised MiniMind skill planner while preserving task success? We compare the same seed-clean SFT initialization with Verified DPO and a modular Verified GRPO policy update. The policy still emits one of three constrained skill JSON responses; the existing specialist executes it.

## What was held fixed

- The Cellworld BotEvade environment, specialist, skill vocabulary, planner horizon 8, and survival-first closed-loop evaluation protocol.
- The counterfactual utility used for SFT labels and DPO preferences. GRPO calls that same utility on fresh exact-state branch outcomes; it does not define a second reward.
- Seed isolation for training: the 320 collection anchors are grouped by seed before constructing the 69-seed DPO/GRPO training pool. GRPO used 276 matching replay anchors. Final ID seeds never enter training or replay-anchor selection.
- The fixed 100-seed final ID contract for the historical SFT, seed-clean SFT, DPO, and initial GRPO comparison. Development model selection uses the separate 40-seed pool.

Private anchors, episode rows, model weights, and traces remain outside Git. Public files contain code, run definitions, and aggregate metrics only.

## The result to keep: one-epoch Verified DPO

| Policy | Task success | Clean success | Capture rate | Captures / episode |
| --- | ---: | ---: | ---: | ---: |
| Published skill-planner LoRA | 97% | 12% | 88% | 7.37 |
| Seed-clean SFT LoRA | 92% | 8% | 92% | 8.94 |
| **Verified DPO, β=0.1** | **100%** | **22%** | **78%** | **4.94** |

All rows use the same 100 final ID seeds and evaluation settings. Relative to its controlled seed-clean SFT initialization, DPO changes task success by +8 percentage points, clean success by +14 points, capture rate by −14 points, and captures per episode by −4.00. Relative to the previously published LoRA, its per-seed captures-per-episode difference is −2.43 (95% paired bootstrap CI [−4.67, −0.38]). The historical LoRA used an earlier anchor-ID training split, so that last comparison matches evaluation conditions but does not isolate the effect of the training algorithm. See [Verified DPO results](VERIFIED_DPO_RESULTS.md) for the other paired intervals and checkpoint provenance.

## What GRPO tested

Each anchor supplies the existing planner prompt. MiniMind samples six constrained JSON skill responses; only distinct skills are branched from the exact replay state. The outcome maps back to all six samples, and the group-standardized advantage drives a clipped policy objective with a frozen SFT reference. A zero-variance group is skipped. `hold_position` remains a possible sampled skill, while the RL reward uses survival-first, balanced, and goal-first preferences; the HOLD preference remains SFT-trained. Every used branch checks the replayed anchor state at the original tolerance.

The first GRPO run used one epoch, temperature 1.0, horizon 8, clip 0.2, and KL coefficient 0.01. Its final ID result was 80% task success, 1% clean success, 99% capture rate, and 13.15 captures per episode. Its development result showed the same direction: 75% task success and 11.75 captures per episode. It was not promoted.

## The longer training round

We predeclared one longer run of each method from the same seed-clean SFT LoRA: DPO β=0.1 for three epochs at learning rate `5e-6`, and GRPO for two epochs at learning rate `1e-6`, with two policy updates per sampled group and stronger adaptive reference KL. Both jobs completed. The complete 40-seed development pool gave:

| Policy | Task success | Clean success | Capture rate | Captures / episode |
| --- | ---: | ---: | ---: | ---: |
| Seed-clean SFT | 95% | 17.5% | 82.5% | 7.10 |
| One-epoch DPO β=0.1 | 95% | 22.5% | 77.5% | 5.80 |
| First GRPO | 75% | 2.5% | 97.5% | 11.75 |
| Three-epoch DPO | 77.5% | 2.5% | 97.5% | 10.73 |
| Two-epoch GRPO | 87.5% | 5.0% | 95.0% | 9.88 |

Both longer candidates failed the predeclared requirement to preserve task success and improve capture outcomes on development. Neither was run on final ID seeds or added to the promoted result table.

## Why the longer candidates failed

The directly observed failure is an EVADE-heavy closed-loop policy. On development episodes, EVADE accounts for 69.9% of skill selections under one-epoch DPO, versus 97.7% under the longer DPO run and 96.0% under the longer GRPO run. That shift accompanies more captures, lower clean success, and less task completion.

The first GRPO run shows a specific feedback loop in its training trace: although EVADE is the highest-utility branch for roughly 55–67% of training anchors across the three RL preferences, its sampled probability rises to about 96.6% in the final 110 groups. Ninety-three of those groups contain six EVADE samples, and 72 have no reward contrast to update from. The reference KL rises to about 0.50 late in training. Repeated sampling of the same action therefore removes the group signal needed to correct it. With one policy update per group, the clipped ratio starts at one, so clipping cannot constrain that first update; the weak reference penalty did not prevent drift.

The longer GRPO configuration reduced average training KL to 0.088 and the skipped-group fraction to 24.0%, but its development rollout still chose EVADE 96.0% of the time. Its measured clip fraction was zero; the smaller individual updates never crossed the clip boundary. This points to an anchor-to-closed-loop state-distribution gap as well as a reward/optimization mismatch. The longer DPO configuration changed both epoch count and learning rate, so its degradation cannot be attributed to epoch count alone. These mechanisms are supported by the traces but are not uniquely proven causal explanations.

Exact-state checks stayed fail-closed: six anchor occurrences in the first GRPO run and twelve across the two-epoch run failed fresh replay verification and were excluded from updates. They are recorded in the private training manifests. They do not explain the strong EVADE bias observed in accepted training groups and development episodes.

## Evidence limits and reproduction

The final ID seeds were never training inputs, but multiple DPO β values and the first GRPO policy were evaluated on that fixed pool. Treat those comparisons as exploratory, not a single untouched confirmatory test. The later extended candidates were filtered on development and did not reach final ID evaluation. No claim is made about OOD improvement or general safety.

The modular RLVR implementation is in [`mouse_llm/verified_rl`](mouse_llm/verified_rl), with its training entry point at [`train_verified_grpo.py`](mouse_llm/training/train_verified_grpo.py). The run definitions are [`verified_grpo.sbatch`](mouse_llm/northwestern/p2/verified_grpo.sbatch), [`verified_alignment_development.sbatch`](mouse_llm/northwestern/p2/verified_alignment_development.sbatch), and [`verified_alignment_extended.sbatch`](mouse_llm/northwestern/p2/verified_alignment_extended.sbatch). Seed-matched comparisons use [`compare_verified_alignment.py`](mouse_llm/evaluation/compare_verified_alignment.py). The recommended checkpoint is the one-epoch DPO β=0.1 LoRA stored in the private P2 run root; checkpoint weights and private replay data are not published here.
