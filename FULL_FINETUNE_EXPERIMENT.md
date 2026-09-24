# Full-parameter skill-planner comparison

This experiment keeps the MouseMind hierarchy, the three-skill JSON policy, the specialist, the seed-clean verified data, and the frozen P2 evaluation contract. It changes only the MiniMind skill-planner adaptation from LoRA to full-parameter updates. The pretrained `full_sft_768.pth` remains the common starting model.

## Fixed training path

| Stage | Initialization | Data | Main configuration |
| --- | --- | --- | --- |
| Full SFT | pretrained MiniMind | seed-clean chosen skill responses from the verified DPO train split | 3 epochs; effective batch 16; AdamW LR `1e-5` |
| Full Verified DPO | full SFT checkpoint; frozen full SFT reference | same seed-clean verified preference-pair train split | β `0.1`; 1 epoch; effective batch 16; AdamW LR `1e-6` |
| Full Verified GRPO | full SFT checkpoint; frozen full SFT reference | seed-isolated, exact-state verified training anchors | 1 epoch; group size 6; rollout horizon 8; clip `0.2`; KL `0.01`; AdamW LR `2e-7` |

The full SFT stage uses the same effective batch size, data, epochs, and 768×8 MiniMind configuration as the seed-clean LoRA SFT experiment. Lower learning rates account for updating every model parameter. This is one predeclared full-parameter configuration, not a parameter sweep. DPO and GRPO are separate descendants of the full SFT checkpoint.

Each training job ran offline skill evaluation and the 40-seed development closed-loop protocol. A dependent job evaluated all three final checkpoints on the unchanged 100-seed final-ID protocol. Final seeds never entered training or model selection. Since prior variants were evaluated on that pool, final comparisons remain exploratory.

## Results

All four Slurm jobs (`23747`–`23750`) completed successfully. The full SFT stage trained on 1,072 chosen responses for 201 optimizer updates; full DPO trained on 1,072 preference pairs for 67 updates. Full GRPO used 276 training anchors, skipped six failed fresh replay checks, and applied 504 updates across 810 sampled groups. The remaining training, episode, and checkpoint files stay private on the cluster.

| Seed-clean planner | Final-ID task success ↑ | Clean success ↑ | Capture rate ↓ | Captures / episode ↓ | Development task success ↑ | Development captures / episode ↓ |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| SFT LoRA | 92% | 8% | 92% | 8.94 | 95% | 7.10 |
| Full SFT | 95% | 18% | 82% | 5.41 | 95% | 5.85 |
| Verified DPO LoRA, β=0.1 | **100%** | **22%** | **78%** | **4.94** | 95% | 5.80 |
| Full Verified DPO, β=0.1 | 80% | 1% | 99% | 13.09 | 82.5% | 10.98 |
| Verified GRPO LoRA | 80% | 1% | 99% | 13.15 | 75% | 11.75 |
| Full Verified GRPO | 85% | 1% | 99% | 11.62 | 87.5% | 10.48 |

All final-ID rows use the same 100 seeds, contract SHA-256 `9572230201a1c2a8afb764012237df8c43d155c4941bd2f7282282889af2ed75`, survival-first preference, and planner horizon 8. Development uses a separate 40-seed pool. The LoRA rows are existing runs under the same evaluation contract; only the full-parameter rows were submitted in this round.

Paired final-ID bootstrap differences on the same seeds show that full SFT versus seed-clean LoRA SFT reduces captures per episode by 3.53 (95% CI [−5.75, −1.44]) and raises clean success by 10 percentage points ([+2, +19] pp). Its task-success difference is +3 pp (CI [−4, +10] pp). Relative to full SFT, full DPO lowers task success by 15 pp (CI [−24, −6]), lowers clean success by 17 pp ([−25, −9]), raises capture rate by 17 pp ([+10, +25]), and raises captures per episode by 7.68 ([+5.12, +10.30]). Full GRPO versus full SFT lowers task success by 10 pp (CI [−18, −2]) and raises captures per episode by 6.21 ([+3.78, +8.73]). These intervals support keeping the full SFT checkpoint ahead of its full DPO and GRPO descendants in this configuration.

The observed closed-loop failure mode is EVADE-heavy behavior. On final-ID episodes, `evade_predator` accounts for 86.7% of executed skill steps under full SFT, 99.8% under full DPO, and 97.4% under full GRPO; it accounts for 68.0% under the successful LoRA DPO. Full GRPO sampled EVADE 81.3% of the time during training and skipped 37.8% of groups because reward variance was effectively zero. These observations are consistent with insufficient goal-directed behavior and longer exposure to captures. They do not establish whether parameter count, learning rate, verified utility, or training-state distribution is the sole cause. Full DPO's offline seen/unseen skill accuracy stayed at 71.8% / 69.2%, matching full SFT despite the closed-loop degradation.

## Selection

**Main language-planner result: seed-clean Verified DPO LoRA, β=0.1.** It has the strongest measured task/safety combination, and its paired improvements over its SFT LoRA initialization exclude zero for all four primary metrics. **Lower-resource baseline: seed-clean SFT LoRA.** Only 393,216 of 63,912,192 model parameters are trainable (0.62%), and the private adapter checkpoint is about 0.8 MB rather than about 275 MB for the full checkpoint. This saves trainable optimizer state and checkpoint storage, while its safety metrics remain weaker than LoRA DPO. The present single full-parameter configuration does not prove that every full-finetuning setting would fail.

## Cluster submission

The Slurm script is [`mouse_llm/northwestern/p2/full_finetune_verified.sbatch`](mouse_llm/northwestern/p2/full_finetune_verified.sbatch). It writes private checkpoints and detailed metrics under `/shares/bcs516/sh/minimind_mouse_p2/full_verified_v1/`, leaving existing LoRA runs untouched. The first submission used jobs SFT `23747`, DPO `23748`, GRPO `23749`, and final comparison `23750` with `afterok` dependencies.

```bash
script=mouse_llm/northwestern/p2/full_finetune_verified.sbatch
sft=$(sbatch --parsable --export=ALL,FULL_METHOD=sft "$script")
dpo=$(sbatch --parsable --dependency=afterok:"$sft" --export=ALL,FULL_METHOD=dpo "$script")
grpo=$(sbatch --parsable --dependency=afterok:"$sft" --export=ALL,FULL_METHOD=grpo "$script")
sbatch --dependency=afterok:"$dpo":"$grpo" --export=ALL,FULL_METHOD=final "$script"
```

Do not rerun this exact submission into `full_verified_v1` without first selecting a new output root. The cluster checkout may contain unrelated untracked experiment files; only the full-finetuning and evaluation files need syncing.
