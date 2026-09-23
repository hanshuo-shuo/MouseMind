# Full-parameter skill-planner comparison (queued)

This experiment keeps the MouseMind hierarchy, the three-skill JSON policy, the specialist, the seed-clean verified data, and the frozen P2 evaluation contract. It changes only the MiniMind skill-planner adaptation from LoRA to full-parameter updates. The pretrained `full_sft_768.pth` remains the common starting model.

## Fixed training path

| Stage | Initialization | Data | Main configuration |
| --- | --- | --- | --- |
| Full SFT | pretrained MiniMind | seed-clean chosen skill responses from the verified DPO train split | 3 epochs; effective batch 16; AdamW LR `1e-5` |
| Full Verified DPO | full SFT checkpoint; frozen full SFT reference | same seed-clean verified preference-pair train split | β `0.1`; 1 epoch; effective batch 16; AdamW LR `1e-6` |
| Full Verified GRPO | full SFT checkpoint; frozen full SFT reference | seed-isolated, exact-state verified training anchors | 1 epoch; group size 6; rollout horizon 8; clip `0.2`; KL `0.01`; AdamW LR `2e-7` |

The full SFT stage uses the same effective batch size, data, epochs, and 768×8 MiniMind configuration as the seed-clean LoRA SFT experiment. Lower learning rates account for updating every model parameter. This is one predeclared full-parameter configuration, not a parameter sweep. DPO and GRPO are separate descendants of the full SFT checkpoint.

Each training job runs offline skill evaluation and the 40-seed development closed-loop protocol. Once all three training stages succeed, one dependent job evaluates their final checkpoints on the unchanged 100-seed final-ID protocol. Final seeds never enter training or model selection. Since prior variants were evaluated on that pool, final comparisons remain exploratory.

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
