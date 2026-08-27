#!/usr/bin/env bash
#SBATCH --job-name=mm-p2-loop
#SBATCH --partition=job
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=18:00:00
#SBATCH --output=mouse-p2-%x-%j.out
#SBATCH --error=mouse-p2-%x-%j.err

set -euo pipefail
umask 077
source "$HOME/anaconda3/etc/profile.d/conda.sh"
conda activate /shares/bcs516/sh/minimind_env

repo_root="${MOUSE_LLM_REPO_ROOT:-$HOME/minimind}"
output_root="${MOUSE_P2_OUTPUT_ROOT:?set MOUSE_P2_OUTPUT_ROOT outside Git}"
cellworld_cache="${CELLWORLD_CACHE:?set CELLWORLD_CACHE}"
read -r -a policies <<< "${MOUSE_P2_POLICIES:-direct-mlp p1-rule numeric-learned numeric-verified}"
condition="${MOUSE_P2_OOD_CONDITION:-id}"
seed_pool="${MOUSE_P2_SEED_POOL:-final_id_test}"
output_dir="$output_root/${seed_pool}/${condition}/${MOUSE_P2_RUN_ID:-slurm-${SLURM_JOB_ID}}"
mkdir -p "$output_dir"
export CELLWORLD_CACHE
export PYTHONPATH="${MOUSE_LLM_RUNTIME_ROOT:-$HOME/minimind_mouse_data/runtime}:$repo_root:${PYTHONPATH:-}"
export PYTHONDONTWRITEBYTECODE=1
export TOKENIZERS_PARALLELISM=false
cd "$repo_root"

command=(python -m mouse_llm.evaluation.p2_benchmark
  --policies "${policies[@]}"
  --output-dir "$output_dir"
  --seed-pool "$seed_pool"
  --ood "$condition"
  --mlp-checkpoint "${MOUSE_LLM_MLP_CHECKPOINT:?set MOUSE_LLM_MLP_CHECKPOINT}"
  --planner-checkpoint "${MOUSE_P2_PLANNER_CHECKPOINT:?set MOUSE_P2_PLANNER_CHECKPOINT}"
  --risk-checkpoint "${MOUSE_P2_RISK_CHECKPOINT:?set MOUSE_P2_RISK_CHECKPOINT}"
  --risk-threshold "${MOUSE_P2_RISK_THRESHOLD:-0.5}"
  --planner-horizon "${MOUSE_P2_PLANNER_HORIZON:-4}"
  --device "${MOUSE_P2_DEVICE:-cpu}")

if [[ " ${policies[*]} " == *" minimind-"* ]]; then
  command+=(
    --base-weight "${MOUSE_LLM_BASE_WEIGHT:?set MOUSE_LLM_BASE_WEIGHT}"
    --skill-lora-weight "${MOUSE_P2_SKILL_LORA_WEIGHT:?set MOUSE_P2_SKILL_LORA_WEIGHT}"
    --tokenizer "$repo_root/model"
    --device cuda)
fi
if [[ " ${policies[*]} " == *" direct-minimind-"* ]]; then
  command+=(
    --base-weight "${MOUSE_LLM_BASE_WEIGHT:?set MOUSE_LLM_BASE_WEIGHT}"
    --direct-lora-weight "${MOUSE_LLM_LORA_WEIGHT:?set MOUSE_LLM_LORA_WEIGHT}"
    --tokenizer "$repo_root/model"
    --device cuda)
fi
"${command[@]}"
echo "P2_CLOSED_LOOP_REPORT=$output_dir/closed_loop_metrics.json"
