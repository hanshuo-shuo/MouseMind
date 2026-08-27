#!/usr/bin/env bash
#SBATCH --job-name=mm-p2-direct
#SBATCH --partition=job
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=mouse-p2-%x-%j.out
#SBATCH --error=mouse-p2-%x-%j.err

set -euo pipefail
umask 077
source "$HOME/anaconda3/etc/profile.d/conda.sh"
conda activate /shares/bcs516/sh/minimind_env
repo_root="${MOUSE_LLM_REPO_ROOT:-$HOME/minimind}"
p2_root="${MOUSE_P2_ROOT:?set MOUSE_P2_ROOT}"
output_dir="$p2_root/closed_loop/final_id/direct_minimind"
mkdir -p "$output_dir"
export CELLWORLD_CACHE="${CELLWORLD_CACHE:?set CELLWORLD_CACHE}"
export PYTHONPATH="${MOUSE_LLM_RUNTIME_ROOT:-$HOME/minimind_mouse_data/runtime}:$repo_root:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM=false
cd "$repo_root"

python -m mouse_llm.evaluation.p2_benchmark \
  --policies direct-minimind-base direct-minimind-lora \
  --reference-policy direct-minimind-base \
  --output-dir "$output_dir" \
  --seed-pool final_id_test \
  --ood id \
  --mlp-checkpoint "${MOUSE_LLM_MLP_CHECKPOINT:?set MLP checkpoint}" \
  --planner-checkpoint "${MOUSE_P2_PLANNER_CHECKPOINT:?set numeric planner}" \
  --base-weight "${MOUSE_LLM_BASE_WEIGHT:?set base weight}" \
  --direct-lora-weight "${MOUSE_LLM_LORA_WEIGHT:?set direct LoRA}" \
  --tokenizer "$repo_root/model" \
  --device cuda
echo "P2_FINAL_DIRECT=$output_dir/closed_loop_metrics.json"
