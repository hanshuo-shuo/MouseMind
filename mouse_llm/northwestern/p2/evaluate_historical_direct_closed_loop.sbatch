#!/usr/bin/env bash
#SBATCH --job-name=mm-p0-loop
#SBATCH --partition=job
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=48G
#SBATCH --time=06:00:00
#SBATCH --output=mouse-p2-%x-%j.out
#SBATCH --error=mouse-p2-%x-%j.err

set -euo pipefail
umask 077
source "$HOME/anaconda3/etc/profile.d/conda.sh"
conda activate /shares/bcs516/sh/minimind_env
repo_root="${MOUSE_LLM_REPO_ROOT:-$HOME/minimind}"
output_dir="${MOUSE_P0_CLOSED_LOOP_OUTPUT:?set MOUSE_P0_CLOSED_LOOP_OUTPUT outside Git}"
export CELLWORLD_CACHE="${CELLWORLD_CACHE:?set CELLWORLD_CACHE}"
export PYTHONPATH="${MOUSE_LLM_RUNTIME_ROOT:-$HOME/minimind_mouse_data/runtime}:$repo_root:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM=false
mkdir -p "$output_dir"
cd "$repo_root"
python -m mouse_llm.evaluation.closed_loop \
  --policies minimind-base minimind-lora \
  --reference-policy minimind-base \
  --base-weight "${MOUSE_LLM_BASE_WEIGHT:?set MOUSE_LLM_BASE_WEIGHT}" \
  --lora-weight "${MOUSE_LLM_LORA_WEIGHT:?set MOUSE_LLM_LORA_WEIGHT}" \
  --tokenizer "$repo_root/model" \
  --output-dir "$output_dir" \
  --episodes 100 \
  --seed-start 1000 \
  --decode-mode json-constrained \
  --device cuda
echo "HISTORICAL_DIRECT_CLOSED_LOOP=$output_dir/closed_loop_metrics.json"
