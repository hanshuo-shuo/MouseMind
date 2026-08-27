#!/usr/bin/env bash
#SBATCH --job-name=mm-p2-horizon
#SBATCH --partition=job
#SBATCH --array=0-3%2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=mouse-p2-%x-%A_%a.out
#SBATCH --error=mouse-p2-%x-%A_%a.err

set -euo pipefail
umask 077
source "$HOME/anaconda3/etc/profile.d/conda.sh"
conda activate /shares/bcs516/sh/minimind_env
repo_root="${MOUSE_LLM_REPO_ROOT:-$HOME/minimind}"
output_root="${MOUSE_P2_OUTPUT_ROOT:?set MOUSE_P2_OUTPUT_ROOT}"
horizons=(1 4 8 16)
horizon="${horizons[$SLURM_ARRAY_TASK_ID]}"
output_dir="$output_root/development/horizon-${horizon}"
export CELLWORLD_CACHE="${CELLWORLD_CACHE:?set CELLWORLD_CACHE}"
export PYTHONPATH="${MOUSE_LLM_RUNTIME_ROOT:-$HOME/minimind_mouse_data/runtime}:$repo_root:${PYTHONPATH:-}"
mkdir -p "$output_dir"
cd "$repo_root"
python -m mouse_llm.evaluation.p2_benchmark \
  --policies direct-mlp p1-rule numeric-learned \
  --reference-policy direct-mlp \
  --output-dir "$output_dir" \
  --seed-pool development \
  --ood id \
  --mlp-checkpoint "${MOUSE_LLM_MLP_CHECKPOINT:?set MLP checkpoint}" \
  --planner-checkpoint "${MOUSE_P2_PLANNER_CHECKPOINT:?set planner checkpoint}" \
  --planner-horizon "$horizon" \
  --device cpu
