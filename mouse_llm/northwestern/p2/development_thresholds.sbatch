#!/usr/bin/env bash
#SBATCH --job-name=mm-p2-threshold
#SBATCH --partition=job
#SBATCH --array=0-4%2
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
thresholds=(0.10 0.30 0.50 0.70 0.90)
threshold="${thresholds[$SLURM_ARRAY_TASK_ID]}"
horizon="${MOUSE_P2_SELECTED_HORIZON:?freeze MOUSE_P2_SELECTED_HORIZON from development horizon sweep}"
output_dir="$output_root/development/threshold-${threshold}"
export CELLWORLD_CACHE="${CELLWORLD_CACHE:?set CELLWORLD_CACHE}"
export PYTHONPATH="${MOUSE_LLM_RUNTIME_ROOT:-$HOME/minimind_mouse_data/runtime}:$repo_root:${PYTHONPATH:-}"
mkdir -p "$output_dir"
cd "$repo_root"
python -m mouse_llm.evaluation.p2_benchmark \
  --policies p1-rule numeric-learned numeric-verified \
  --reference-policy numeric-learned \
  --output-dir "$output_dir" \
  --seed-pool development \
  --ood id \
  --mlp-checkpoint "${MOUSE_LLM_MLP_CHECKPOINT:?set MLP checkpoint}" \
  --planner-checkpoint "${MOUSE_P2_PLANNER_CHECKPOINT:?set planner checkpoint}" \
  --risk-checkpoint "${MOUSE_P2_RISK_CHECKPOINT:?set risk checkpoint}" \
  --risk-threshold "$threshold" \
  --planner-horizon "$horizon" \
  --device cpu
