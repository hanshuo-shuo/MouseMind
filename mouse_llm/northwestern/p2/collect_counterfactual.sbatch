#!/usr/bin/env bash
#SBATCH --job-name=mm-p2-data
#SBATCH --partition=job
#SBATCH --nodes=1
#SBATCH --ntasks=1
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
p2_root="${MOUSE_P2_ROOT:?set MOUSE_P2_ROOT outside Git}"
mlp_checkpoint="${MOUSE_LLM_MLP_CHECKPOINT:?set MOUSE_LLM_MLP_CHECKPOINT}"
cellworld_cache="${CELLWORLD_CACHE:?set CELLWORLD_CACHE}"
run_dir="$p2_root/data/${MOUSE_P2_RUN_ID:-slurm-${SLURM_JOB_ID}}"
mkdir -p "$run_dir"
export CELLWORLD_CACHE
export PYTHONPATH="${MOUSE_LLM_RUNTIME_ROOT:-$HOME/minimind_mouse_data/runtime}:$repo_root:${PYTHONPATH:-}"
export PYTHONDONTWRITEBYTECODE=1
cd "$repo_root"

python -m mouse_llm.data.collect_planner_anchors \
  --output "$run_dir/anchors_private.jsonl" \
  --aggregate-output "$run_dir/anchor_aggregate.json" \
  --mlp-checkpoint "$mlp_checkpoint" \
  --max-anchors-per-seed "${MOUSE_P2_ANCHORS_PER_SEED:-6}" \
  --anchor-stride "${MOUSE_P2_ANCHOR_STRIDE:-8}" \
  --device cpu

python -m mouse_llm.data.build_counterfactual_skill_dataset \
  --anchors "$run_dir/anchors_private.jsonl" \
  --output-dir "$run_dir/dataset" \
  --aggregate-output "$run_dir/counterfactual_aggregate.json" \
  --mlp-checkpoint "$mlp_checkpoint" \
  --horizons 4 8 \
  --label-horizon 8 \
  --device cpu

echo "P2_DATASET_DIR=$run_dir/dataset"
