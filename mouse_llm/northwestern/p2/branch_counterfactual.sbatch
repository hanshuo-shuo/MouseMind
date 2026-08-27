#!/usr/bin/env bash
#SBATCH --job-name=mm-p2-branch
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
run_dir="${MOUSE_P2_DATA_RUN_DIR:?set the completed anchor collection directory}"
mlp_checkpoint="${MOUSE_LLM_MLP_CHECKPOINT:?set MOUSE_LLM_MLP_CHECKPOINT}"
export CELLWORLD_CACHE="${CELLWORLD_CACHE:?set CELLWORLD_CACHE}"
export PYTHONPATH="${MOUSE_LLM_RUNTIME_ROOT:-$HOME/minimind_mouse_data/runtime}:$repo_root:${PYTHONPATH:-}"
export PYTHONDONTWRITEBYTECODE=1
cd "$repo_root"
test -s "$run_dir/anchors_private.jsonl"
python -m mouse_llm.data.build_counterfactual_skill_dataset \
  --anchors "$run_dir/anchors_private.jsonl" \
  --output-dir "$run_dir/dataset" \
  --aggregate-output "$run_dir/counterfactual_aggregate.json" \
  --mlp-checkpoint "$mlp_checkpoint" \
  --horizons 4 8 \
  --label-horizon 8 \
  --device cpu
echo "P2_DATASET_DIR=$run_dir/dataset"
