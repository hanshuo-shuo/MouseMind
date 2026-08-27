#!/usr/bin/env bash
#SBATCH --job-name=mm-p2-small
#SBATCH --partition=job
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=mouse-p2-%x-%j.out
#SBATCH --error=mouse-p2-%x-%j.err

set -euo pipefail
umask 077
source "$HOME/anaconda3/etc/profile.d/conda.sh"
conda activate /shares/bcs516/sh/minimind_env

repo_root="${MOUSE_LLM_REPO_ROOT:-$HOME/minimind}"
dataset_dir="${MOUSE_P2_DATASET_DIR:?set MOUSE_P2_DATASET_DIR}"
model_dir="${MOUSE_P2_MODEL_DIR:?set MOUSE_P2_MODEL_DIR outside Git}"
mkdir -p "$model_dir/numeric" "$model_dir/risk"
cd "$repo_root"

python -m mouse_llm.baselines.planner_mlp \
  --train-data "$dataset_dir/planner_train.jsonl" \
  --validation-data "$dataset_dir/planner_validation.jsonl" \
  --seen-test-data "$dataset_dir/planner_seen_test.jsonl" \
  --unseen-test-data "$dataset_dir/planner_unseen_test.jsonl" \
  --output-dir "$model_dir/numeric" \
  --device cpu \
  --research-evidence

python -m mouse_llm.training.train_risk_critic \
  --train-data "$dataset_dir/risk_train.jsonl" \
  --validation-data "$dataset_dir/risk_validation.jsonl" \
  --output-dir "$model_dir/risk" \
  --device cpu \
  --research-evidence

echo "PLANNER_CHECKPOINT=$model_dir/numeric/planner_mlp.pt"
echo "RISK_CHECKPOINT=$model_dir/risk/risk_critic.pt"
