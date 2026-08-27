#!/usr/bin/env bash
#SBATCH --job-name=mm-p2-skill
#SBATCH --partition=job
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=mouse-p2-%x-%j.out
#SBATCH --error=mouse-p2-%x-%j.err

set -euo pipefail
umask 077
source "$HOME/anaconda3/etc/profile.d/conda.sh"
conda activate /shares/bcs516/sh/minimind_env

repo_root="${MOUSE_LLM_REPO_ROOT:-$HOME/minimind}"
dataset_dir="${MOUSE_P2_DATASET_DIR:?set MOUSE_P2_DATASET_DIR}"
model_dir="${MOUSE_P2_MODEL_DIR:?set MOUSE_P2_MODEL_DIR outside Git}"
mkdir -p "$model_dir/minimind"
cd "$repo_root"

python -m mouse_llm.training.train_skill_planner \
  --repo-root "$repo_root" \
  --data-path "$dataset_dir/planner_train.jsonl" \
  --save-dir "$model_dir/minimind" \
  --epochs "${MOUSE_P2_SKILL_EPOCHS:-3}" \
  --batch-size "${MOUSE_P2_SKILL_BATCH_SIZE:-16}" \
  --max-seq-len 1024 \
  --device cuda:0

python -m mouse_llm.evaluation.evaluate_skill_planner \
  --base-weight "${MOUSE_LLM_BASE_WEIGHT:?set MOUSE_LLM_BASE_WEIGHT}" \
  --lora-weight "$model_dir/minimind/lora_mousemind_skill_planner_768.pth" \
  --tokenizer "$repo_root/model" \
  --seen-test-data "$dataset_dir/planner_seen_test.jsonl" \
  --unseen-test-data "$dataset_dir/planner_unseen_test.jsonl" \
  --output "$model_dir/minimind/offline_metrics.json" \
  --device cuda \
  --research-evidence
