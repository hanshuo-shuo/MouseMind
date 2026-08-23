#!/usr/bin/env bash
#SBATCH --job-name=mm-mouse-synthetic-smoke
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:20:00
#SBATCH --output=mouse-llm-%x-%j.out
#SBATCH --error=mouse-llm-%x-%j.err

set -euo pipefail
umask 007

source "$HOME/anaconda3/etc/profile.d/conda.sh"
conda activate /shares/bcs516/sh/minimind_env

artifact_root="${MOUSE_LLM_ARTIFACT_ROOT:-/shares/bcs516/sh/minimind_artifacts}"
data_root="${MOUSE_LLM_DATA_ROOT:-$HOME/minimind_mouse_data}"
repo_root="${MOUSE_LLM_REPO_ROOT:-$HOME/MouseMind}"
synthetic_dir="$data_root/synthetic_smoke/v1"

cd "$repo_root"
python -m mouse_llm.data.create_synthetic_smoke \
  --output-dir "$synthetic_dir" \
  --train-samples 64 \
  --validation-samples 16 \
  --test-samples 16 \
  --constant-action 24

export MOUSE_LLM_PROCESSED_DIR="$synthetic_dir"
export MOUSE_LLM_DATA_ROOT="$data_root"
export MOUSE_LLM_RUN_ID="synthetic-smoke-${SLURM_JOB_ID}"
export MOUSE_LLM_LORA_NAME="lora_mouse_policy_synthetic_smoke"
export MOUSE_LLM_EPOCHS=20
export MOUSE_LLM_BATCH_SIZE=8
export MOUSE_LLM_MAX_SEQ_LEN=256
export MOUSE_LLM_NUM_WORKERS=2
export MOUSE_LLM_SAVE_INTERVAL=1000
export MOUSE_LLM_MAX_EVAL_SAMPLES=16

bash "$repo_root/mouse_llm/northwestern/train_and_evaluate.sbatch"
