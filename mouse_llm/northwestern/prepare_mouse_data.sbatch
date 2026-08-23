#!/usr/bin/env bash
#SBATCH --job-name=mm-mouse-prepare
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:20:00
#SBATCH --output=mouse-llm-%x-%j.out
#SBATCH --error=mouse-llm-%x-%j.err

set -euo pipefail
umask 077

source "$HOME/anaconda3/etc/profile.d/conda.sh"
conda activate /shares/bcs516/sh/minimind_env

data_root="${MOUSE_LLM_DATA_ROOT:-$HOME/minimind_mouse_data}"
repo_root="${MOUSE_LLM_REPO_ROOT:-$HOME/MouseMind}"
raw_data="${MOUSE_LLM_RAW_DATA:-$data_root/raw/mouse_data_processed.csv}"
processed_dir="${MOUSE_LLM_PROCESSED_DIR:-$data_root/processed/v1}"
log_out="$data_root/logs/${SLURM_JOB_NAME}-${SLURM_JOB_ID}.out"
log_err="$data_root/logs/${SLURM_JOB_NAME}-${SLURM_JOB_ID}.err"
trap 'chmod 600 "$log_out" "$log_err" 2>/dev/null || true' EXIT

test -s "$raw_data"
mkdir -p "$processed_dir" "$data_root/cache/huggingface" "$data_root/logs"
chmod 700 "$data_root" "$processed_dir" "$data_root/cache" "$data_root/cache/huggingface" "$data_root/logs"

export HF_HOME="$data_root/cache/huggingface"
export HF_DATASETS_CACHE="$data_root/cache/huggingface/datasets"
export PYTHONDONTWRITEBYTECODE=1

cd "$repo_root"
python -m mouse_llm.data.prepare_dataset \
  --input "$raw_data" \
  --output-dir "$processed_dir" \
  --action-count 295 \
  --seed 42 \
  --train-ratio 0.8 \
  --validation-ratio 0.1 \
  --continuity-atol 0.00001 \
  --precision 4 \
  --max-train-samples 60000 \
  --max-validation-samples 5000 \
  --max-test-samples 5000

python -m mouse_llm.privacy_guard --repo-root "$repo_root"
ls -lh "$processed_dir"/manifest.json "$processed_dir"/*.jsonl
