#!/usr/bin/env bash
#SBATCH --job-name=mm-mouse-mlp
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:15:00
#SBATCH --output=mouse-llm-%x-%j.out
#SBATCH --error=mouse-llm-%x-%j.err

set -euo pipefail
umask 077

source "$HOME/anaconda3/etc/profile.d/conda.sh"
conda activate /shares/bcs516/sh/minimind_env

data_root="${MOUSE_LLM_DATA_ROOT:-$HOME/minimind_mouse_data}"
repo_root="${MOUSE_LLM_REPO_ROOT:-$HOME/MouseMind}"
processed_dir="${MOUSE_LLM_PROCESSED_DIR:-$data_root/processed/v1}"
run_id="${MOUSE_LLM_MLP_RUN_ID:-slurm-${SLURM_JOB_ID}}"
output_dir="${MOUSE_LLM_MLP_OUTPUT_DIR:-$data_root/mlp/$run_id}"
catalog="$repo_root/mouse_llm/envs/mice/assets/action_catalog_21_05.json"

mkdir -p "$output_dir"
chmod 700 "$data_root" "$output_dir"
test -s "$processed_dir/train.jsonl"
test -s "$processed_dir/validation.jsonl"
test -s "$processed_dir/test.jsonl"

cd "$repo_root"
python -m mouse_llm.baselines.mlp_bc \
  --train-data "$processed_dir/train.jsonl" \
  --validation-data "$processed_dir/validation.jsonl" \
  --test-data "$processed_dir/test.jsonl" \
  --action-catalog "$catalog" \
  --output-dir "$output_dir" \
  --epochs "${MOUSE_LLM_MLP_EPOCHS:-30}" \
  --batch-size "${MOUSE_LLM_MLP_BATCH_SIZE:-512}" \
  --patience "${MOUSE_LLM_MLP_PATIENCE:-5}" \
  --seed 42

python -m mouse_llm.privacy_guard --repo-root "$repo_root"
echo "MLP_CHECKPOINT=$output_dir/mlp_bc.pt"
echo "MLP_METRICS=$output_dir/mlp_metrics.json"
