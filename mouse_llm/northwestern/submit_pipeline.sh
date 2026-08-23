#!/usr/bin/env bash
set -euo pipefail

repo_root="${MOUSE_LLM_REPO_ROOT:-$HOME/MouseMind}"
artifact_root="${MOUSE_LLM_ARTIFACT_ROOT:-/shares/bcs516/sh/minimind_artifacts}"
data_root="${MOUSE_LLM_DATA_ROOT:-$HOME/minimind_mouse_data}"
raw_data="${MOUSE_LLM_RAW_DATA:-$data_root/raw/mouse_data_processed.csv}"

export MOUSE_LLM_ARTIFACT_ROOT="$artifact_root"
export MOUSE_LLM_DATA_ROOT="$data_root"
export MOUSE_LLM_RAW_DATA="$raw_data"
export MOUSE_LLM_REPO_ROOT="$repo_root"

mkdir -p "$data_root/logs"
chmod 700 "$data_root" "$data_root/logs"

test -s "$raw_data"
test -s "$repo_root/mouse_llm/northwestern/prepare_mouse_data.sbatch"
test -s "$repo_root/mouse_llm/northwestern/train_and_evaluate.sbatch"
test -s "$repo_root/mouse_llm/northwestern/train_mlp.sbatch"

prepare_job="$(sbatch --parsable \
  --output="$data_root/logs/%x-%j.out" \
  --error="$data_root/logs/%x-%j.err" \
  "$repo_root/mouse_llm/northwestern/prepare_mouse_data.sbatch")"
train_job="$(sbatch --parsable \
  --dependency="afterok:$prepare_job" \
  --output="$data_root/logs/%x-%j.out" \
  --error="$data_root/logs/%x-%j.err" \
  "$repo_root/mouse_llm/northwestern/train_and_evaluate.sbatch")"
mlp_job="$(sbatch --parsable \
  --dependency="afterok:$prepare_job" \
  --output="$data_root/logs/%x-%j.out" \
  --error="$data_root/logs/%x-%j.err" \
  "$repo_root/mouse_llm/northwestern/train_mlp.sbatch")"

echo "PREPARE_JOB=$prepare_job"
echo "TRAIN_EVAL_JOB=$train_job"
echo "MLP_JOB=$mlp_job"
echo "Monitor: squeue -j $prepare_job,$train_job,$mlp_job"
