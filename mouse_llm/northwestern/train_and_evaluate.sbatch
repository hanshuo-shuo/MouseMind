#!/usr/bin/env bash
#SBATCH --job-name=mm-mouse-lora
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=48G
#SBATCH --time=00:30:00
#SBATCH --output=mouse-llm-%x-%j.out
#SBATCH --error=mouse-llm-%x-%j.err

set -euo pipefail
umask 077

source "$HOME/anaconda3/etc/profile.d/conda.sh"
conda activate /shares/bcs516/sh/minimind_env

artifact_root="${MOUSE_LLM_ARTIFACT_ROOT:-/shares/bcs516/sh/minimind_artifacts}"
data_root="${MOUSE_LLM_DATA_ROOT:-$HOME/minimind_mouse_data}"
repo_root="${MOUSE_LLM_REPO_ROOT:-$HOME/MouseMind}"
processed_dir="${MOUSE_LLM_PROCESSED_DIR:-$data_root/processed/v1}"
run_id="${MOUSE_LLM_RUN_ID:-slurm-${SLURM_JOB_ID}}"
log_out="$data_root/logs/${SLURM_JOB_NAME}-${SLURM_JOB_ID}.out"
log_err="$data_root/logs/${SLURM_JOB_NAME}-${SLURM_JOB_ID}.err"
trap 'chmod 600 "$log_out" "$log_err" 2>/dev/null || true' EXIT
model_dir="$artifact_root/out/mouse_policy/$run_id"
eval_dir="$data_root/eval/$run_id"
report_dir="$data_root/reports/$run_id"
base_weight="$repo_root/out/full_sft_768.pth"
lora_name="${MOUSE_LLM_LORA_NAME:-lora_mouse_policy}"
epochs="${MOUSE_LLM_EPOCHS:-2}"
batch_size="${MOUSE_LLM_BATCH_SIZE:-32}"
max_seq_len="${MOUSE_LLM_MAX_SEQ_LEN:-256}"
num_workers="${MOUSE_LLM_NUM_WORKERS:-4}"
save_interval="${MOUSE_LLM_SAVE_INTERVAL:-1000}"
max_eval_samples="${MOUSE_LLM_MAX_EVAL_SAMPLES:-512}"

mkdir -p \
  "$model_dir" \
  "$eval_dir" \
  "$report_dir" \
  "$data_root/cache/huggingface" \
  "$data_root/cache/matplotlib"
chmod 700 "$data_root" "$eval_dir" "$report_dir" "$data_root/cache" "$data_root/cache/huggingface" "$data_root/cache/matplotlib"

export HF_HOME="$data_root/cache/huggingface"
export HF_DATASETS_CACHE="$data_root/cache/huggingface/datasets"
export MPLCONFIGDIR="$data_root/cache/matplotlib"
export PYTHONDONTWRITEBYTECODE=1
export TOKENIZERS_PARALLELISM=false

test -s "$processed_dir/train.jsonl"
test -s "$processed_dir/test.jsonl"
test -s "$base_weight"

cd "$repo_root/trainer"
python train_lora.py \
  --data_path "$processed_dir/train.jsonl" \
  --save_dir "$model_dir" \
  --lora_name "$lora_name" \
  --from_weight full_sft \
  --epochs "$epochs" \
  --batch_size "$batch_size" \
  --accumulation_steps 1 \
  --learning_rate 0.0001 \
  --hidden_size 768 \
  --num_hidden_layers 8 \
  --max_seq_len "$max_seq_len" \
  --num_workers "$num_workers" \
  --log_interval 25 \
  --save_interval "$save_interval"

lora_weight="$model_dir/${lora_name}_768.pth"
test -s "$lora_weight"

cd "$repo_root"
python -m mouse_llm.evaluation.evaluate_policy \
  --base-weight "$base_weight" \
  --lora-weight "$lora_weight" \
  --tokenizer "$repo_root/model" \
  --test-data "$processed_dir/test.jsonl" \
  --action-catalog "$repo_root/mouse_llm/envs/mice/assets/action_catalog_21_05.json" \
  --output-dir "$eval_dir" \
  --hidden-size 768 \
  --num-hidden-layers 8 \
  --max-seq-len 256 \
  --max-new-tokens 24 \
  --max-samples "$max_eval_samples" \
  --decode-mode "${MOUSE_LLM_DECODE_MODE:-free}" \
  --seed 42

python -m mouse_llm.evaluation.plot_comparison \
  --metrics "$eval_dir/comparison_metrics.json" \
  --output-dir "$report_dir"

python -m mouse_llm.privacy_guard --repo-root "$repo_root"

echo "MODEL_WEIGHT=$lora_weight"
echo "METRICS=$eval_dir/comparison_metrics.json"
echo "PRIVATE_PREDICTIONS=$eval_dir/predictions_private.jsonl"
echo "FIGURE_PNG=$report_dir/mouse_policy_before_after.png"
echo "FIGURE_SVG=$report_dir/mouse_policy_before_after.svg"
