#!/usr/bin/env bash
#SBATCH --job-name=mm-mouse-online
#SBATCH --partition=job
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --output=mouse-llm-%x-%j.out
#SBATCH --error=mouse-llm-%x-%j.err

set -euo pipefail
umask 077

source "$HOME/anaconda3/etc/profile.d/conda.sh"
conda activate /shares/bcs516/sh/minimind_env

data_root="${MOUSE_LLM_DATA_ROOT:-$HOME/minimind_mouse_data}"
repo_root="${MOUSE_LLM_REPO_ROOT:-$HOME/MouseMind}"
base_weight="${MOUSE_LLM_BASE_WEIGHT:-$repo_root/out/full_sft_768.pth}"
lora_weight="${MOUSE_LLM_LORA_WEIGHT:?set MOUSE_LLM_LORA_WEIGHT to the trained LoRA .pth}"
mlp_checkpoint="${MOUSE_LLM_MLP_CHECKPOINT:?set MOUSE_LLM_MLP_CHECKPOINT to mlp_bc.pt}"
cellworld_cache="${CELLWORLD_CACHE:?set CELLWORLD_CACHE to the offline Cellworld cache}"
run_id="${MOUSE_LLM_BENCHMARK_RUN_ID:-slurm-${SLURM_JOB_ID}}"
output_dir="$data_root/closed_loop/$run_id"
read -r -a policies <<< "${MOUSE_LLM_POLICIES:-random mlp-bc hierarchical-mlp minimind-base minimind-lora}"
reference_policy="${MOUSE_LLM_REFERENCE_POLICY:-${policies[0]}}"

test -s "$base_weight"
test -s "$lora_weight"
test -s "$mlp_checkpoint"
test -d "$cellworld_cache"
test -s "$cellworld_cache/paths/hexagonal.21_05.astar.robot"
mkdir -p "$output_dir"
chmod 700 "$data_root" "$output_dir"

export CELLWORLD_CACHE="$cellworld_cache"
export PYTHONDONTWRITEBYTECODE=1
export TOKENIZERS_PARALLELISM=false

cd "$repo_root"
python -m mouse_llm.evaluation.closed_loop \
  --policies "${policies[@]}" \
  --reference-policy "$reference_policy" \
  --mlp-checkpoint "$mlp_checkpoint" \
  --base-weight "$base_weight" \
  --lora-weight "$lora_weight" \
  --tokenizer "$repo_root/model" \
  --output-dir "$output_dir" \
  --episodes "${MOUSE_LLM_CLOSED_LOOP_EPISODES:-100}" \
  --seed-start "${MOUSE_LLM_CLOSED_LOOP_SEED_START:-1000}" \
  --decode-mode json-constrained \
  --device cuda

if git -C "$repo_root" rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  python -m mouse_llm.privacy_guard --repo-root "$repo_root"
fi
echo "CLOSED_LOOP_METRICS=$output_dir/closed_loop_metrics.json"
echo "CLOSED_LOOP_EPISODES=$output_dir/closed_loop_episodes.csv"
