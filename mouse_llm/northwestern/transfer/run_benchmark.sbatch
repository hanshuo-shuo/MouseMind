#!/usr/bin/env bash
#SBATCH --job-name=mm-transfer
#SBATCH --partition=job
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=mouse-transfer-%x-%j.out
#SBATCH --error=mouse-transfer-%x-%j.err

set -euo pipefail
umask 077
source "$HOME/anaconda3/etc/profile.d/conda.sh"
conda activate /shares/bcs516/sh/minimind_env

repo_root="${MOUSE_TRANSFER_REPO_ROOT:-$HOME/mousemind-transfer-study}"
output_root="${MOUSE_TRANSFER_OUTPUT_ROOT:-/shares/bcs516/sh/minimind_mouse_transfer}"
seed_pool="${MOUSE_TRANSFER_SEED_POOL:-development}"
condition="${MOUSE_TRANSFER_CONDITION:-seen_instruction}"
run_id="${MOUSE_TRANSFER_RUN_ID:-slurm-${SLURM_JOB_ID}}"
output_dir="$output_root/$seed_pool/$condition/$run_id"

export CELLWORLD_CACHE="${CELLWORLD_CACHE:-$HOME/minimind_mouse_data/cellworld_cache}"
export PYTHONPATH="${MOUSE_LLM_RUNTIME_ROOT:-$HOME/minimind_mouse_data/runtime}:$repo_root:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM=false
mkdir -p "$output_dir"
cd "$repo_root"

read -r -a policies <<< "${MOUSE_TRANSFER_POLICIES:-random literal-direct-mlp literal-p1-rule literal-numeric literal-minimind aligned-goal-only aligned-p1-rule aligned-numeric aligned-minimind}"

python -m mouse_llm.evaluation.transfer_benchmark \
  --compatibility-audit "${MOUSE_TRANSFER_COMPATIBILITY_AUDIT:?set compatibility audit}" \
  --policies "${policies[@]}" \
  --reference-policy "${MOUSE_TRANSFER_REFERENCE_POLICY:-aligned-goal-only}" \
  --seed-pool "$seed_pool" \
  --condition "$condition" \
  --output-dir "$output_dir" \
  --mlp-checkpoint "${MOUSE_LLM_MLP_CHECKPOINT:-$HOME/minimind_mouse_data/formal_stage/20260822_p1/mlp_bc.pt}" \
  --planner-checkpoint "${MOUSE_P2_PLANNER_CHECKPOINT:-/shares/bcs516/sh/minimind_mouse_p2/models/p2-v1/numeric/planner_mlp.pt}" \
  --base-weight "${MOUSE_LLM_BASE_WEIGHT:-$HOME/minimind/out/full_sft_768.pth}" \
  --skill-lora-weight "${MOUSE_P2_SKILL_LORA_WEIGHT:-/shares/bcs516/sh/minimind_mouse_p2/models/p2-v1/minimind/lora_mousemind_skill_planner_768.pth}" \
  --tokenizer "$repo_root/model" \
  --device cuda

echo "TRANSFER_REPORT=$output_dir/closed_loop_metrics.json"
