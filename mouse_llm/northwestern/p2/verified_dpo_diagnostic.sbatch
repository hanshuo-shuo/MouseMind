#!/usr/bin/env bash
#SBATCH --job-name=mm-dpo-diagnostic
#SBATCH --partition=job
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=mouse-p2-%x-%j.out
#SBATCH --error=mouse-p2-%x-%j.err

set -euo pipefail
umask 077
source "$HOME/anaconda3/etc/profile.d/conda.sh"
conda activate /shares/bcs516/sh/minimind_env
repo_root="${MOUSE_LLM_REPO_ROOT:-$HOME/MouseMind}"
p2_root="${MOUSE_P2_ROOT:-/shares/bcs516/sh/minimind_mouse_p2}"
run_dir="$p2_root/verified_dpo/beta_0.1"
export CELLWORLD_CACHE="${CELLWORLD_CACHE:-$HOME/minimind_mouse_data/cellworld_cache}"
export PYTHONPATH="${MOUSE_LLM_RUNTIME_ROOT:-$HOME/minimind_mouse_data/runtime}:$repo_root:${PYTHONPATH:-}"
export PYTHONDONTWRITEBYTECODE=1 TOKENIZERS_PARALLELISM=false
cd "$repo_root"
python -m mouse_llm.evaluation.diagnose_verified_dpo \
  --trace "$run_dir/final_id/decision_trace_private.jsonl" \
  --base-weight "$repo_root/out/full_sft_768.pth" \
  --sft-lora-weight "${MOUSE_DPO_SFT_LORA_WEIGHT:?set MOUSE_DPO_SFT_LORA_WEIGHT}" \
  --mlp-checkpoint "$HOME/minimind_mouse_data/formal_stage/20260822_p1/mlp_bc.pt" \
  --tokenizer "$repo_root/model" --device cuda \
  --output "$run_dir/diagnostic.json"
