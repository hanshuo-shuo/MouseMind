#!/usr/bin/env bash
#SBATCH --job-name=mm-transfer-gif
#SBATCH --partition=job
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=mouse-transfer-%x-%j.out
#SBATCH --error=mouse-transfer-%x-%j.err

set -euo pipefail
umask 077
source "$HOME/anaconda3/etc/profile.d/conda.sh"
conda activate /shares/bcs516/sh/minimind_env

repo_root="${MOUSE_TRANSFER_REPO_ROOT:-$HOME/mousemind-transfer-study}"
output="${MOUSE_TRANSFER_GIF_OUTPUT:?set MOUSE_TRANSFER_GIF_OUTPUT}"
export CELLWORLD_CACHE="${CELLWORLD_CACHE:-$HOME/minimind_mouse_data/cellworld_cache}"
export PYTHONPATH="${MOUSE_LLM_RUNTIME_ROOT:-$HOME/minimind_mouse_data/runtime}:$repo_root:${PYTHONPATH:-}"
mkdir -p "$(dirname "$output")"
cd "$repo_root"

python -m mouse_llm.evaluation.render_transfer_rollout \
  --seed "${MOUSE_TRANSFER_GIF_SEED:-42000}" \
  --base-weight "${MOUSE_LLM_BASE_WEIGHT:-$HOME/minimind/out/full_sft_768.pth}" \
  --skill-lora-weight "${MOUSE_P2_SKILL_LORA_WEIGHT:-/shares/bcs516/sh/minimind_mouse_p2/models/p2-v1/minimind/lora_mousemind_skill_planner_768.pth}" \
  --tokenizer "$repo_root/model" \
  --device cuda \
  --output "$output"

echo "TRANSFER_GIF=$output"
