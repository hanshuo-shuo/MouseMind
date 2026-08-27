#!/usr/bin/env bash
#SBATCH --job-name=mm-p0-json
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=48G
#SBATCH --time=00:30:00
#SBATCH --output=mouse-p2-%x-%j.out
#SBATCH --error=mouse-p2-%x-%j.err

set -euo pipefail
umask 077
source "$HOME/anaconda3/etc/profile.d/conda.sh"
conda activate /shares/bcs516/sh/minimind_env
repo_root="${MOUSE_LLM_REPO_ROOT:-$HOME/minimind}"
output_dir="${MOUSE_P0_CONSTRAINED_OUTPUT:?set MOUSE_P0_CONSTRAINED_OUTPUT outside Git}"
mkdir -p "$output_dir"
export TOKENIZERS_PARALLELISM=false
cd "$repo_root"
python -m mouse_llm.evaluation.evaluate_policy \
  --base-weight "${MOUSE_LLM_BASE_WEIGHT:?set MOUSE_LLM_BASE_WEIGHT}" \
  --lora-weight "${MOUSE_LLM_LORA_WEIGHT:?set MOUSE_LLM_LORA_WEIGHT}" \
  --tokenizer "$repo_root/model" \
  --test-data "${MOUSE_LLM_TEST_DATA:?set MOUSE_LLM_TEST_DATA}" \
  --action-catalog "$repo_root/mouse_llm/envs/mice/assets/action_catalog_21_05.json" \
  --output-dir "$output_dir" \
  --max-samples 512 \
  --decode-mode json-constrained \
  --seed 42
echo "DIRECT_CONSTRAINED_METRICS=$output_dir/comparison_metrics.json"
