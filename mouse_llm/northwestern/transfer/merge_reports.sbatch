#!/usr/bin/env bash
#SBATCH --job-name=mm-transfer-merge
#SBATCH --partition=job
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=mouse-transfer-%x-%j.out
#SBATCH --error=mouse-transfer-%x-%j.err

set -euo pipefail
umask 077
source "$HOME/anaconda3/etc/profile.d/conda.sh"
conda activate /shares/bcs516/sh/minimind_env

repo_root="${MOUSE_TRANSFER_REPO_ROOT:-$HOME/mousemind-transfer-study}"
read -r -a runs <<< "${MOUSE_TRANSFER_MERGE_RUNS:?set policy-partition run directories}"
output_dir="${MOUSE_TRANSFER_MERGE_OUTPUT:?set merged output directory}"
export PYTHONPATH="$repo_root:${PYTHONPATH:-}"
cd "$repo_root"

python -m mouse_llm.evaluation.merge_transfer_runs \
  --runs "${runs[@]}" \
  --seed-pool "${MOUSE_TRANSFER_SEED_POOL:-final_test}" \
  --condition "${MOUSE_TRANSFER_CONDITION:-seen_instruction}" \
  --reference-policy "${MOUSE_TRANSFER_REFERENCE_POLICY:-aligned-goal-only}" \
  --output-dir "$output_dir"

echo "MERGED_TRANSFER_REPORT=$output_dir/closed_loop_metrics.json"
