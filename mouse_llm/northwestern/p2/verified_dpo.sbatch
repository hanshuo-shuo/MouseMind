#!/usr/bin/env bash
#SBATCH --job-name=mm-verified-dpo
#SBATCH --partition=job
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=18:00:00
#SBATCH --output=mouse-p2-%x-%j.out
#SBATCH --error=mouse-p2-%x-%j.err

set -euo pipefail
umask 077
source "$HOME/anaconda3/etc/profile.d/conda.sh"
conda activate /shares/bcs516/sh/minimind_env

repo_root="${MOUSE_LLM_REPO_ROOT:-$HOME/MouseMind}"
p2_root="${MOUSE_P2_ROOT:-/shares/bcs516/sh/minimind_mouse_p2}"
beta="${MOUSE_DPO_BETA:?set MOUSE_DPO_BETA}"
case "$beta" in 0.05|0.1|0.2) ;; *) echo "Unapproved beta: $beta" >&2; exit 2;; esac
run_dir="$p2_root/verified_dpo/beta_$beta"
data_dir="$p2_root/verified_dpo/data"
mkdir -p "$run_dir"
export CELLWORLD_CACHE="${CELLWORLD_CACHE:-$HOME/minimind_mouse_data/cellworld_cache}"
export PYTHONPATH="${MOUSE_LLM_RUNTIME_ROOT:-$HOME/minimind_mouse_data/runtime}:$repo_root:${PYTHONPATH:-}"
export PYTHONDONTWRITEBYTECODE=1 TOKENIZERS_PARALLELISM=false
cd "$repo_root"

python -m mouse_llm.training.train_verified_dpo \
  --train-data "$data_dir/dpo_train.jsonl" \
  --base-weight "$HOME/MouseMind/out/full_sft_768.pth" \
  --sft-lora-weight "${MOUSE_DPO_SFT_LORA_WEIGHT:?set MOUSE_DPO_SFT_LORA_WEIGHT}" \
  --tokenizer "$repo_root/model" \
  --output-dir "$run_dir" --beta "$beta"

weight="$run_dir/lora_mousemind_verified_dpo_768.pth"
offline_flags=(--only-lora --no-ablations)
if [[ "${MOUSE_DPO_SEED_CLEAN:-0}" == 1 ]]; then offline_flags+=(--research-evidence); fi
python -m mouse_llm.evaluation.evaluate_skill_planner \
  --base-weight "$HOME/MouseMind/out/full_sft_768.pth" \
  --lora-weight "$weight" --tokenizer "$repo_root/model" \
  --seen-test-data "$data_dir/dpo_seen_test.jsonl" \
  --unseen-test-data "$data_dir/dpo_unseen_test.jsonl" \
  --output "$run_dir/offline_metrics.json" \
  --device cuda "${offline_flags[@]}"

python -m mouse_llm.evaluation.p2_benchmark \
  --policies minimind-learned \
  --reference-policy minimind-learned \
  --output-dir "$run_dir/final_id" \
  --seed-pool final_id_test --ood id \
  --mlp-checkpoint "$HOME/minimind_mouse_data/formal_stage/20260822_p1/mlp_bc.pt" \
  --risk-threshold 0.9 --planner-horizon 8 --p1-planner-horizon 4 \
  --base-weight "$HOME/MouseMind/out/full_sft_768.pth" \
  --skill-lora-weight "$weight" --tokenizer "$repo_root/model" --device cuda \
  --diagnostic-trace "$run_dir/final_id/decision_trace_private.jsonl"

echo "VERIFIED_DPO_RESULT=$run_dir/final_id/closed_loop_metrics.json"
