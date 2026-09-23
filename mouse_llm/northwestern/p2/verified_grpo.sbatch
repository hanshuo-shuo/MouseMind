#!/usr/bin/env bash
#SBATCH --job-name=mm-verified-grpo
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
run_dir="$p2_root/verified_grpo/main"
mkdir -p "$run_dir"
export CELLWORLD_CACHE="${CELLWORLD_CACHE:-$HOME/minimind_mouse_data/cellworld_cache}"
export PYTHONPATH="${MOUSE_LLM_RUNTIME_ROOT:-$HOME/minimind_mouse_data/runtime}:$repo_root:${PYTHONPATH:-}"
export PYTHONDONTWRITEBYTECODE=1 TOKENIZERS_PARALLELISM=false
cd "$repo_root"

# Supply the same raw and verified collection anchors used to build DPO data.
python -m mouse_llm.training.train_verified_grpo \
  --anchors "${MOUSE_GRPO_ANCHORS:?set raw collection anchors JSONL}" \
  --verified-counterfactuals "${MOUSE_GRPO_VERIFIED:?set verified counterfactuals JSONL}" \
  --base-weight "$repo_root/out/full_sft_768.pth" \
  --sft-lora-weight "${MOUSE_GRPO_SFT_LORA_WEIGHT:?set seed-clean SFT LoRA}" \
  --tokenizer "$repo_root/model" \
  --mlp-checkpoint "$HOME/minimind_mouse_data/formal_stage/20260822_p1/mlp_bc.pt" \
  --output-dir "$run_dir" --device cuda \
  --group-size 6 --temperature 1.0 --rollout-horizon 8 \
  --clip-epsilon 0.2 --kl-coefficient 0.01 --learning-rate 2e-6

weight="$run_dir/lora_mousemind_verified_grpo_768.pth"
python -m mouse_llm.evaluation.evaluate_skill_planner \
  --base-weight "$repo_root/out/full_sft_768.pth" \
  --lora-weight "$weight" --tokenizer "$repo_root/model" \
  --seen-test-data "$p2_root/verified_dpo/data/dpo_seen_test.jsonl" \
  --unseen-test-data "$p2_root/verified_dpo/data/dpo_unseen_test.jsonl" \
  --output "$run_dir/offline_metrics.json" \
  --device cuda --research-evidence --only-lora --no-ablations

# Frozen final ID protocol: identical to SFT and Verified DPO evaluation.
python -m mouse_llm.evaluation.p2_benchmark \
  --policies minimind-learned --reference-policy minimind-learned \
  --output-dir "$run_dir/final_id" \
  --seed-pool final_id_test --ood id \
  --mlp-checkpoint "$HOME/minimind_mouse_data/formal_stage/20260822_p1/mlp_bc.pt" \
  --risk-threshold 0.9 --planner-horizon 8 --p1-planner-horizon 4 \
  --base-weight "$repo_root/out/full_sft_768.pth" \
  --skill-lora-weight "$weight" --tokenizer "$repo_root/model" --device cuda

echo "VERIFIED_GRPO_RESULT=$run_dir/final_id/closed_loop_metrics.json"
