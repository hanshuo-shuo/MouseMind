#!/usr/bin/env bash
#SBATCH --job-name=mm-dpo-clean-sft
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
data_dir="$p2_root/verified_dpo/data"
run_dir="$p2_root/verified_dpo/seed_clean_sft"
mkdir -p "$run_dir/minimind"
export CELLWORLD_CACHE="${CELLWORLD_CACHE:-$HOME/minimind_mouse_data/cellworld_cache}"
export PYTHONPATH="${MOUSE_LLM_RUNTIME_ROOT:-$HOME/minimind_mouse_data/runtime}:$repo_root:${PYTHONPATH:-}"
export PYTHONDONTWRITEBYTECODE=1 TOKENIZERS_PARALLELISM=false
cd "$repo_root"

python -m mouse_llm.training.train_skill_planner \
  --repo-root "$repo_root" --data-path "$data_dir/dpo_train.jsonl" \
  --save-dir "$run_dir/minimind" --epochs 3 --batch-size 16 \
  --max-seq-len 1024 --device cuda:0

weight="$run_dir/minimind/lora_mousemind_skill_planner_768.pth"
python -m mouse_llm.evaluation.evaluate_skill_planner \
  --base-weight "$repo_root/out/full_sft_768.pth" \
  --lora-weight "$weight" --tokenizer "$repo_root/model" \
  --seen-test-data "$data_dir/dpo_seen_test.jsonl" \
  --unseen-test-data "$data_dir/dpo_unseen_test.jsonl" \
  --output "$run_dir/offline_metrics.json" \
  --device cuda --research-evidence --only-lora --no-ablations

python -m mouse_llm.evaluation.p2_benchmark \
  --policies minimind-learned --reference-policy minimind-learned \
  --output-dir "$run_dir/final_id" \
  --seed-pool final_id_test --ood id \
  --mlp-checkpoint "$HOME/minimind_mouse_data/formal_stage/20260822_p1/mlp_bc.pt" \
  --risk-threshold 0.9 --planner-horizon 8 --p1-planner-horizon 4 \
  --base-weight "$repo_root/out/full_sft_768.pth" \
  --skill-lora-weight "$weight" --tokenizer "$repo_root/model" --device cuda

echo "SEED_CLEAN_SFT_RESULT=$run_dir/final_id/closed_loop_metrics.json"
