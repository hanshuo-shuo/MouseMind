#!/usr/bin/env bash
#SBATCH --job-name=mm-align-extended
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
run_root="$p2_root/verified_alignment_extended"
sft_weight="$p2_root/verified_dpo/seed_clean_sft/minimind/lora_mousemind_skill_planner_768.pth"
base_weight="$repo_root/out/full_sft_768.pth"
mlp_weight="$HOME/minimind_mouse_data/formal_stage/20260822_p1/mlp_bc.pt"
mkdir -p "$run_root"
export CELLWORLD_CACHE="${CELLWORLD_CACHE:-$HOME/minimind_mouse_data/cellworld_cache}"
export PYTHONPATH="${MOUSE_LLM_RUNTIME_ROOT:-$HOME/minimind_mouse_data/runtime}:$repo_root:${PYTHONPATH:-}"
export PYTHONDONTWRITEBYTECODE=1 TOKENIZERS_PARALLELISM=false
cd "$repo_root"

# One predeclared longer DPO run; no final-test-based parameter selection.
python -m mouse_llm.training.train_verified_dpo \
  --train-data "$p2_root/verified_dpo/data/dpo_train.jsonl" \
  --base-weight "$base_weight" --sft-lora-weight "$sft_weight" \
  --tokenizer "$repo_root/model" --output-dir "$run_root/dpo" \
  --beta 0.1 --epochs 3 --learning-rate 5e-6

# One predeclared longer GRPO run on the same seed-isolated collection anchors.
python -m mouse_llm.training.train_verified_grpo \
  --anchors "$p2_root/data/p2-v1-collection/anchors_private.jsonl" \
  --verified-counterfactuals "$p2_root/data/p2-v1-collection/dataset/counterfactuals_private.jsonl" \
  --base-weight "$base_weight" --sft-lora-weight "$sft_weight" \
  --tokenizer "$repo_root/model" --mlp-checkpoint "$mlp_weight" \
  --output-dir "$run_root/grpo" --device cuda \
  --group-size 6 --temperature 1.0 --rollout-horizon 8 \
  --clip-epsilon 0.2 --kl-coefficient 0.2 --target-kl 0.05 \
  --policy-update-epochs 2 --learning-rate 1e-6 --epochs 2

for method in dpo grpo; do
  if [[ "$method" == dpo ]]; then
    weight="$run_root/dpo/lora_mousemind_verified_dpo_768.pth"
  else
    weight="$run_root/grpo/lora_mousemind_verified_grpo_768.pth"
  fi
  test -s "$weight"
  python -m mouse_llm.evaluation.p2_benchmark \
    --policies minimind-learned --reference-policy minimind-learned \
    --output-dir "$run_root/$method/development" \
    --seed-pool development --ood id \
    --mlp-checkpoint "$mlp_weight" \
    --risk-threshold 0.9 --planner-horizon 8 --p1-planner-horizon 4 \
    --base-weight "$base_weight" --skill-lora-weight "$weight" \
    --tokenizer "$repo_root/model" --device cuda
  echo "EXTENDED_DEVELOPMENT_RESULT=$method:$run_root/$method/development/closed_loop_metrics.json"
done

python -m mouse_llm.evaluation.compare_verified_alignment \
  --baseline "$p2_root/verified_alignment_development/seed_clean_sft" \
  --candidate extended_dpo="$run_root/dpo/development" \
  --candidate extended_grpo="$run_root/grpo/development" \
  --output "$run_root/paired_development.json"
