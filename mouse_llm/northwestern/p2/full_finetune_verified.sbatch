#!/usr/bin/env bash
#SBATCH --job-name=mm-full-verified
#SBATCH --partition=job
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=18:00:00
#SBATCH --output=mouse-full-%x-%j.out
#SBATCH --error=mouse-full-%x-%j.err
set -euo pipefail
umask 077
source "$HOME/anaconda3/etc/profile.d/conda.sh"
conda activate /shares/bcs516/sh/minimind_env
repo_root="${MOUSE_LLM_REPO_ROOT:-$HOME/MouseMind}"
p2_root="${MOUSE_P2_ROOT:-/shares/bcs516/sh/minimind_mouse_p2}"
run_root="$p2_root/full_verified_v1"
data_dir="$p2_root/verified_dpo/data"
mlp_weight="$HOME/minimind_mouse_data/formal_stage/20260822_p1/mlp_bc.pt"
base_weight="$repo_root/out/full_sft_768.pth"
sft_weight="$run_root/sft/full_skill_planner_768.pth"
method="${FULL_METHOD:?set FULL_METHOD to sft, dpo, grpo, or final}"
export CELLWORLD_CACHE="${CELLWORLD_CACHE:-$HOME/minimind_mouse_data/cellworld_cache}"
export PYTHONPATH="${MOUSE_LLM_RUNTIME_ROOT:-$HOME/minimind_mouse_data/runtime}:$repo_root:${PYTHONPATH:-}"
export PYTHONDONTWRITEBYTECODE=1 TOKENIZERS_PARALLELISM=false
mkdir -p "$run_root/$method"
cd "$repo_root"

case "$method" in
  sft)
    python -m mouse_llm.training.train_full_skill_planner \
      --data-path "$data_dir/dpo_train.jsonl" --base-weight "$base_weight" \
      --tokenizer "$repo_root/model" --output-dir "$run_root/sft" \
      --epochs 3 --batch-size 2 --gradient-accumulation 8 --learning-rate 1e-5
    weight="$sft_weight"
    ;;
  dpo)
    test -s "$sft_weight"
    python -m mouse_llm.training.train_verified_dpo_full \
      --train-data "$data_dir/dpo_train.jsonl" --sft-full-weight "$sft_weight" \
      --tokenizer "$repo_root/model" --output-dir "$run_root/dpo" \
      --beta 0.1 --epochs 1 --learning-rate 1e-6
    weight="$run_root/dpo/full_verified_dpo_768.pth"
    ;;
  grpo)
    test -s "$sft_weight"
    python -m mouse_llm.training.train_verified_grpo_full \
      --anchors "$p2_root/data/p2-v1-collection/anchors_private.jsonl" \
      --verified-counterfactuals "$p2_root/data/p2-v1-collection/dataset/counterfactuals_private.jsonl" \
      --sft-full-weight "$sft_weight" --tokenizer "$repo_root/model" \
      --mlp-checkpoint "$mlp_weight" --output-dir "$run_root/grpo" --device cuda \
      --group-size 6 --temperature 1.0 --rollout-horizon 8 \
      --clip-epsilon 0.2 --kl-coefficient 0.01 --learning-rate 2e-7
    weight="$run_root/grpo/full_verified_grpo_768.pth"
    ;;
  final)
    for candidate in sft dpo grpo; do
      case "$candidate" in
        sft) weight="$sft_weight" ;;
        dpo) weight="$run_root/dpo/full_verified_dpo_768.pth" ;;
        grpo) weight="$run_root/grpo/full_verified_grpo_768.pth" ;;
      esac
      test -s "$weight"
      python -m mouse_llm.evaluation.p2_benchmark \
        --policies minimind-learned --reference-policy minimind-learned \
        --output-dir "$run_root/$candidate/final_id" --seed-pool final_id_test --ood id \
        --mlp-checkpoint "$mlp_weight" --risk-threshold 0.9 \
        --planner-horizon 8 --p1-planner-horizon 4 \
        --skill-full-weight "$weight" --tokenizer "$repo_root/model" --device cuda
      echo "FULL_FINAL_RESULT=$candidate:$run_root/$candidate/final_id/closed_loop_metrics.json"
    done
    exit 0
    ;;
  *) echo "Unknown FULL_METHOD=$method" >&2; exit 2 ;;
esac

test -s "$weight"
python -m mouse_llm.evaluation.evaluate_skill_planner \
  --base-weight "$base_weight" --full-weight "$weight" --tokenizer "$repo_root/model" \
  --seen-test-data "$data_dir/dpo_seen_test.jsonl" \
  --unseen-test-data "$data_dir/dpo_unseen_test.jsonl" \
  --output "$run_root/$method/offline_metrics.json" \
  --device cuda --research-evidence --no-ablations
python -m mouse_llm.evaluation.p2_benchmark \
  --policies minimind-learned --reference-policy minimind-learned \
  --output-dir "$run_root/$method/development" --seed-pool development --ood id \
  --mlp-checkpoint "$mlp_weight" --risk-threshold 0.9 \
  --planner-horizon 8 --p1-planner-horizon 4 \
  --skill-full-weight "$weight" --tokenizer "$repo_root/model" --device cuda
echo "FULL_DEVELOPMENT_RESULT=$method:$run_root/$method/development/closed_loop_metrics.json"
