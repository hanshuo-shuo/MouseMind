#!/usr/bin/env bash
#SBATCH --job-name=mm-p2-correct
#SBATCH --partition=job
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=mouse-p2-%x-%j.out
#SBATCH --error=mouse-p2-%x-%j.err

set -euo pipefail
umask 077
source "$HOME/anaconda3/etc/profile.d/conda.sh"
conda activate /shares/bcs516/sh/minimind_env

repo_root="${MOUSE_LLM_REPO_ROOT:-$HOME/minimind}"
p2_root="${MOUSE_P2_ROOT:?set MOUSE_P2_ROOT}"
base_dataset="${MOUSE_P2_DATASET_DIR:?set MOUSE_P2_DATASET_DIR}"
base_models="${MOUSE_P2_MODEL_DIR:?set MOUSE_P2_MODEL_DIR}"
mlp_checkpoint="${MOUSE_LLM_MLP_CHECKPOINT:?set MOUSE_LLM_MLP_CHECKPOINT}"
cellworld_cache="${CELLWORLD_CACHE:?set CELLWORLD_CACHE}"
horizon="${MOUSE_P2_SELECTED_HORIZON:-8}"
source_run="${MOUSE_P2_SOURCE_DEV_RUN:?set selected development run directory}"
run_dir="$p2_root/p2_1"
corrective_dataset="$run_dir/counterfactual_dataset"
merged_dataset="$run_dir/merged_dataset"
model_dir="$run_dir/models"
dev_output="$run_dir/development"
mkdir -p "$run_dir" "$model_dir/numeric" "$model_dir/risk" "$dev_output"

export CELLWORLD_CACHE="$cellworld_cache"
export PYTHONPATH="${MOUSE_LLM_RUNTIME_ROOT:-$HOME/minimind_mouse_data/runtime}:$repo_root:${PYTHONPATH:-}"
export PYTHONDONTWRITEBYTECODE=1
cd "$repo_root"

python -m mouse_llm.evaluation.mine_failures \
  --episodes "$source_run/closed_loop_episodes.csv" \
  --metrics "$source_run/closed_loop_metrics.json" \
  --policy numeric-learned \
  --per-mode "${MOUSE_P2_CORRECTIVE_PER_MODE:-5}" \
  --output "$run_dir/failure_manifest_private.json"

python -m mouse_llm.data.collect_corrective_anchors \
  --failure-manifest "$run_dir/failure_manifest_private.json" \
  --output "$run_dir/anchors_private.jsonl" \
  --aggregate-output "$run_dir/anchor_aggregate.json" \
  --mlp-checkpoint "$mlp_checkpoint" \
  --planner-checkpoint "$base_models/numeric/planner_mlp.pt" \
  --planner-horizon "$horizon" \
  --max-anchors-per-seed "${MOUSE_P2_CORRECTIVE_ANCHORS_PER_SEED:-4}" \
  --device cpu

python -m mouse_llm.data.build_counterfactual_skill_dataset \
  --anchors "$run_dir/anchors_private.jsonl" \
  --output-dir "$corrective_dataset" \
  --aggregate-output "$run_dir/counterfactual_aggregate.json" \
  --mlp-checkpoint "$mlp_checkpoint" \
  --horizons 8 \
  --label-horizon 8 \
  --source-pool development_corrective \
  --research-evidence \
  --device cpu

python -m mouse_llm.data.merge_corrective_dataset \
  --base-dir "$base_dataset" \
  --corrective-dir "$corrective_dataset" \
  --output-dir "$merged_dataset" \
  --corrective-weight "${MOUSE_P2_CORRECTIVE_WEIGHT:-3}"

python -m mouse_llm.baselines.planner_mlp \
  --train-data "$merged_dataset/planner_train.jsonl" \
  --validation-data "$merged_dataset/planner_validation.jsonl" \
  --seen-test-data "$merged_dataset/planner_seen_test.jsonl" \
  --unseen-test-data "$merged_dataset/planner_unseen_test.jsonl" \
  --output-dir "$model_dir/numeric" \
  --device cpu \
  --research-evidence

python -m mouse_llm.training.train_risk_critic \
  --train-data "$merged_dataset/risk_train.jsonl" \
  --validation-data "$merged_dataset/risk_validation.jsonl" \
  --output-dir "$model_dir/risk" \
  --device cpu \
  --research-evidence

python -m mouse_llm.evaluation.p2_benchmark \
  --policies numeric-learned \
  --reference-policy numeric-learned \
  --output-dir "$dev_output" \
  --seed-pool development \
  --ood id \
  --mlp-checkpoint "$mlp_checkpoint" \
  --planner-checkpoint "$model_dir/numeric/planner_mlp.pt" \
  --planner-horizon "$horizon" \
  --device cpu

echo "P2_1_COUNTERFACTUAL_AGGREGATE=$run_dir/counterfactual_aggregate.json"
echo "P2_1_NUMERIC_METRICS=$model_dir/numeric/planner_mlp_metrics.json"
echo "P2_1_RISK_METRICS=$model_dir/risk/risk_critic_metrics.json"
echo "P2_1_DEVELOPMENT=$dev_output/closed_loop_metrics.json"
