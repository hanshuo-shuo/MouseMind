#!/usr/bin/env bash
#SBATCH --job-name=mm-continuous
#SBATCH --partition=job
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=mouse-continuous-%j.out
#SBATCH --error=mouse-continuous-%j.err

# Uses the project's existing BCS516 partition convention. Submit only after
# copying the local candidate to a separate cluster checkout; this script has
# no git/network/publish operations. Outputs must be outside any Git checkout.
set -euo pipefail
umask 077
export PYTHONDONTWRITEBYTECODE=1
export MPLBACKEND=Agg
export PYGAME_HIDE_SUPPORT_PROMPT=1
export SDL_VIDEODRIVER=dummy
export SDL_AUDIODRIVER=dummy

cd "${MOUSE_XY_REPO_ROOT:?set the isolated local-candidate checkout on the cluster}"
xy_python="${MOUSE_XY_PYTHON:?set the project environment Python executable}"
xy_root="${MOUSE_XY_ROOT:?set a new private output root outside Git}"
xy_stage="${MOUSE_XY_STAGE:?choose bc, collect, labels, sft, dpo, or dev}"
xy_labels="${MOUSE_XY_LABEL_MODE:-next_position}"
xy_specialist="${MOUSE_XY_SPECIALIST:-$xy_root/bc-$xy_labels/continuous_specialist.pt}"

case "$xy_stage" in
  bc)
    "$xy_python" -m mouse_llm.continuous.specialist \
      --train-data "$xy_root/$xy_labels/train.jsonl" \
      --validation-data "$xy_root/$xy_labels/validation.jsonl" \
      --output-dir "$xy_root/bc-$xy_labels" --device cuda --components 5 ;;
  collect)
    "$xy_python" -m mouse_llm.continuous.evaluation --policy rule \
      --specialist-checkpoint "$xy_specialist" --seed-pool data_collection \
      --trace --output-dir "$xy_root/collection" --device cuda ;;
  labels)
    "$xy_python" -m mouse_llm.continuous.posttraining \
      --trace "$xy_root/collection/trace_private.jsonl" \
      --specialist-checkpoint "$xy_specialist" --output-dir "$xy_root/planner-data" \
      --candidate-steps 8 --horizon 32 --device cuda ;;
  sft)
    "$xy_python" -m mouse_llm.continuous.posttrain sft \
      --data-dir "$xy_root/planner-data" --specialist-checkpoint "$xy_specialist" \
      --base-weight "${MOUSE_XY_BASE:?set the base checkpoint}" --tokenizer model \
      --output-dir "$xy_root/planner-sft" --epochs 3 --device cuda ;;
  dpo)
    "$xy_python" -m mouse_llm.continuous.posttrain dpo \
      --data-dir "$xy_root/planner-data" --specialist-checkpoint "$xy_specialist" \
      --base-weight "${MOUSE_XY_BASE:?set the base checkpoint}" --tokenizer model \
      --sft-lora-weight "$xy_root/planner-sft/skill_lora.pth" \
      --sft-manifest "$xy_root/planner-sft/planner_manifest.json" \
      --output-dir "$xy_root/planner-dpo" --epochs 1 --beta 0.1 --device cuda ;;
  dev)
    "$xy_python" -m mouse_llm.continuous.evaluation --policy minimind \
      --specialist-checkpoint "$xy_specialist" --base-weight "${MOUSE_XY_BASE:?set the base checkpoint}" \
      --skill-lora-weight "$xy_root/planner-dpo/skill_lora.pth" \
      --planner-manifest "$xy_root/planner-dpo/planner_manifest.json" \
      --tokenizer model --seed-pool development --output-dir "$xy_root/dev-dpo" --device cuda ;;
  *) echo "Unknown MOUSE_XY_STAGE: $xy_stage" >&2; exit 2 ;;
esac
