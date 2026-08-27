#!/usr/bin/env bash
#SBATCH --job-name=mm-p2-ood
#SBATCH --partition=job
#SBATCH --array=0-3
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=18:00:00
#SBATCH --output=mouse-p2-%x-%A_%a.out
#SBATCH --error=mouse-p2-%x-%A_%a.err

set -euo pipefail
conditions=(faster_predator_020 faster_predator_025 shorter_los_070 unseen_language)
export MOUSE_P2_OOD_CONDITION="${conditions[$SLURM_ARRAY_TASK_ID]}"
export MOUSE_P2_SEED_POOL=final_id_test
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
bash "$script_dir/closed_loop.sbatch"
