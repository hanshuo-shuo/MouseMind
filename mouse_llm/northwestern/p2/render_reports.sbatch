#!/usr/bin/env bash
#SBATCH --job-name=mm-p2-report
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=mouse-p2-%x-%j.out
#SBATCH --error=mouse-p2-%x-%j.err

set -euo pipefail
source "$HOME/anaconda3/etc/profile.d/conda.sh"
conda activate /shares/bcs516/sh/minimind_env
repo_root="${MOUSE_LLM_REPO_ROOT:-$HOME/minimind}"
figure_dir="${MOUSE_P2_FIGURE_DIR:?set MOUSE_P2_FIGURE_DIR}"
cd "$repo_root"
python -m mouse_llm.evaluation.plot_p2 \
  --id-report "${MOUSE_P2_ID_REPORT:?set MOUSE_P2_ID_REPORT}" \
  --ood-report "faster_020=${MOUSE_P2_OOD_020_REPORT:?set report}" \
  --ood-report "faster_025=${MOUSE_P2_OOD_025_REPORT:?set report}" \
  --ood-report "shorter_los=${MOUSE_P2_OOD_LOS_REPORT:?set report}" \
  --ood-report "unseen_language=${MOUSE_P2_OOD_LANGUAGE_REPORT:?set report}" \
  --output-dir "$figure_dir"
python -m mouse_llm.privacy_guard --repo-root "$repo_root"
