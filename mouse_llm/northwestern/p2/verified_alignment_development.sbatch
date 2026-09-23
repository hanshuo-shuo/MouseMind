#!/usr/bin/env bash
#SBATCH --job-name=mm-align-dev
#SBATCH --partition=job
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=12:00:00
#SBATCH --output=mouse-p2-%x-%j.out
#SBATCH --error=mouse-p2-%x-%j.err

set -euo pipefail
umask 077
source "$HOME/anaconda3/etc/profile.d/conda.sh"
conda activate /shares/bcs516/sh/minimind_env
repo_root="${MOUSE_LLM_REPO_ROOT:-$HOME/MouseMind}"
p2_root="${MOUSE_P2_ROOT:-/shares/bcs516/sh/minimind_mouse_p2}"
output_root="$p2_root/verified_alignment_development"
mkdir -p "$output_root"
export CELLWORLD_CACHE="${CELLWORLD_CACHE:-$HOME/minimind_mouse_data/cellworld_cache}"
export PYTHONPATH="${MOUSE_LLM_RUNTIME_ROOT:-$HOME/minimind_mouse_data/runtime}:$repo_root:${PYTHONPATH:-}"
export PYTHONDONTWRITEBYTECODE=1 TOKENIZERS_PARALLELISM=false
cd "$repo_root"

names=(seed_clean_sft dpo_beta_0.05 dpo_beta_0.1 dpo_beta_0.2 grpo_main)
weights=(
  "$p2_root/verified_dpo/seed_clean_sft/minimind/lora_mousemind_skill_planner_768.pth"
  "$p2_root/verified_dpo/beta_0.05/lora_mousemind_verified_dpo_768.pth"
  "$p2_root/verified_dpo/beta_0.1/lora_mousemind_verified_dpo_768.pth"
  "$p2_root/verified_dpo/beta_0.2/lora_mousemind_verified_dpo_768.pth"
  "$p2_root/verified_grpo/main/lora_mousemind_verified_grpo_768.pth"
)
for index in "${!names[@]}"; do
  name="${names[$index]}"
  weight="${weights[$index]}"
  test -s "$weight"
  python -m mouse_llm.evaluation.p2_benchmark \
    --policies minimind-learned --reference-policy minimind-learned \
    --output-dir "$output_root/$name" \
    --seed-pool development --ood id \
    --mlp-checkpoint "$HOME/minimind_mouse_data/formal_stage/20260822_p1/mlp_bc.pt" \
    --risk-threshold 0.9 --planner-horizon 8 --p1-planner-horizon 4 \
    --base-weight "$repo_root/out/full_sft_768.pth" \
    --skill-lora-weight "$weight" --tokenizer "$repo_root/model" --device cuda
  echo "DEVELOPMENT_RESULT=$name:$output_root/$name/closed_loop_metrics.json"
done

python - "$output_root" "${names[@]}" <<'PY'
import json
import sys
from pathlib import Path
root = Path(sys.argv[1])
summary = {}
for name in sys.argv[2:]:
    data = json.loads((root / name / 'closed_loop_metrics.json').read_text())
    metrics = data['policies']['minimind-learned']
    if metrics['episode_count'] != 40:
        raise RuntimeError(f'{name}: expected all 40 development seeds')
    summary[name] = {
        'task_success_rate': metrics['success_rate']['mean'],
        'clean_success_rate': metrics['clean_success_rate']['mean'],
        'capture_rate': metrics['capture_rate']['mean'],
        'captures_per_episode': metrics['captures_per_episode']['mean'],
    }
(root / 'summary.json').write_text(json.dumps(summary, indent=2, sort_keys=True) + '\n')
print(json.dumps(summary, sort_keys=True), flush=True)
PY
