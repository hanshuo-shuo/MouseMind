# MouseMind P2 Northwestern runbook

All private states, datasets, checkpoints, per-episode rows, and logs must live
outside Git. Set one private root before submitting jobs:

```bash
export MOUSE_LLM_REPO_ROOT="$HOME/minimind"
export MOUSE_P2_ROOT="/shares/bcs516/sh/minimind_mouse_p2"
export MOUSE_P2_OUTPUT_ROOT="$MOUSE_P2_ROOT/closed_loop"
export MOUSE_P2_MODEL_DIR="$MOUSE_P2_ROOT/models"
export MOUSE_LLM_MLP_CHECKPOINT="$HOME/minimind_mouse_data/formal_stage/20260822_p1/mlp_bc.pt"
export MOUSE_LLM_BASE_WEIGHT="$HOME/minimind/out/full_sft_768.pth"
export CELLWORLD_CACHE="$HOME/minimind_mouse_data/cellworld_cache"
```

Run in dependency order. Record every returned job ID.

```bash
sbatch --export=ALL mouse_llm/northwestern/p2/collect_counterfactual.sbatch

export MOUSE_P2_DATASET_DIR="$MOUSE_P2_ROOT/data/<completed-run>/dataset"
sbatch --export=ALL mouse_llm/northwestern/p2/train_numeric_and_risk.sbatch
sbatch --export=ALL mouse_llm/northwestern/p2/train_skill_planner.sbatch

export MOUSE_P2_PLANNER_CHECKPOINT="$MOUSE_P2_MODEL_DIR/numeric/planner_mlp.pt"
export MOUSE_P2_RISK_CHECKPOINT="$MOUSE_P2_MODEL_DIR/risk/risk_critic.pt"
export MOUSE_P2_SKILL_LORA_WEIGHT="$MOUSE_P2_MODEL_DIR/minimind/lora_mousemind_skill_planner_768.pth"
sbatch --export=ALL mouse_llm/northwestern/p2/closed_loop.sbatch
sbatch --export=ALL mouse_llm/northwestern/p2/ood_closed_loop.sbatch
```

Select planner horizon and verifier threshold only on the complete development
pool. Freeze both values before running `final_id_test`. Do not retrain or filter
from final-test failures. For P2.1, mine only development failures, rebuild once,
retrain once, compare on development, and then select the final system.

For episode shards, merge with `mouse_llm.evaluation.merge_p2_shards`; it rejects
duplicate, missing, unexpected, or config-mismatched seeds. Run main evidence
unsharded when exact per-action latency percentiles are required.
