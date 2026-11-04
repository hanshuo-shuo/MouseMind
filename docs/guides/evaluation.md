# Run and evaluate MouseMind

Run commands from the repository root. Private model checkpoints, trajectories, and rollout records stay outside Git.

## Quick start

Install the project and environment dependencies from a checkout, then run the public synthetic demo:

```bash
python -m pip install -e '.[env]'
python demo.py
```

The demo exercises the evaluation path and action catalog; it does not reproduce the research results. Full BotEvade evaluation requires separately stored model checkpoints and the Cellworld cache. Set `CELLWORLD_CACHE`, `MOUSE_XY_SPECIALIST` (commanded-destination continuous BC), `MOUSE_LLM_BASE_WEIGHT`, `MOUSE_P2_SKILL_LORA_WEIGHT` (verified DPO β=0.1), and a private `MOUSE_XY_OUTPUT_ROOT` before running:

```bash
python -m mouse_llm.continuous.evaluation \
  --policy legacy-minimind-continuous-bc \
  --specialist-checkpoint "$MOUSE_XY_SPECIALIST" \
  --base-weight "$MOUSE_LLM_BASE_WEIGHT" \
  --skill-lora-weight "$MOUSE_P2_SKILL_LORA_WEIGHT" \
  --tokenizer model --device cuda --reuse-environment \
  --contract mouse_llm/continuous/experiments/legacy_hybrid_final_v1.json \
  --seed-pool development --world 21_05 --max-steps 300 \
  --time-step 0.25 --predator-speed-ratio 0.15 --max-line-of-sight-distance 1.0 \
  --destination-mode project --projection-epsilon 0.01 \
  --planner-horizon 8 --memory-steps 8 --evade-distance 0.35 \
  --preference survival_first --instruction-split train --planner-decoding greedy_trie \
  --action-catalog mouse_llm/envs/mice/assets/action_catalog_21_05.json \
  --output-dir "$MOUSE_XY_OUTPUT_ROOT/development-hybrid"
```

This command uses development seeds for checking the setup. The published final results use the registered 200-seed `final_id_test` pool; those outcomes are already known and must not be reused for candidate selection. The [frozen contract](../../mouse_llm/continuous/experiments/legacy_hybrid_final_v1.json) preserves the actual 0.01 projection inset used in the completed comparison. The general experimental default is different, so use this explicit contract when reproducing this configuration.

See the [continuous workflow](continuous-workflow.md) for training and evaluation entry points, and the [P2 runbook](p2-cluster.md) for the historical planner benchmark. Private trajectories, per-episode records, and weights are not included in Git.
