# Frozen BotEvade-to-Oasis transfer runbook

This study has two distinct questions:

1. **Literal full-stack transfer:** reuse the frozen BotEvade planner and 10D
   low-level specialist unchanged. The compatibility contract marks this mode
   insufficient for arbitrary Oasis goals because the specialist never sees
   active goal coordinates.
2. **Planner-isolation transfer:** freeze the high-level planner and pair every
   planner with the same parameter-free active-goal controller. This isolates
   strategic transfer from the known low-level interface mismatch.

All checkpoints, episode rows, logs, and intermediate reports stay outside Git.
The public compatibility audit uses no private data.

```bash
export MOUSE_TRANSFER_REPO_ROOT="$HOME/mousemind-transfer-study"
export MOUSE_TRANSFER_OUTPUT_ROOT="/shares/bcs516/sh/minimind_mouse_transfer"
export MOUSE_TRANSFER_COMPATIBILITY_AUDIT="$HOME/minimind_mouse_data/transfer-study/compatibility/audit.json"
export CELLWORLD_CACHE="$HOME/minimind_mouse_data/cellworld_cache"
```

Run the compatibility audit before any benchmark:

```bash
python -m mouse_llm.evaluation.audit_transfer_compatibility \
  --output "$MOUSE_TRANSFER_COMPATIBILITY_AUDIT"
```

Run the frozen development pool without selecting new parameters:

```bash
export MOUSE_TRANSFER_SEED_POOL=development
export MOUSE_TRANSFER_CONDITION=seen_instruction
export MOUSE_TRANSFER_RUN_ID=frozen-development-v1
sbatch --export=ALL mouse_llm/northwestern/transfer/run_benchmark.sbatch
```

After the development report passes structural checks, run the untouched final
pool once:

```bash
export MOUSE_TRANSFER_SEED_POOL=final_test
export MOUSE_TRANSFER_CONDITION=seen_instruction
export MOUSE_TRANSFER_RUN_ID=frozen-final-seen-v1
sbatch --export=ALL mouse_llm/northwestern/transfer/run_benchmark.sbatch
```

Run the language ablations on the same final seeds using frozen unseen
instruction templates:

```bash
export MOUSE_TRANSFER_CONDITION=unseen_instruction
export MOUSE_TRANSFER_RUN_ID=frozen-final-unseen-v1
export MOUSE_TRANSFER_POLICIES="aligned-goal-only aligned-minimind aligned-minimind-no-history aligned-minimind-no-instruction"
sbatch --export=ALL mouse_llm/northwestern/transfer/run_benchmark.sbatch
```

Do not tune horizons, goal projections, thresholds, policy membership, or
training data from final-test outcomes. A failed transfer is a valid result.
