# Continuous MouseMind local-candidate workflow

Keep all data, checkpoints, traces and episode rows outside Git. No command in
this workflow pushes code. Use a new run directory for every changed executor,
label mode, controller contract or model. Commands below are experiment entry
points, not a claim that training or evaluation has completed.

## Environment

Use the existing project environment; install project dependencies there:

```bash
python -m pip install -e '.[env,test,rl]'
export PYTHONDONTWRITEBYTECODE=1
export MPLBACKEND=Agg
export PYGAME_HIDE_SUPPORT_PROMPT=1
export SDL_VIDEODRIVER=dummy
export SDL_AUDIODRIVER=dummy
export MOUSE_XY_ROOT=/private/path/to/mousemind-continuous-v1
export MOUSE_XY_SOURCE=/private/path/to/mouse_data_processed.csv
export CELLWORLD_CACHE=/private/path/to/cellworld_cache
export MOUSE_XY_BASE=/private/path/to/full_sft_768.pth
```

The RL extra pins SB3 2.3.2 to match the existing Gymnasium 0.28.1 environment.
The local implementation checks can run without Transformers/SB3; actual LLM
and SAC training require their dependencies. Do not install a second Torch copy
into a production cluster environment without checking its CUDA build.

## Prepare both continuous label controls

```bash
python -m mouse_llm.continuous.data --input "$MOUSE_XY_SOURCE" \
  --output-dir "$MOUSE_XY_ROOT/next_position" --label-mode next_position \
  --max-train-samples 60000 --max-validation-samples 5000 --max-test-samples 5000
python -m mouse_llm.continuous.data --input "$MOUSE_XY_SOURCE" \
  --output-dir "$MOUSE_XY_ROOT/commanded_destination" --label-mode commanded_destination \
  --max-train-samples 60000 --max-validation-samples 5000 --max-test-samples 5000
```

Next-position labels are achieved waypoints, not original commands. Never use
their retained historical reward/next_obs as an offline RL replay transition.
The data manifest audits both label meanings, short-step displacement and
episode/input split isolation. Raw rows are private.

## Train and qualify the specialist

```bash
python -m mouse_llm.continuous.specialist \
  --train-data "$MOUSE_XY_ROOT/next_position/train.jsonl" \
  --validation-data "$MOUSE_XY_ROOT/next_position/validation.jsonl" \
  --output-dir "$MOUSE_XY_ROOT/bc-next" --components 5 --device cuda
```

Repeat with commanded-destination labels under `bc-command`. Use `--components 1`
as a matched single-component control. Select label/model settings using the
development closed loop, not test XY error. Freeze the chosen checkpoint path:

```bash
export MOUSE_XY_SPECIALIST="$MOUSE_XY_ROOT/bc-next/continuous_specialist.pt"
python -m mouse_llm.continuous.evaluation --policy continuous-bc \
  --specialist-checkpoint "$MOUSE_XY_SPECIALIST" --seed-pool development \
  --output-dir "$MOUSE_XY_ROOT/dev-bc" --device cuda
python -m mouse_llm.continuous.evaluation --policy rule \
  --specialist-checkpoint "$MOUSE_XY_SPECIALIST" --seed-pool development \
  --output-dir "$MOUSE_XY_ROOT/dev-rule" --device cuda
```

Optional whole-policy SAC teacher, with supervised actor warmstart. The critic
and replay buffer do not ingest relabeled historical transitions:

```bash
python -m mouse_llm.continuous.rl_teacher train \
  --output-dir "$MOUSE_XY_ROOT/sac-teacher" --total-timesteps 100000 --seed 42 \
  --bc-checkpoint "$MOUSE_XY_SPECIALIST" \
  --warmstart-data "$MOUSE_XY_ROOT/next_position/train.jsonl" --warmstart-steps 1000
python -m mouse_llm.continuous.evaluation --policy sac-teacher \
  --teacher-checkpoint "$MOUSE_XY_ROOT/sac-teacher/sac_teacher.zip" \
  --seed-pool development --output-dir "$MOUSE_XY_ROOT/dev-teacher"
```

The old TD-MPC2 checkpoint is not silently load-compatible with this SAC class.
Its observation semantics, third wait component, reset and metrics require an
explicit adapter and a separate teacher qualification experiment.

## Collect and relabel with the frozen continuous executor

```bash
python -m mouse_llm.continuous.evaluation --policy rule \
  --specialist-checkpoint "$MOUSE_XY_SPECIALIST" --seed-pool data_collection \
  --trace --output-dir "$MOUSE_XY_ROOT/collection" --device cuda
python -m mouse_llm.continuous.posttraining \
  --trace "$MOUSE_XY_ROOT/collection/trace_private.jsonl" \
  --specialist-checkpoint "$MOUSE_XY_SPECIALIST" \
  --output-dir "$MOUSE_XY_ROOT/planner-data" --candidate-steps 8 --horizon 32 \
  --max-anchors 320 --continuation specialist --reuse-environments --device cuda
```

The collector can also be a qualified SAC teacher or a current continuous
MiniMind policy. Keep all collection seeds in `50000..50199`. Every candidate
branch is replay-verified and repeated; failed anchors remain excluded and
reported. Hidden fingerprints never enter prompts. Labels share one utility
ordering and one frozen-specialist continuation; they are not unrestricted oracles.
Use `--continuation rule` only for its explicit ablation. Pass `--contract`
with the frozen study JSON when using a nondefault projection inset; trace and
contract environment settings must agree.
Changing the specialist requires new labels and a new dataset directory.

## MiniMind post-training

```bash
python -m mouse_llm.continuous.posttrain sft \
  --data-dir "$MOUSE_XY_ROOT/planner-data" --specialist-checkpoint "$MOUSE_XY_SPECIALIST" \
  --base-weight "$MOUSE_XY_BASE" --tokenizer model --epochs 3 \
  --output-dir "$MOUSE_XY_ROOT/planner-sft" --device cuda
python -m mouse_llm.continuous.posttrain dpo \
  --data-dir "$MOUSE_XY_ROOT/planner-data" --specialist-checkpoint "$MOUSE_XY_SPECIALIST" \
  --base-weight "$MOUSE_XY_BASE" --tokenizer model \
  --sft-lora-weight "$MOUSE_XY_ROOT/planner-sft/skill_lora.pth" \
  --sft-manifest "$MOUSE_XY_ROOT/planner-sft/planner_manifest.json" \
  --output-dir "$MOUSE_XY_ROOT/planner-dpo" --epochs 1 --beta 0.1 --device cuda
```

Use `--validate-only` to verify dataset/executor/initialization contracts without
loading a model. Adapters bind base, LoRA, executor, data, context schema and
execution parameters. Full prompt+response length is checked before training.
An old skill LoRA is not labeled as a newly trained continuous planner.

Evaluate SFT and DPO on development with the same frozen executor:

```bash
python -m mouse_llm.continuous.evaluation --policy minimind \
  --specialist-checkpoint "$MOUSE_XY_SPECIALIST" --base-weight "$MOUSE_XY_BASE" \
  --skill-lora-weight "$MOUSE_XY_ROOT/planner-dpo/skill_lora.pth" \
  --planner-manifest "$MOUSE_XY_ROOT/planner-dpo/planner_manifest.json" \
  --tokenizer model --seed-pool development \
  --output-dir "$MOUSE_XY_ROOT/dev-dpo" --device cuda
```

Repeat with `planner-sft` and a separate output directory. Inspect capture,
goal progress, skill occupancy, invalid-destination and latency failures.
Use `--instruction-split unseen_test`, predator speed and LOS overrides only as
declared OOD conditions. The primary ID final contract stays fixed.

## Final paired comparison, then manual review

Only after freezing the candidate, run its evaluation with `--seed-pool
final_id_test` into `final-candidate`. Evaluate the historical complete hierarchy
on the same 200 seeds and the same versioned environment with `--policy
legacy-minimind`, its old specialist and skill LoRA, into `final-reference`.
This is an explicit catalog-constrained reference, not a new continuous model.

```bash
python -m mouse_llm.continuous.promotion \
  --reference-dir "$MOUSE_XY_ROOT/final-reference" \
  --candidate-dir "$MOUSE_XY_ROOT/final-candidate" \
  --output "$MOUSE_XY_ROOT/promotion.json"
```

The gate checks complete new seed coverage, report consistency, matched
environment/nav/reset contracts and paired intervals. It never publishes.
Repeat training seeds before making a robust algorithmic claim. If a candidate
fails, retain the failed result and return to development; do not tune on final
failures or report old and new seed pools as a paired improvement.
