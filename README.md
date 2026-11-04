# MouseMind

**Counterfactual post-training for hierarchical control.**

MouseMind adapts a 64M-class MiniMind model to make strategic decisions in the
Cellworld BotEvade simulator. Exact-state counterfactual rollouts provide SFT
labels and DPO preference pairs; task-specific controllers execute the chosen skill.

The complete system combines a DPO-trained planner with continuous behavior
cloning for goal execution. On **200 held-out final seeds**, it achieves
**97.5% task success and 41.5% capture-free success**.

## Method

![Counterfactual outcomes train the MiniMind planner; controllers execute its selected skill](mouse_llm/reports/figures/mousemind_architecture.svg)

1. **Compare outcomes.** Replay candidate skills from the same simulator state
   and score their outcomes to construct supervision and preference pairs.
2. **Adapt the planner.** Train a skill-planner LoRA with SFT, then DPO using a
   frozen copy of the SFT adapter as its reference. The MiniMind backbone stays frozen.
3. **Execute skills.** Given an instruction, semantic state, and eight steps of
   history, select `go_to_goal`, `evade_predator`, or `hold_position` through constrained JSON.

The goal specialist learns from a legacy simulator-policy export containing
118,861 transitions and 4,916 reconstructed episodes. Planner supervision uses
verified counterfactual branches, with seed-isolated data and paired closed-loop evaluation.
See the [method guide](docs/guides/method.md) for data and training details.

## Results

**Planner post-training.** In the controlled 100-seed study, DPO was compared
with its own seed-isolated SFT initialization using the same specialist.

| Planner | Task success | Capture-free success | Captures / episode |
| --- | ---: | ---: | ---: |
| SFT initialization | 92% | 8% | 8.94 |
| Verified LoRA DPO | 100% | 22% | 4.94 |

Multiple DPO settings used that fixed test pool, so this study is exploratory.
The [planner results](docs/results/verified-dpo.md) retain the paired intervals
and distinguish the SFT initialization from earlier published baselines.

**Complete system.** The final configuration was selected on 80 development
seeds and frozen before evaluation on a separate 200-seed pool.

| Metric | Final system |
| --- | ---: |
| Task success | **97.5% (195/200)** |
| Capture-free success | **41.5% (83/200)** |
| Captures / episode | **5.275** |

Task success means reaching the goal; capture-free success also requires zero
captures. Multiple captures can occur in one episode. These full-system results
use a different evaluation contract from the planner study above.

The [final report](docs/results/final-system.md) provides matched component
comparisons, paired intervals, and provenance. Executor design and the frozen-planner
ablation are explained in the [implementation guide](docs/guides/implementation.md).

## Quick start

```bash
python -m pip install -e '.[env]'
python demo.py
```

The public demo uses synthetic data to exercise the code path. Reproducing the
research evaluation requires separately stored checkpoints and the Cellworld cache.
Follow the [evaluation guide](docs/guides/evaluation.md) for setup and commands.

## Documentation

Start with the [documentation index](docs/README.md).

| Read about | Guide |
| --- | --- |
| Counterfactual labels, SFT/DPO, and data contracts | [Method](docs/guides/method.md) |
| Skill execution, interfaces, and ablations | [Implementation](docs/guides/implementation.md) |
| Installation, checkpoints, and evaluation | [Evaluation](docs/guides/evaluation.md) |
| Benchmarks, controls, and retained negative results | [Results index](docs/README.md#results) |
| Earlier plans and experiment logs | [Archive index](docs/README.md#archive) |

## Code map

| Path | Purpose |
| --- | --- |
| `mouse_llm/data`, `mouse_llm/training` | Supervision preparation and policy training. |
| `mouse_llm/hierarchical`, `mouse_llm/continuous` | Planner context, skills, execution, and continuous control. |
| `mouse_llm/baselines`, `mouse_llm/evaluation`, `mouse_llm/verified_rl` | Controls, evaluation, and additional alignment experiments. |
| `model`, `trainer`, `dataset` | MiniMind architecture, tokenizer, and base training utilities. |

## Scope and attribution

The final results cover fixed checkpoints in one simulator world. The hierarchy
improves capture-free completion over direct BC but has more captures per episode.
Cross-map and training-run generalization remain unmeasured for this executor.
Failed training variants remain documented; private trajectories and weights are excluded from Git.

MiniMind architecture and base utilities come from [jingyaogong/minimind](https://github.com/jingyaogong/minimind).
The repository retains Apache 2.0; the vendored Cellworld runtime retains its MIT notice.
See [third-party notices](THIRD_PARTY_NOTICES.md) and [environment provenance](mouse_llm/envs/mice/SOURCE.md).
