# MouseMind documentation

The project follows one method: verified counterfactual outcomes provide supervision
and preference pairs for an SFT/DPO skill planner, with specialist controllers handling
execution. Start with the [method](guides/method.md), [run the project](guides/evaluation.md),
or inspect the [final system evidence](results/final-system.md).

## Guides

| Document | What it covers |
| --- | --- |
| [Method](guides/method.md) | Counterfactual labels, data isolation, LoRA SFT/DPO, and the hierarchical interface. |
| [Implementation and ablations](guides/implementation.md) | Skill execution, continuous XY control, frozen-planner controls, and history/instruction ablations. |
| [Run and evaluate](guides/evaluation.md) | Installation, private assets, demo, and the complete evaluation command. |
| [Technical reference](guides/technical-guide.md) | Detailed observation contracts, offline diagnostics, evaluation tools, and provenance. |
| [Continuous workflow](guides/continuous-workflow.md) | Specialist preparation, post-training entry points, and evaluation contracts. |
| [P2 cluster workflow](guides/p2-cluster.md) | Historical planner-training and Slurm jobs. |
| [Transfer cluster workflow](guides/transfer-cluster.md) | Cross-task interface checks and Oasis evaluation. |

Commands in the guides are run from the repository root. Model weights, raw
trajectories, and per-episode records are stored separately from the public repository.

## Results

These reports preserve their original evaluation contracts. The historical planner
study, development selection, and final system evaluation use different seed pools.

| Report | Scope |
| --- | --- |
| [Final system](results/final-system.md) | Frozen **200-seed** comparison: 97.5% task success, 41.5% capture-free success, matched controls and intervals. |
| [Final protocol](results/final-system-protocol.md) | Candidate, seeds, and acceptance criteria frozen before final evaluation. |
| [Development comparison](results/continuous-development.md) | **80-seed** selection of the old DPO planner with continuous BC execution. |
| [Verified DPO](results/verified-dpo.md) | Historical **100-seed** SFT-to-DPO comparison, with its own SFT initialization. |
| [Historical benchmark](results/historical-benchmark.md) | Complete original policy table, resource discussion, and ablations. |
| [Planner benchmarks](results/planner-benchmarks.md) | P2 ID/OOD results, verification sweep, and rejected corrective iteration. |
| [Full-parameter training](results/full-finetuning.md) | Full SFT, DPO, and GRPO comparisons, including negative results. |
| [Verified RLVR](results/verified-rlvr.md) | LoRA/full-parameter alignment experiments and unsuccessful longer runs. |
| [Transfer](results/transfer.md) | Frozen BotEvade-to-Oasis transfer and instruction limitations. |
| [Upper references](results/oracle-analysis.md) | Numeric and privileged-controller comparisons under their own protocols. |
| [Action alignment](results/action-alignment.md) | One-step agreement with the source simulator-policy export. |
| [Trajectory alignment](results/trajectory-alignment.md) | Closed-loop behavioral-profile comparison. |

Aggregate JSON evidence and figures stay in [`mouse_llm/reports`](../mouse_llm/reports)
because the reporting tools, package data, and tests consume those paths.

## Archive

These are dated planning and validation snapshots. Statements about pending work,
unopened final seeds, or unpublished branches describe the time of each record.

| Record | Scope |
| --- | --- |
| [Continuous post-training plan](archive/continuous-posttraining-plan.md) | Original redesign and experiment plan. |
| [First continuous experiment](archive/continuous-pilot-20261102.md) | Completed BC and rejected continuous-context SFT/DPO runs; failure evidence retained. |
| [Initial validation](archive/continuous-validation-20261102.md) | Implementation checks recorded at the start of the continuous-control work. |

## Code and attribution

- [Source package map](../mouse_llm/README.md): data, planners, controllers, evaluation, and cluster scripts.
- [Environment provenance](../mouse_llm/envs/mice/SOURCE.md) and [third-party notices](../THIRD_PARTY_NOTICES.md).
- [Project overview](../README.md).
