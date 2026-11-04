# MouseMind source package

For the project story and headline results, start at the [repository homepage](../README.md).
The [documentation index](../docs/README.md) groups guides, experiment reports, and historical records.

| Directory | Responsibility |
| --- | --- |
| [`data`](data) | Observation contracts, episode reconstruction, and planner supervision. |
| [`training`](training) | Specialist and MiniMind task-adaptation training entry points. |
| [`hierarchical`](hierarchical) | Semantic context, constrained skill planning, and the original skill controllers. |
| [`continuous`](continuous) | Continuous GO specialists, legacy-planner adapter, navigation, and evaluation. |
| [`baselines`](baselines) | Direct-action and non-language comparison policies. |
| [`evaluation`](evaluation) | Offline, closed-loop, transfer, and report-generation tools. |
| [`verified_rl`](verified_rl) | Verified rollout utilities and GRPO experiments. |
| [`envs`](envs) | Environment adapters, task assets, and vendored runtime with provenance. |
| [`northwestern`](northwestern) | Slurm job scripts; see the cluster guides in the documentation index. |
| [`tests`](tests) | Public synthetic tests and opt-in cached-environment integration checks. |
| [`reports`](reports) | Public aggregate evidence and figures used by report generators and tests. |

The detailed former package README is now the [technical reference](../docs/guides/technical-guide.md).
Python module paths and CLI entry points are unchanged. Private data and model weights
are not stored in this package.
