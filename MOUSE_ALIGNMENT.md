# Source mouse trajectory alignment

This is a low-level component diagnostic on the frozen, episode-isolated held-out subset (N=512). It measures one-step agreement with the source trajectory actions. Hierarchical methods are compared separately over complete closed-loop episodes in [TRAJECTORY_ALIGNMENT.md](TRAJECTORY_ALIGNMENT.md).

| Policy | Overall action agreement | Predator hidden (N=492) | Predator visible (N=20) | Destination error | Action-distribution JS |
| --- | ---: | ---: | ---: | ---: | ---: |
| Base MiniMind (constrained) | 1.2% (0.4–2.1%) | 1.2% (0.4–2.2%) | 0.0% (0.0–0.0%) | 29.1% (27.7–30.7%) | 0.954 bits |
| Mouse-policy LoRA (constrained) | 41.0% (36.5–45.1%) | 41.1% (36.6–45.1%) | 40.0% (20.0–60.0%) | 9.4% (8.2–10.7%) | 0.238 bits |
| MLP BC (low-level upper reference) | 55.7% (51.4–60.0%) | 56.5% (52.2–60.8%) | 35.0% (15.0–55.0%) | 2.2% (2.0–2.5%) | 0.128 bits |

Agreement is exact action-ID match (higher is better). Destination error and Jensen–Shannon divergence are lower-is-better. Parentheses are sample-bootstrap 95% confidence intervals; JS is shown as a point estimate.

The source file is a legacy BotEvade simulator-policy transition export. It should not be described as biological mouse behavior. One-step alignment also must not be used as a substitute for the separate closed-loop task and capture results.
