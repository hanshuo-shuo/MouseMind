# MouseMind baseline, proposed hierarchy, and oracle analysis

## Claim and comparison roles

MouseMind's primary contribution is the **MiniMind-based hierarchical policy**:
a language-conditioned high-level planner selects `go_to_goal`,
`evade_predator`, or `hold_position`, while a compact behavior-cloning
specialist executes low-level goal motion. The central comparison is therefore
the full hierarchy against direct MiniMind deployment and its own structural
ablations—not a claim that MiniMind exceeds every non-language reference.

| Role | Methods | Purpose |
| --- | --- | --- |
| Baselines | Random, direct MiniMind base/LoRA, P1 rule hierarchy | Establish chance, direct-policy, and hand-coded hierarchy performance. |
| Low-level upper reference | MLP behavior-cloning specialist | Measures the strongest supervised 10D-to-action specialist available to the hierarchy; it is not a literal oracle. |
| **Proposed method** | **Full MiniMind hierarchy** | Tests learned language-conditioned strategy with semantic temporal history. |
| Structural ablations | MiniMind without history; MiniMind without instruction | Test whether the proposed context and instruction pathways matter. |
| High-level upper reference | Numeric learned planner | Uses the same semantic context without language generation; treated as a non-language ceiling/teacher, not as the proposed method. |
| Privileged target oracle | Aligned goal-coordinate controller | Reads the active Oasis goal coordinates directly and therefore bounds the cost of target-interface uncertainty. |
| Training-label oracle | Exact-replay counterfactual branch selection | Supplies outcome-grounded skill labels during training; unavailable as a deployment policy. |

“Upper reference” is used where the comparator is strong but lacks privileged
ground truth. “Oracle” is reserved for extra target information or exact replay
outcomes that the deployed MiniMind policy does not receive.

## Proposed hierarchy versus baselines

All deltas below are paired over the same 100 untouched BotEvade final seeds.
Positive task/clean values and negative capture values favor the full MiniMind
hierarchy.

| Full MiniMind minus reference | Δ task success | Δ clean success | Δ capture rate | Δ captures / episode |
| --- | ---: | ---: | ---: | ---: |
| Direct MiniMind LoRA | **+71 pts** (+62 to +80) | **+11 pts** (+4 to +18) | **−11 pts** (−18 to −5) | **−91.23** (−100.99 to −81.18) |
| MiniMind without history | **+17 pts** (+8 to +26) | **+11 pts** (+5 to +18) | **−11 pts** (−18 to −4) | **−5.78** (−8.67 to −2.89) |
| MiniMind without instruction | **+17 pts** (+9 to +26) | **+11 pts** (+5 to +18) | **−11 pts** (−18 to −5) | **−5.78** (−8.65 to −2.94) |
| P1 rule hierarchy | **+18 pts** (+10 to +26) | −2 pts (−12 to +8) | +2 pts (−8 to +11) | **−3.83** (−6.65 to −1.11) |

The direct-LoRA comparison isolates the architectural result: moving MiniMind
from direct 295-way action generation to high-level skill planning raises task
success from 26% to 97%, raises clean success from 1% to 12%, and removes 91.23
captures per episode. Both ablations regress to 80% task / 1% clean success,
supporting the full temporal and instruction-conditioned design. Relative to
P1, MiniMind improves completion and capture count, while clean-success and
capture-incidence intervals overlap; this is a trade-off, not strict dominance.

## Alignment with source mouse trajectories

### Low-level action agreement

The 512-state teacher-forced table is a component diagnostic. The supervised
MLP is the low-level upper reference; the proposed hierarchy uses this
specialist for `go_to_goal`, so direct MiniMind is not expected to win this
particular 295-way coordinate prediction task.

| Method | Exact action agreement | Destination error | Action-distribution JS |
| --- | ---: | ---: | ---: |
| Direct MiniMind base | 1.2% | 29.1% | 0.954 bits |
| Direct MiniMind LoRA | 41.0% | 9.4% | 0.238 bits |
| MLP BC low-level upper reference | **55.7%** | **2.2%** | **0.128 bits** |

### Closed-loop behavioral-profile alignment

The more relevant hierarchy-level diagnostic compares 100 fresh-ID rollouts
per policy with 500 episode-isolated held-out source episodes. The equal-weight
distance covers task/clean success, capture behavior, action switching,
oscillation, motion, goal progress, path efficiency, and predator visibility.
Lower is better.

| Role | Method | Behavioral-profile distance (95% CI) ↓ |
| --- | --- | ---: |
| Baseline | Direct MiniMind LoRA | 0.531 (0.503–0.554) |
| Low-level reference | Direct MLP BC | 0.520 (0.490–0.547) |
| Rule baseline | P1 rule hierarchy | 0.315 (0.288–0.340) |
| Ablation | MiniMind without history | 0.347 (0.332–0.361) |
| Ablation | MiniMind without instruction | 0.347 (0.332–0.361) |
| **Proposed** | **Full MiniMind hierarchy** | **0.288 (0.272–0.304)** |
| Rejected variant | MiniMind + verifier | 0.297 (0.276–0.316) |
| Non-language upper reference | Numeric learned planner | 0.200 (0.174–0.225) |

Full MiniMind improves the alignment distance by 0.243 versus direct MiniMind
LoRA (95% bootstrap interval 0.212–0.270) and by 0.059 versus either structural
ablation (0.038–0.079). Its 0.026 point-estimate improvement over P1 is not
resolved by the bootstrap interval (−0.056 to +0.005 for MiniMind minus P1).
The full feature-level evidence is in
[trajectory_profile_alignment.json](mouse_llm/reports/trajectory_profile_alignment.json).

## Oracle and upper-reference gap

Oracle/reference rows are deliberately reported after the proposed-method
analysis. They show remaining headroom rather than redefining the contribution.

| Evaluation | Proposed MiniMind hierarchy | Oracle / upper reference | Proposed minus reference |
| --- | ---: | ---: | ---: |
| BotEvade task success | 97% | Numeric: 100% | −3 pts (95% CI −7 to 0) |
| BotEvade clean success | 12% | Numeric: 38% | −26 pts (−37 to −15) |
| BotEvade captures / episode | 7.37 | Numeric: 2.66 | +4.71 (+2.95 to +6.74) |
| Behavioral-profile distance | 0.288 | Numeric: 0.200 | +0.089 (+0.062 to +0.117) |
| Oasis task success | 89% | Goal-coordinate oracle: 100% | −11 pts (−17 to −5) |
| Oasis clean success | 0% | Goal-coordinate oracle: 41% | −41 pts (−51 to −31) |
| Oasis captures / episode | 32.39 | Goal-coordinate oracle: 1.21 | +31.18 (+27.70 to +34.65) |

The numeric reference shows that the semantic hierarchy can support better
safety than the current language planner. The goal-coordinate oracle shows
that most Oasis completion failure is removable at the interface, while the
large clean-success gap shows that the source safety strategy does not yet
transfer. These are explicit next-model targets.

## Strongest supported positioning

> MouseMind's MiniMind hierarchy converts a weak direct language policy into a
> high-performing strategic controller: on 100 untouched BotEvade seeds it
> improves task success from 26% to 97% and reduces captures by 91.23 per
> episode versus direct MiniMind LoRA. It is the strongest MiniMind variant,
> requires both temporal history and instruction conditioning, and is
> substantially closer to held-out source trajectory behavior than direct
> policies. Non-language and privileged oracle references remain higher,
> quantifying rather than hiding the remaining alignment gap.

The source trajectories are from a BotEvade simulator-policy export, not
biological mice. Behavioral alignment and task success are separate from a
safety guarantee.
