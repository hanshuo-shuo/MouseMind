# Mouse trajectory behavioral-profile alignment

The score compares each policy's 100 fresh-ID closed-loop episodes with 500 episode-isolated held-out source episodes across 11 bounded behavioral features. Lower is better; parentheses are bootstrap 95% confidence intervals.

| Role | Method | Alignment distance ↓ |
| --- | --- | ---: |
| baseline | Random | 0.483 (0.475–0.490) |
| baseline | Direct MiniMind base | 0.590 (0.582–0.597) |
| baseline | Direct MiniMind LoRA | 0.531 (0.503–0.554) |
| specialist baseline | Direct MLP BC | 0.520 (0.490–0.547) |
| rule hierarchy baseline | P1 rule hierarchy | 0.315 (0.288–0.340) |
| proposed ablation | MiniMind hierarchy, no history | 0.347 (0.332–0.361) |
| proposed ablation | MiniMind hierarchy, no instruction | 0.347 (0.332–0.361) |
| proposed MiniMind hierarchy | **MiniMind hierarchy (full)** | 0.288 (0.272–0.304) |
| proposed + rejected verifier | MiniMind hierarchy + verifier | 0.297 (0.276–0.316) |
| non-language upper reference | Numeric planner | 0.200 (0.174–0.225) |
| upper-reference verifier variant | Numeric planner + verifier | 0.215 (0.190–0.240) |

The full MiniMind hierarchy is the best MiniMind-based variant and is clearly closer than the direct policies and its ablations. Its point estimate is also lower than P1, although that comparison's bootstrap interval overlaps zero. The numeric planner remains a non-language upper reference rather than evidence that MiniMind is the unconstrained overall winner.

This source is a BotEvade simulator-policy export, not biological mouse behavior. The scalar score is descriptive; the aggregate JSON retains every feature-level gap.
