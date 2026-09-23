# Verified DPO results

## Configuration

- Source: 320 verified exact-state P2 collection anchors, horizon 8; one best-versus-worst pair per anchor and preference when utility margin is positive.
- The original anchor-ID split shared collection seeds across train/validation/test. For strict seed isolation, this experiment regrouped all 320 anchors by collection seed (69 train, 6 validation, 5 test seeds), removed zero-margin pairs, and retrained SFT on the same 1,072 chosen responses used by DPO. The original best SFT checkpoint was not used as initialization because it had seen the held-out collection seeds.
- DPO: this seed-clean SFT LoRA initializes both the trainable policy and frozen reference; base MiniMind is frozen. β = 0.1, 0.05, 0.2; one epoch; 67 LoRA optimizer updates per run.
- Evaluation: frozen P2 final ID contract, 100 seeds excluded from training, survival-first preference, planner horizon 8, the same specialist and constrained three-skill JSON output. Three DPO β values were evaluated on this same final pool, so the comparisons are exploratory rather than a one-shot confirmatory test.
- Contract SHA-256: `9572230201a1c2a8afb764012237df8c43d155c4941bd2f7282282889af2ed75`; seed SHA-256: `89727fc394044af79b18eb2bf77f9bf95c4925494621c90ca9e1cbc8e0a391ba`; seed-clean SFT LoRA SHA-256: `ab8dd692b542b8068e0e1eba0f204e3e28adc3af715db208dce8233cd2d93472`.

## Closed-loop results

| Policy | Task success | Clean success | Capture rate | Captures / episode |
| --- | ---: | ---: | ---: | ---: |
| Seed-clean SFT | 92.0% | 8.0% | 92.0% | 8.94 |
| DPO β=0.1 | 100.0% | 22.0% | 78.0% | 4.94 |
| DPO β=0.05 | 96.0% | 21.0% | 78.0% | 5.55 |
| DPO β=0.2 | 91.0% | 15.0% | 85.0% | 8.27 |

## Historical current-LoRA reference

The previously published skill-planner LoRA used the same frozen contract, final-seed SHA-256, 100 final ID seeds, planner horizon, preference, specialist, and environment settings. It was trained with the earlier anchor-ID split, so this is a matched evaluation comparison, **not** the seed-clean initialization comparison above.

| Policy | Task success | Clean success | Capture rate | Captures / episode |
| --- | ---: | ---: | ---: | ---: |
| Published skill-planner LoRA | 97.0% | 12.0% | 88.0% | 7.37 |
| Verified DPO β=0.1 | 100.0% | 22.0% | 78.0% | 4.94 |

On these paired seeds, DPO minus the published LoRA changed task success by +3.00 pp (95% CI [0.00, +7.00]), clean success by +10.00 pp ([-0.03, +20.00]), capture rate by −10.00 pp ([−20.00, +1.00]), and captures per episode by −2.43 ([−4.67, −0.38]). The captures-per-episode interval excludes zero; the clean-success and capture-rate intervals do not. The per-seed comparison is stored privately at `verified_dpo/paired_vs_historical_sft.json` under the P2 run root.

## Paired DPO − SFT

Mean difference and 95% paired bootstrap CI across the same 100 seeds.

| Policy | Metric | Delta | 95% CI |
| --- | --- | ---: | ---: |
| DPO β=0.1 | task success | +8.00 pp | [+3.00, +14.00] pp |
| DPO β=0.1 | clean success | +14.00 pp | [+5.00, +23.00] pp |
| DPO β=0.1 | capture rate | -14.00 pp | [-23.00, -5.00] pp |
| DPO β=0.1 | captures / episode | -4.00 | [-6.21, -1.91] |
| DPO β=0.05 | task success | +4.00 pp | [-2.00, +11.00] pp |
| DPO β=0.05 | clean success | +13.00 pp | [+4.00, +22.00] pp |
| DPO β=0.05 | capture rate | -14.00 pp | [-23.00, -5.00] pp |
| DPO β=0.05 | captures / episode | -3.39 | [-5.51, -1.29] |
| DPO β=0.2 | task success | -1.00 pp | [-8.00, +6.00] pp |
| DPO β=0.2 | clean success | +7.00 pp | [-2.00, +16.00] pp |
| DPO β=0.2 | capture rate | -7.00 pp | [-16.00, +2.00] pp |
| DPO β=0.2 | captures / episode | -0.67 | [-3.50, +2.26] |

## Offline skill accuracy

- Seed-clean SFT: seen instruction 0.590; unseen paraphrase 0.564.
- DPO β=0.1: seen instruction 0.551; unseen paraphrase 0.474.
- DPO β=0.05: seen instruction 0.449; unseen paraphrase 0.385.
- DPO β=0.2: seen instruction 0.603; unseen paraphrase 0.538.

## Exact-state final-rollout diagnostic

- all: 302 / 1684 decisions changed; SFT wrong → DPO correct 4; SFT correct → DPO wrong 1; both wrong 2; replay unverified 295.
- predator_visible: 210 / 1212 decisions changed; SFT wrong → DPO correct 3; SFT correct → DPO wrong 0; both wrong 0; replay unverified 207.
- near_occlusion: 111 / 566 decisions changed; SFT wrong → DPO correct 2; SFT correct → DPO wrong 1; both wrong 0; replay unverified 108.
Only 7 / 302 changed decisions passed deterministic replay. The oracle-correction counts apply only to this verified subset and cannot explain the overall closed-loop improvement.

## Conclusion

On the same 100 training-excluded final seeds, β=0.1 changed task success from 92.0% to 100.0%, clean success from 8.0% to 22.0%, and captures per episode from 8.94 to 4.94. This supports a closed-loop safety gain without sacrificing task success in this fixed ID protocol.
The β=0.05 run shows a similar but smaller safety gain; β=0.2 is inconclusive by the paired intervals. β=0.1 has lower offline seen and unseen-paraphrase accuracy than seed-clean SFT, so offline skill accuracy did not predict the closed-loop safety change.
This controlled comparison uses a retrained seed-clean SFT baseline. The previously published SFT LoRA is a historical reference, not the initialization here. Retrospective development evaluation also favored β=0.1, but it does not undo the multiple-variant exposure of the final pool. The result should not be read as a general safety guarantee. The [Verified RLVR summary](VERIFIED_ALIGNMENT_RESULTS.md) reports the unsuccessful GRPO and longer-training runs without promoting them.
