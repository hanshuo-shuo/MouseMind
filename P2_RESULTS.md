# MouseMind P2 results

## 1. What was implemented

A frozen fresh-seed contract, semantic temporal planner context, exact seeded replay, counterfactual branching for all three skills, outcome-grounded language targets, a numeric planner, MiniMind skill LoRA, a calibrated capture-risk critic, propose-verify control, development-only operating-point selection, OOD evaluation, and one corrective-data path.

## 2. Exact experiments run

- Counterfactual anchors: 320 anchors and 1920 verified branches over horizons [4, 8].
- Numeric planner validation: 124 samples.
- MiniMind unseen-paraphrase test: 148 samples.
- Fresh ID: 100 paired seeds from `final_id_test`.
- OOD conditions: faster_020, faster_025, shorter_los, unseen_language.

## 3. Verified versus pending

All numbers below come from aggregate artifacts marked `research_evidence=true`. Private states, trajectories, predictions, and episode CSV files remain outside Git. Historical P1 clean success is pending because its joint per-episode success/capture records were unavailable; no value was inferred.

## 4. Main fresh-ID results

| Policy | Clean success | Task success | Capture rate | Captures / episode |
| --- | ---: | ---: | ---: | ---: |
| direct-minimind-base | 0.0% | 0.0% | 100.0% | 113.60 |
| direct-minimind-lora | 1.0% | 26.0% | 99.0% | 98.60 |
| direct-mlp | 5.0% | 20.0% | 94.0% | 87.90 |
| minimind-learned | 12.0% | 97.0% | 88.0% | 7.37 |
| minimind-no-history | 1.0% | 80.0% | 99.0% | 13.15 |
| minimind-no-instruction | 1.0% | 80.0% | 99.0% | 13.15 |
| minimind-verified | 12.0% | 92.0% | 88.0% | 8.47 |
| numeric-learned | 38.0% | 100.0% | 62.0% | 2.66 |
| numeric-verified | 35.0% | 100.0% | 65.0% | 3.13 |
| p1-rule | 14.0% | 79.0% | 86.0% | 11.20 |
| random | 0.0% | 3.0% | 100.0% | 65.19 |

## 5. Main OOD results

| Condition | Clean success | Task success | Capture rate |
| --- | ---: | ---: | ---: |
| faster_020 | 34.0% | 100.0% | 66.0% |
| faster_025 | 35.0% | 100.0% | 65.0% |
| shorter_los | 42.0% | 100.0% | 58.0% |
| unseen_language | 38.0% | 100.0% | 62.0% |

## 6. Risk-model calibration

Validation AUROC: 0.9585904005019346; AUPRC: 0.9087351404424037; Brier: 0.0753; ECE: 0.0255. The best swept verifier threshold was 0.9, but development clean success changed from 52.5% to 37.5% and capture rate moved in the wrong direction. Status: not promoted: no swept operating point improved clean success and capture rate without material task-success loss.

## 7. Ablation conclusions

Numeric planner validation accuracy / macro F1: 0.839 / 0.767. MiniMind unseen-paraphrase accuracy / macro F1: 0.622 / 0.565. Full MiniMind no-history and instruction-removed ablations are stored in the offline planner report.

## 8. Failure modes that remain

Selected-policy aggregate taxonomy: `{"capture_near_occlusion": 32, "capture_other": 22, "open_space_capture": 8}`.

## 9. Does P2 beat P1?

For `numeric-learned`, candidate minus P1 is +0.240 clean success (95% CI +0.120 to +0.360), +0.210 task success (+0.130 to +0.290), and -0.240 capture rate (-0.360 to -0.120). Interpret improvement under the metric named; do not call the policy safe from task success alone.

## P2.1 corrective-data iteration

Using 14 development failures, P2.1 added 56 anchors and 168 verified branches. Clean success changed from 52.5% to 45.0%, capture rate from 47.5% to 55.0%, and captures/episode from 1.90 to 2.55. The iteration was rejected for final.

## 10. Strongest supported resume bullet

Built a counterfactual learned control hierarchy and evaluated calibrated runtime risk overrides; on 100 untouched paired ID seeds, the selected `numeric-learned` reached 100.0% task success and 38.0% clean success (+24.0 points versus the P1 rule hierarchy), with fresh OOD evaluation across 4 shifts; verifier overrides were rejected when they worsened closed-loop clean success despite strong offline AUROC.
