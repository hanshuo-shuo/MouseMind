# MouseMind P2 results

## 1. What was implemented

A MiniMind-based hierarchical policy over three strategic skills, semantic temporal context, exact seeded replay, counterfactual outcome labels, MiniMind skill LoRA, structural history/instruction ablations, a non-language numeric upper reference, a calibrated capture-risk critic, development-only operating-point selection, OOD evaluation, and one corrective-data path.

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
| *Baselines* |  |  |  |  |
| random | 0.0% | 3.0% | 100.0% | 65.19 |
| direct-minimind-base | 0.0% | 0.0% | 100.0% | 113.60 |
| direct-minimind-lora | 1.0% | 26.0% | 99.0% | 98.60 |
| direct-mlp | 5.0% | 20.0% | 94.0% | 87.90 |
| p1-rule | 14.0% | 79.0% | 86.0% | 11.20 |
| *Proposed MiniMind hierarchy and ablations* |  |  |  |  |
| minimind-no-history | 1.0% | 80.0% | 99.0% | 13.15 |
| minimind-no-instruction | 1.0% | 80.0% | 99.0% | 13.15 |
| **minimind-learned (proposed)** | 12.0% | 97.0% | 88.0% | 7.37 |
| minimind-verified | 12.0% | 92.0% | 88.0% | 8.47 |
| *Non-language upper references* |  |  |  |  |
| numeric-learned | 38.0% | 100.0% | 62.0% | 2.66 |
| numeric-verified | 35.0% | 100.0% | 65.0% | 3.13 |

## 5. Main OOD results

| Condition | Proposed clean | Proposed task | Upper-reference clean | Upper-reference task |
| --- | ---: | ---: | ---: | ---: |
| faster_020 | 5.0% | 97.0% | 34.0% | 100.0% |
| faster_025 | 11.0% | 95.0% | 35.0% | 100.0% |
| shorter_los | 17.0% | 95.0% | 42.0% | 100.0% |
| unseen_language | 4.0% | 71.0% | 38.0% | 100.0% |

## 6. Risk-model calibration

Validation AUROC: 0.9585904005019346; AUPRC: 0.9087351404424037; Brier: 0.0753; ECE: 0.0255. The best swept verifier threshold was 0.9, but development clean success changed from 52.5% to 37.5% and capture rate moved in the wrong direction. Status: not promoted: no swept operating point improved clean success and capture rate without material task-success loss.

## 7. Ablation conclusions

MiniMind unseen-paraphrase accuracy / macro F1: 0.622 / 0.565. Full MiniMind no-history and instruction-removed ablations are stored in the offline planner report. The numeric upper reference reaches 0.839 / 0.767.

## 8. Failure modes that remain

Proposed-policy aggregate taxonomy: `{"capture_near_occlusion": 40, "capture_other": 37, "open_space_capture": 11}`.

## 9. Proposed hierarchy evidence

Against direct MiniMind LoRA, `minimind-learned` changes task success by +0.710 (95% CI +0.620 to +0.800), clean success by +0.110 (+0.040 to +0.180), and captures per episode by -91.23 (-100.99 to -81.18). Removing history changes task success by -0.170; removing instruction changes it by -0.170.

## 10. Oracle and upper-reference gap

The non-language `numeric-learned` upper reference reaches 100.0% task / 38.0% clean success. Proposed MiniMind reaches 97.0% / 12.0%; its paired clean-success gap is -0.260 (95% CI -0.370 to -0.150). This is reported as remaining headroom, not hidden or relabeled as the proposed method.

## P2.1 corrective-data iteration

Using 14 development failures, P2.1 added 56 anchors and 168 verified branches. Clean success changed from 52.5% to 45.0%, capture rate from 47.5% to 55.0%, and captures/episode from 1.90 to 2.55. The iteration was rejected for final.

## 11. Strongest supported resume bullet

Built a MiniMind-based hierarchical control policy over outcome-grounded counterfactual skills; on 100 untouched paired ID seeds, `minimind-learned` reached 97.0% task / 12.0% clean success, adding +71.0 task points and removing 91.23 captures per episode versus direct MiniMind LoRA; history and instruction ablations each lost about 17 task points, while a separately labeled non-language upper reference quantified the remaining clean-success gap.
