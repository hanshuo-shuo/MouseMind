# MouseMind — Learning Strategy Under Closed-Loop Risk

**Offline imitation did not predict closed-loop control.**

A 145K behavior-cloning specialist achieved **55.7% held-out action accuracy**
but only **17% historical closed-loop task success**. A hand-coded hierarchical
controller raised that to 74%, revealing that the missing abstraction was
strategy rather than low-level capacity. MouseMind P2 then learned the
high-level policy from 1,920 exact-replay counterfactual branches and evaluated
it once on 100 untouched final seeds.

The selected 17.9K-parameter numeric planner reached **100% task success and
38% clean success**, versus 79% / 14% for the P1 rule hierarchy. The paired
improvement was +21 task points (95% CI +13…+29), +24 clean points (+12…+36),
and -24 capture points. A calibrated risk critic scored 0.959 AUROC offline but
made every swept closed-loop operating point worse, so it was not promoted.

[Technical guide](mouse_llm/README.md) · [P2 results](P2_RESULTS.md) · [Public smoke demo](#run-the-public-demo)

![Fresh-ID safety and task-success frontier](mouse_llm/reports/figures/p2_safety_frontier.png)

| Fresh ID, 100 paired seeds | Task success | Clean success | Capture rate | Captures / episode |
| --- | ---: | ---: | ---: | ---: |
| **Numeric learned planner, K=8** | **100%** | **38%** | **62%** | **2.66** |
| MiniMind learned planner | 97% | 12% | 88% | 7.37 |
| P1 rule hierarchy | 79% | 14% | 86% | 11.20 |
| Direct 145K MLP | 20% | 5% | 94% | 87.90 |

![Fresh ID and OOD robustness](mouse_llm/reports/figures/p2_ood_generalization.png)

| Selected numeric planner | ID | Predator 0.20 | Predator 0.25 | Shorter LOS | Unseen language |
| --- | ---: | ---: | ---: | ---: | ---: |
| Task success | 100% | 100% | 100% | 100% | 100% |
| Clean success | 38% | 34% | 35% | 42% | 38% |

The numeric planner does not consume language strings, so unseen-language
performance is unchanged by construction. MiniMind is the language test: its
task / clean success fell from 97% / 12% on ID to 71% / 4% on unseen language,
despite 62.2% offline unseen-paraphrase skill accuracy.

> Thirty-second takeaway: hierarchical strategy solved task completion;
> counterfactual learning made the strategy adaptive and substantially cleaner;
> language conditioning helped, but the numeric temporal planner was stronger;
> offline risk calibration did not translate into useful runtime overrides.

## What I built

- Reconstructed 4,916 episodes from 118,861 legacy transitions using explicit
  terminals plus state-continuity boundaries.
- Prevented leakage with episode-level splits and serialized-prompt ownership
  across train, validation, and test.
- Extracted and packaged the BotEvade and Oasis Gymnasium environments with
  their minimal Cellworld runtime dependency closure.
- Adapted a 64.3M MiniMind checkpoint with a native-PyTorch LoRA path.
- Added a fair `10 → 256 → 256 → 295` MLP behavior-cloning specialist.
- Added free and token-constrained JSON decoding so output formatting can be
  separated from policy quality.
- Built paired closed-loop rollouts over identical seeds with bootstrap
  confidence intervals, return, success, capture, survival, path efficiency,
  p50/p95/p99 latency, action Hz, and control-deadline misses.
- Recovered the authoritative 2025 observation source and verified all ten
  fields with commit/blob hashes, full-dataset distribution checks, and signed
  angle state replay; formal reports now carry `research_evidence=true`.
- Added an instruction-conditioned high-level planner over the MLP goal
  specialist, with a temporal-history interface, replanning cadence, safety
  interrupts, and planner/controller latency accounting.
- Froze disjoint collection/development/final seed pools and added clean success
  without changing the historical task-success definition.
- Collected 320 strategic anchors from rule, direct-MLP, and perturbed-skill
  trajectories; exact replay verified all 1,920 skill/horizon branches.
- Trained a 17.9K numeric planner, MiniMind skill LoRA, and a calibrated 17.7K
  capture-risk critic over the same semantic eight-step temporal context.
- Swept planner horizons and verifier thresholds on development only, rejected
  the verifier when it worsened every operating point, and evaluated the frozen
  system once on fresh ID and OOD conditions.
- Closed one P2.1 failure-data iteration; hard-failure upweighting worsened clean
  success from 52.5% to 45.0%, so the iteration was not selected.
- Added deterministic failure taxonomy and replay-queue mining for targeted
  corrective demonstrations.
- Built private-data guards, synthetic CI, a public one-command demo, and Slurm
  pipelines for data preparation, training, evaluation, and benchmarking.

## What is upstream

The MiniMind architecture, tokenizer, base training code, and original model
utilities come from [jingyaogong/minimind](https://github.com/jingyaogong/minimind).
The extracted environment originates from my separate `Mice` project; its
vendored `cellworld_game` runtime retains its upstream MIT license. See
[environment provenance](mouse_llm/envs/mice/SOURCE.md) for exact commits and
local packaging changes.

This attribution boundary is intentional: MouseMind owns the policy-system
work listed above; it does not claim authorship of MiniMind.

## System

```mermaid
flowchart LR
    A["Private transition CSV"] --> B["Validation + episode reconstruction"]
    B --> C["Leakage-safe JSONL splits"]
    C --> D["MiniMind direct-action BC"]
    C --> E["145K MLP goal specialist"]
    I["Instruction + semantic history"] --> P["Learned skill proposal"]
    X["Exact replay + counterfactual branches"] --> P
    P --> V["Calibrated risk critic"]
    V --> Q["Goal / evade / hold skill"]
    E --> Q
    Q --> G["Seeded BotEvade rollouts"]
    D --> G
    H["Random policy"] --> G
    G --> J["Success · captures · return · latency"]
    J --> K["Failure taxonomy + replay queue"]
    K --> L["Corrective demonstrations / next policy"]
```

## System evolution

| Version | Evidence and decision |
| --- | --- |
| V0 — Direct MiniMind BC | LoRA reaches 41.0% offline accuracy, but only 23% historical task / 1% clean success under valid constrained decoding. |
| V1 — Specialist deployment | A 145K MLP wins offline at 55.7%, but reaches only 17% historical closed-loop success. |
| P1 — Rule hierarchy | Keep the MLP as goal specialist; strategic abstraction reaches 74% historical success. |
| P2 — Learned hierarchy | Counterfactual numeric planning reaches 100% task / 38% clean success on 100 untouched final seeds. |
| P2 Verify | The critic is well calibrated offline but all runtime threshold points are worse; verifier not promoted. |
| P2.1 | Corrective hard-failure upweighting reduces development clean success; iteration rejected. |

## Run the public demo

The public demo exercises the exact closed-loop reporting path with a synthetic
arena and the real 295-action catalog. It needs no private data, checkpoint, or
Cellworld download:

```bash
python demo.py
```

It writes paired episode records and aggregate metrics to
`/tmp/mousemind-public-demo/`. The report marks itself
`research_evidence=false`; its numbers are an engineering smoke test only.

Run the full public test suite and privacy check with:

```bash
python -m pytest mouse_llm/tests -q
python -m mouse_llm.privacy_guard
```

## Verified offline result

Northwestern BCS516 Slurm job `19489` finished successfully in 5 minutes 44
seconds. It trained
0.393M LoRA parameters for two epochs / 3,750 optimizer steps on 60,000 private
training transitions. The 145,447-parameter MLP used the same 60K train split,
5K validation split, and deterministic 512-sample held-out subset.

![MouseMind held-out offline policy baselines](mouse_llm/reports/figures/mousemind_offline_baselines.png)

| Held-out metric | Base MiniMind | Mouse-policy LoRA | MLP BC |
| --- | ---: | ---: | ---: |
| Strict JSON output | 0.0% | 100.0% | N/A—direct logits |
| Exact action accuracy | 0.0% | 41.0% (95% CI 36.9–45.3%) | **55.7%** (95% CI 51.4–60.0%) |
| Action-response NLL | 2.447 | **0.296** (95% CI 0.277–0.316) | N/A |
| Normalized destination error | 100.0% | 9.39% (95% CI 8.21–10.61%) | **2.23%** (95% CI 1.97–2.49%) |

The table preserves the original free-decoding run. A repaired explicit trie
decoder then made both MiniMind models 100% JSON-valid on the same 512 samples:
base reached 1.17% exact accuracy / 29.13% destination error, while LoRA retained
41.02% / 9.45%. Formatting therefore did not explain the policy-quality gap.
For this numeric task, the 145K specialist remained the stronger offline model.

## Closed-loop benchmark contract

The online benchmark defaults to 100 paired seeds. Every policy is evaluated
under the same environment configuration and seed list.

| Historical policy, seeds 1000..1099 | Task success | Clean success | Capture rate | Captures / episode |
| --- | ---: | ---: | ---: | ---: |
| Direct MiniMind base, constrained | 0% | 0% | 100% | 111.49 |
| Direct MiniMind LoRA, constrained | 23% | 1% | 99% | 96.92 |
| Direct MLP | 17% | pending recomputation | 99% | 90.95 |
| P1 rule hierarchy | 74% | pending recomputation | 85% | 9.96 |

The random/MLP rows are verified over the identical 100 local BotEvade seeds;
the hierarchical row uses those same seeds and the verified observation audit.
Against direct MLP, it improved success by 57 points (paired 95% CI 44–69),
reduced captures by 80.99 per episode (72.98–88.88), and improved return by
81.56 (73.27–89.28). Historical clean success for direct MLP and P1 is
intentionally pending because their old private episode rows are unavailable;
no value was inferred from separate success and capture marginals. Per-episode
records remain private.

## Why use a language model?

For a single numeric control task, the MLP is the more parameter-efficient
specialist. Closed-loop deployment then showed its limitation: it can make goal
progress while repeatedly entering unsafe states. MouseMind therefore keeps the
MLP at the low-level controller and moves language/context to high-level skill
selection and replanning.

The language model is therefore evaluated at the strategic layer, not used as a
needlessly expensive coordinate classifier. Skill LoRA improved offline planner
accuracy from 25.0% to 63.5% and reached 62.2% on frozen unseen paraphrases;
removing history reduced accuracy to 57.4%, and removing the instruction reduced
it to 50.7%. In fresh closed loop it reached 97% task success, demonstrating
useful learned strategy, but only 12% clean success. The numeric planner reached
38% clean success with sub-millisecond p95 latency. Language conditioning added
capability at this scale, but it did not win the control benchmark.

## Repository map

```text
mouse_llm/
├── baselines/             action BC and numeric temporal planner
├── data/                  reconstruction, anchors, counterfactual skill labels
├── envs/mice/             BotEvade/Oasis Gymnasium environments + provenance
├── evaluation/            offline, constrained-decoding, and closed-loop eval
├── hierarchical/          semantic context, planners, skills, risk verification
├── training/              skill-LoRA and compact risk-critic training
├── northwestern/p2/       reproducible collection/training/ID/OOD Slurm jobs
├── reports/figures/       aggregate, Git-safe evidence only
├── tests/                 synthetic contracts and privacy checks
├── demo.py                dependency-light public evaluator demo
└── privacy_guard.py       rejects tracked data/checkpoints/weights
```

The detailed commands, storage layout, metric definitions, and limitations are
in [mouse_llm/README.md](mouse_llm/README.md).

## 中文摘要

MouseMind 是一个完整的 Cellworld policy 工程项目，而不只是一次 LoRA
微调。它覆盖私有轨迹审计、episode 防泄漏切分、MiniMind LoRA、MLP 公平
基线、受约束动作解码、seeded closed-loop 评估、延迟/控制周期分析、Slurm
和隐私 CI。

项目最重要的发现是：离线 imitation accuracy 无法预测 closed-loop control。
145K MLP 的离线准确率为 55.7%，但历史 closed-loop success 只有 17%；P1
层级策略将其提高到 74%。P2 使用 320 个 anchor、1,920 个 exact-replay
counterfactual branches 学习高层策略，并在未使用过的 100 个 final seeds 上
一次评估。Numeric planner 达到 100% task success、38% clean success；P1 为
79% / 14%，direct MLP 为 20% / 5%。MiniMind skill planner 达到 97% task
success，但 clean success 只有 12%，因此语言策略有能力增益，却没有胜过
numeric temporal planner。Risk critic 的离线 AUROC 为 0.959，但所有 runtime
threshold 都使 closed-loop 表现变差，所以没有被包装成“安全”功能上线。

## License

The repository retains MiniMind's Apache 2.0 license. Vendored Cellworld runtime
code retains its MIT notice in `mouse_llm/envs/mice/_vendor/LICENSE`.
