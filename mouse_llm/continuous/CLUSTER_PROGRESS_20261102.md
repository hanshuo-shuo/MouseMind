# Cluster continuation — 2026-11-02

Local-only candidate; no GitHub transfer or push. The original BCS516 cluster
is reached through the existing Quest SSH jump host. Code was transferred as a
hash-verified archive into an isolated run directory.

## Completed first stage

- Source commit: `d4d8c10`.
- Training job `31792`: completed, exit `0:0`, 2m01s, one A100 80GB.
- Cluster preflight: Torch 2.6.0+cu124, Transformers 4.57.6, 218 tests passed;
  one optional SB3 test skipped because that dependency is absent there.
- Both label modes used 60000 train / 5000 validation examples, the same
  five-component BC architecture, seed 42, at most 20 epochs and patience 5.
- Next-position selected epoch 17, validation mean XY error 0.03342.
- Commanded-destination selected epoch 20, validation mean XY error 0.03506.
- Dataset test split was not used to select either checkpoint.

## Four-seed engineering pilot

Job array `31793`, all six tasks completed. These are only development seeds
51000–51003; results do not establish an improvement or population success rate.
All rows used the original exact-XY/reject execution contract.

| Policy | Task | Clean | Captures/episode | Rejected destinations |
| --- | ---: | ---: | ---: | ---: |
| Historical DPO hierarchy | 4/4 | 1/4 | 3.00 | 0.0% |
| Historical discrete MLP | 3/4 | 0/4 | 70.50 | 0.0% |
| Continuous next-position BC | 4/4 | 0/4 | 2.50 | 11.9% |
| Continuous commanded-destination BC | 4/4 | 1/4 | 2.25 | 6.5% |
| Next-position BC + rule planner | 3/4 | 0/4 | 30.50 | 46.8% |
| Commanded-destination BC + rule planner | 3/4 | 0/4 | 30.50 | 46.7% |

The rule hierarchy exposes a concrete execution defect: continuous geometric
escape often requests points in obstacles or outside the hex arena, causing
the reject policy to hold. The continuous BC checkpoints are retained without
retraining while a common feasibility layer is tested on the same development
seeds. A valid request stays unchanged; an invalid request is explicitly mapped
to a continuous point in the current free-space component. This is not a return
to the 295-action catalog and is not evidence of a learned strategy improvement.

The project/reject v2 contract records original invalid requests, applied
destinations, projection rates/distances and final rejections. Nearest feasible
point projection alone does not guarantee body clearance or sensible evasion.
Closed-loop captures and stagnation remain the acceptance criteria.

Environment reuse and strict shard merging have also been implemented. They
must pass the compute-node fresh/reused A→B→A fingerprint check before being
used for the full 80-seed development comparison and counterfactual collection.

## Complete development comparison

Jobs `31802` and `31803` completed successfully on all 80 development seeds
51000–51079. Frozen execution uses `project`, target inset 0.01, the v2
navigation contract and verified environment reuse. The private contract SHA256
is `c577a3b8828ba19cb100f5f73b24fde946c3aeef60a05c2c45906b8537c09ece`.

| Policy | Task success | Capture-free success | Captures/episode |
| --- | ---: | ---: | ---: |
| Historical discrete MLP | 44/80 (55.00%) | 14/80 (17.50%) | 68.3375 |
| Historical DPO hierarchy | 75/80 (93.75%) | 11/80 (13.75%) | 9.2750 |
| Continuous next-position BC | 68/80 (85.00%) | 0/80 (0.00%) | 19.7250 |
| Continuous commanded-destination BC | 79/80 (98.75%) | 13/80 (16.25%) | 3.4625 |

Commanded-destination BC is frozen as the executor for the next stage. The
choice reflects task completion and capture burden; the two-episode clean
success difference from historical DPO does not establish a safety improvement.
The discrete MLP's higher clean point estimate accompanies many more failures
and captures. These are development-selection results, not final evidence.
BC/discrete evaluation used CPU and legacy DPO used GPU; the final comparison
must use a common backend. Latency must not be compared across those devices.

Both new BC variants have continuous XY outputs. The next-position variant
predicts the achieved state, whereas commanded-destination predicts the target
given to the simulator's navigation controller. Replacing commands with achieved
positions changes the controller's behavior; a smaller offline XY error did not
translate into better closed-loop outcomes here. Next-position BC remains a
recorded ablation rather than being silently discarded.

## Post-training data pilot

Source `1b597f7` adds an explicit frozen experiment contract and changes the
counterfactual CLI continuation to the same frozen BC after the candidate skill.
Job `31806` passed 71 focused tests, collected 32 training-pool seeds and
completed 32 anchors with zero replay rejections. It produced 56 training and
6 validation rows, plus 4 rows in each seen/unseen instruction test file.
These are pipeline diagnostics and are too small for formal planner training.
Job `31807` completed collection of all 200 data-collection seeds 50000–50199.

Before full label generation, inspect ties between GO and EVADE, per-preference
label retention, instruction/target distributions and coverage before/after
capture. EVADE can execute the same BC as GO without public predator memory;
unresolved ties can remove non-HOLD labels while explicit HOLD instructions
remain. Passing replay checks alone does not establish useful supervision.

## Full label review and first planner training

Job `31811` completed 320 anchors in 1h22m33s with zero replay rejections.
All 200 collection seeds are represented. After cross-split context filtering,
the dataset has 658 train rows from 167 seeds and 38 validation rows from
11 seeds; both contain all three target skills. The immutable dataset manifest
SHA256 is `9cc7943578429031b56f265d4da46f3d2d8708810f8aefe97421bf1fa08a4197`.

Training has 261 explicit HOLD-instruction rows. Among the other 397 rows,
119 target GO, 226 EVADE and 52 HOLD. The actual diagnostic found zero
`GO = EVADE > HOLD` cases; 97 anchors tie all three utility values. There is
therefore no evidence here for the hypothesized loss of superior tied GO/EVADE
pairs, and the conservative unique-winner labeling remains unchanged.

Validation coverage is limited: survival-first has only six rows and balanced
nine, neither with a GO target. Validation loss remains the fixed checkpoint
selection rule, but it cannot establish per-preference reliability. Closed-loop
development performance and skill counts must decide whether the planner helps.
Also, 71 anchors occur after an earlier capture; these labels concern future
outcomes and cannot restore capture-free success for that episode.

A hash-bound quality review permits the first fixed-configuration experiment.
Job `31812` runs three SFT epochs then one DPO epoch (beta 0.1), using the
new SFT adapter as both DPO initialization and frozen reference. Job `31813`
depends on its successful completion and evaluates direct BC, SFT and DPO
on all 80 development seeds using the same GPU backend. Training completion
alone is not evidence of improved control.

## First planner failure (same-GPU development)

Training job `31812` completed successfully in 3m22s. SFT made 126 optimizer
updates and its validation loss fell from 1.3411 to 0.7466. DPO made 42 updates
and finished at validation loss 0.5601, with the new SFT adapter as its verified
reference. Both checkpoints are backed up privately with verified hashes.

The completed SFT evaluation contradicts a performance claim from those losses:
all 80 episodes executed HOLD for all 300 steps, producing zero task success,
zero capture-free success, zero path length and 133.725 captures per episode.
Direct continuous BC on the same GPU backend reached 79/80 task success,
12/80 capture-free success and 3.6125 captures per episode. The earlier CPU
BC values are retained, but same-GPU values are the appropriate comparison.

The MiniMind parser falls back to GO, not HOLD, when parsing fails. However,
the continuous hierarchy does not propagate planner parsing validity into the
action-validity metric; `valid_action_rate=1` alone does not prove valid model
responses. Diagnose raw constrained responses, prompt lengths, finite logits
and complete-candidate scores before attributing the failure to data imbalance
or changing training. The DPO development evaluation is still in progress at
that snapshot. Final seeds remain unopened.

Job `31813` subsequently completed in 1h08m37s. DPO also produced zero task
and capture-free success, zero path length, 24,000 HOLD actions and 133.725
captures per episode. Both first-round greedy planners are development failures.

Read-only GPU diagnostic `31814` completed in 30s. For each checkpoint it
examined 24 stratified training contexts, 24 validation contexts and eight
collection-pool initial contexts. All responses were valid, with no prompt
truncation, template mismatch or non-finite logits. Greedy constrained decoding
selected HOLD on every sampled context. Full canonical JSON-plus-EOS sequence
log-probability sums gave different choices: SFT selected 15 EVADE / 4 GO /
5 HOLD on the 24 validation contexts, and DPO selected 20 EVADE / 4 HOLD.
These diagnostic subsets are not new independent performance evaluations.
Both scoring methods still selected HOLD on all eight initial contexts.

The decoder masks later tokens to a single continuation once a skill's first
branch token is chosen, whereas SFT/DPO optimize complete-response likelihood.
An opt-in complete-sequence-sum inference ablation is being implemented to
measure this mismatch. It reuses the same weights and data; it is not another
training run and does not erase either failed greedy evaluation. A small
development pilot must precede any claim about closed-loop behavior.

Not yet completed: new continuous-context MiniMind SFT/DPO, fresh 200-seed
final comparison, or promotion. The final seed pool remains unopened. All
weights, traces and per-episode records stay in the private run directory,
with local backups. No GitHub transfer or push has occurred.
