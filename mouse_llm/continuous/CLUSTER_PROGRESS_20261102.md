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

Not yet completed: full development comparison, new continuous-context
MiniMind SFT/DPO, fresh 200-seed final comparison, or promotion. The final seed
pool remains unopened. All weights, traces and per-episode records stay in the
private run directory, with local backups.
