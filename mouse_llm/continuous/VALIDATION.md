# Local implementation verification — 2026-11-02

Status: implementation candidate, no new policy-performance result.

The complete repository suite passed **219 tests**, including the previously
optional real-environment tests, with the existing Mice-BotEvade Python and
local offline Cellworld cache. No test was skipped in this run.

```bash
PYTHONDONTWRITEBYTECODE=1 MPLBACKEND=Agg \
PYGAME_HIDE_SUPPORT_PROMPT=1 SDL_AUDIODRIVER=dummy SDL_VIDEODRIVER=dummy \
CELLWORLD_CACHE=/path/to/existing/cellworld_cache \
python -B -m pytest mouse_llm/tests -q -p no:cacheprovider
```

The tests cover:

- label semantics, episode isolation, preservation of within-split multimodal
  targets, current-input-only split filtering, and private-output protection;
- mixture-density loss, bounded component-mode inference, checkpoint contracts,
  train-only normalization and tiny synthetic training/save/load;
- actual off-center terminal waypoints, geometry rejection, hidden-predator
  memory, fresh seeded reset and unchanged predator navigation;
- public trace → physical/controller/RNG replay fingerprint → repeated skill
  branches in the cached simulator;
- shared SFT/DPO utility ordering, ambiguous-label filtering and forbidden
  development/final seeds;
- SFT/DPO data validation and checkpoint/executor lineage without loading real
  language-model weights;
- SAC actor warmstart/API checks, real-online-only replay and teacher metadata;
- complete paired-final evidence validation and promotion decisions on
  synthetic fixtures (these fixtures are not scientific evidence).

A real-engine integration test exposed a `3.58e-7` discrepancy when recovering
signed legacy angles from the rounded 15D observation. Trace collection now
records the simulator's canonical public legacy encoder before stepping.
The `1e-7` replay tolerance was not relaxed to hide the discrepancy.

Both private continuous datasets were generated from the unchanged audited
CSV, with **60000 / 5000 / 5000** train/validation/test rows for each label mode.
The source contains 118861 transitions and 4916 reconstructed episodes.
The manifest includes achieved-versus-commanded differences and displacement
dead-zone statistics; no source rows or model weights were copied into Git.

Shell syntax validation and `git diff --check` passed. The historical reports
remain unchanged. The new readme banner explicitly separates the candidate
from historical discrete-controller results.

Not yet executed: real-data continuous BC training, whole-policy SAC training,
continuous-context MiniMind SFT/DPO, OOD evaluation, or the new 200-seed final
comparison. No improvement in clean success, completion or captures is claimed.
The repeated-seed integration checks are engineering checks, not trained-policy
results. Reusing old TD-MPC2 weights still requires a separately validated
observation/action adapter.
