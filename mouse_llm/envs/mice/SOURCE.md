# Mice environment provenance

This directory contains only the environment runtime extracted from
[`hanshuo-shuo/Mice`](https://github.com/hanshuo-shuo/Mice), not its SAC
trainer, experiment logs, render demos, or configuration generators.

- Source branch: `Position_control_navigation_env`
- Source commit: `e6b2984d58949b20aa8b20eb706dbb22acd1b5ed`
- Extracted: `botevade_gym.py`, `oasis_gym.py`, `reward.py`, `util.py`, the
  `cellworld_game` runtime, and the three geometry-index arrays required by
  `BotEvadeEnv`.
- Local packaging changes: relative package imports, package-relative geometry
  assets, removal of import-time console output, Gymnasium seed initialization,
  task-owned BotEvade RNG seeding, distinct success/survival terminal metadata,
  explicit use of the cached `astar.robot` navigation resource, and pickle-free
  loading for numeric geometry arrays.

The bundled `cellworld_game` runtime retains its upstream MIT license in
[`_vendor/LICENSE`](_vendor/LICENSE). The environment adapters originate from
the user's own Mice repository. Cellworld world caches are deliberately not
vendored here; set `CELLWORLD_CACHE` to a private/shared cache directory.

## Recovered legacy policy observation

The 10D dataset contract is independently anchored to
`hanshuo-shuo/Mice@67e769fd410b325b5d2c517d9d5966e5e80fac23:env.py`
(SHA256 `775d672ecfb518584005558a2ae635981bdacac29b68ed2fba754579e79bee3f`).
That source predates the September 2025 transition export and defines the exact
field order and signed-angle convention. See
[`../../reports/observation_contract_audit.json`](../../reports/observation_contract_audit.json)
for source integrity, full-dataset distribution, and simulator state-replay
evidence.
