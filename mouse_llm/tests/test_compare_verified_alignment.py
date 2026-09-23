"""Selection must use development seeds, never the final evaluation pool."""

import csv
import hashlib
import json

from mouse_llm.evaluation.compare_verified_alignment import compare_runs


def _pool(start, stop):
    seeds = list(range(start, stop))
    return {
        "start": start,
        "stop_exclusive": stop,
        "count": len(seeds),
        "sha256": hashlib.sha256(",".join(map(str, seeds)).encode()).hexdigest(),
    }


def _run(path, *, pool, seeds, captures):
    path.mkdir()
    metadata = {
        "contract_sha256": "same",
        "seed_pool": pool,
        "seed_sha256": "same",
        "episode_count": len(seeds),
        "planner_horizon": 8,
        "p1_planner_horizon": 4,
        "risk_threshold": 0.9,
        "ood_condition": "id",
        "preference": "survival_first",
        "instruction_split": "train",
        "environment_parameters": {"predator_prey_forward_speed_ratio": 0.15},
    }
    (path / "closed_loop_metrics.json").write_text(json.dumps({"metadata": metadata}))
    with (path / "closed_loop_episodes.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=("policy", "seed", "success", "clean_success", "captured", "captures"))
        writer.writeheader()
        for seed in seeds:
            writer.writerow({"policy": "minimind-learned", "seed": seed, "success": 1, "clean_success": int(captures == 0), "captured": int(captures > 0), "captures": captures})


def test_only_development_can_mark_a_candidate(tmp_path):
    contract = tmp_path / "contract.json"
    contract.write_text(json.dumps({
        "schema_version": 1,
        "seed_pools": {
            "development": _pool(1, 3),
            "final_id_test": _pool(3, 5),
        },
    }))
    for pool, seeds in (("development", (1, 2)), ("final_id_test", (3, 4))):
        baseline = tmp_path / f"{pool}_baseline"
        candidate = tmp_path / f"{pool}_candidate"
        _run(baseline, pool=pool, seeds=seeds, captures=2)
        _run(candidate, pool=pool, seeds=seeds, captures=0)
        result = compare_runs(baseline, {"better": candidate}, seed_pool=pool, contract=contract)
        expected = True if pool == "development" else None
        assert result["candidates"]["better"]["development_candidate"] is expected
