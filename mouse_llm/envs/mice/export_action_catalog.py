"""Export the Cellworld discrete-action destinations used by mouse datasets."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ._vendor import cellworld_game as cwgame


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--world", default="21_05")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    loader = cwgame.CellWorldLoader(world_name=args.world)
    actions = [
        {"action": index, "destination": [float(point[0]), float(point[1])]}
        for index, point in enumerate(loader.full_action_list)
    ]
    payload = {
        "schema_version": 1,
        "world": args.world,
        "action_count": len(actions),
        "actions": actions,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"Wrote {len(actions)} actions to {args.output}")


if __name__ == "__main__":
    main()
