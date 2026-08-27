"""Fail when private mouse artifacts or model weights are tracked by Git."""

from __future__ import annotations

import argparse
import fnmatch
import subprocess
from pathlib import Path


BLOCKED_PATTERNS = (
    "mouse_data*.csv",
    "*.pth",
    "*.pt",
    "*.ckpt",
    "*.safetensors",
    "mouse_llm/artifacts/*",
    "mouse_llm/.local/*",
    "mouse_llm/reports/private/*",
    "dataset/mouse_*.jsonl",
    "closed_loop_episodes.csv",
    "failure_replay_manifest.json",
    "predictions_private.jsonl",
    "anchors_private.jsonl",
    "counterfactuals_private.jsonl",
    "planner_train.jsonl",
    "planner_validation.jsonl",
    "planner_seen_test.jsonl",
    "planner_unseen_test.jsonl",
    "risk_train.jsonl",
    "risk_validation.jsonl",
    "*private*.jsonl",
)


def blocked_tracked_files(repo_root: Path) -> list[str]:
    result = subprocess.run(
        ["git", "ls-files", "-z"],
        cwd=repo_root,
        check=True,
        capture_output=True,
    )
    tracked = [item.decode("utf-8") for item in result.stdout.split(b"\0") if item]
    blocked: list[str] = []
    for path in tracked:
        basename = Path(path).name
        if any(
            fnmatch.fnmatch(path, pattern) or fnmatch.fnmatch(basename, pattern)
            for pattern in BLOCKED_PATTERNS
        ):
            blocked.append(path)
    return sorted(blocked)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    args = parser.parse_args()
    blocked = blocked_tracked_files(args.repo_root.resolve())
    if blocked:
        formatted = "\n".join(f"- {path}" for path in blocked)
        raise SystemExit(f"Private artifacts are tracked by Git:\n{formatted}")
    print("Privacy guard passed: no private mouse data or model weights are tracked.")


if __name__ == "__main__":
    main()
