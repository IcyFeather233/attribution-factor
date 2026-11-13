"""Replay a frozen mini-swe-agent scaffold on the fixed failure set."""

from __future__ import annotations

import argparse
import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import yaml

from .dataset import load_failed_instances
from .model_config import TASK_STEP_LIMIT, task_api_key
from .runner import run_task


WORKSPACE = Path(__file__).resolve().parents[2]
UDOCKER_REPO = WORKSPACE / "native_baseline_runs" / "udocker"
UDOCKER_EXECUTABLE = WORKSPACE / "native_baseline_runs" / ".venv" / "bin" / "udocker"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scaffold", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--step-limit", type=int, default=TASK_STEP_LIMIT)
    args = parser.parse_args()

    api_key = task_api_key()
    scaffold = yaml.safe_load(args.scaffold.read_text()) or {}
    instances = load_failed_instances()
    if args.limit:
        instances = instances[: args.limit]

    def replay(instance: dict) -> dict:
        return run_task(
            instance=instance,
            output_dir=args.output_dir / instance["instance_id"],
            udocker_repo=UDOCKER_REPO,
            udocker_executable=UDOCKER_EXECUTABLE,
            api_key=api_key,
            agent_overrides=scaffold,
            step_limit=args.step_limit,
        )

    by_id: dict[str, dict] = {}
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(replay, item): item["instance_id"] for item in instances}
        for future in as_completed(futures):
            task_id = futures[future]
            try:
                by_id[task_id] = future.result()
            except Exception as exc:
                by_id[task_id] = {"instance_id": task_id, "error": str(exc)}

    results = []
    for instance in instances:
        run = by_id[instance["instance_id"]]
        evaluation = run.get("evaluation", {})
        results.append(
            {
                "instance_id": instance["instance_id"],
                "resolved": bool(evaluation.get("resolved")),
                "status": evaluation.get(
                    "error", "resolved" if evaluation.get("resolved") else "unresolved"
                ),
            }
        )
    summary = {
        "resolved": sum(item["resolved"] for item in results),
        "total": len(results),
        "results": results,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


