"""Load the fixed 148-task confirmed Qwen failure set."""

from __future__ import annotations

import random
import os
from pathlib import Path

import pyarrow.parquet as pq


WORKSPACE = Path(__file__).resolve().parents[2]
DEFAULT_PARQUET = Path(os.environ.get("ATTRIBUTION_PARQUET", WORKSPACE / "data" / "tasks.parquet"))
DEFAULT_FAILED_TASKS = Path(os.environ.get("ATTRIBUTION_TASK_IDS", WORKSPACE / "qwen_gpt55_experiments" / "task_ids.txt"))


def load_failed_task_ids(path: Path = DEFAULT_FAILED_TASKS) -> list[str]:
    task_ids = [line.strip() for line in path.read_text().splitlines() if line.strip()]
    if not task_ids or len(set(task_ids)) != len(task_ids):
        raise ValueError(f"Expected a nonempty list of unique failures, found {len(task_ids)}")
    return task_ids


def load_failed_instances(
    parquet_path: Path = DEFAULT_PARQUET,
    task_ids_path: Path = DEFAULT_FAILED_TASKS,
) -> list[dict]:
    task_ids = load_failed_task_ids(task_ids_path)
    table = pq.read_table(parquet_path, filters=[("instance_id", "in", task_ids)])
    by_id = {row["instance_id"]: row for row in table.to_pylist()}
    missing = [task_id for task_id in task_ids if task_id not in by_id]
    if missing:
        raise KeyError(f"Tasks missing from SWE-bench Verified parquet: {missing}")
    return [by_id[task_id] for task_id in task_ids]


def image_for_instance(instance: dict) -> str:
    image_id = instance["instance_id"].replace("__", "_1776_").lower()
    return f"docker.io/swebench/sweb.eval.x86_64.{image_id}:latest"



def select_repository_balanced_instances(
    instances: list[dict], *, limit: int, seed: int
) -> list[dict]:
    """Select a deterministic repository-balanced subset."""
    if limit < 1 or limit > len(instances):
        raise ValueError(f"limit must be between 1 and {len(instances)}")
    if limit == len(instances):
        return list(instances)

    by_repo: dict[str, list[dict]] = {}
    for instance in instances:
        by_repo.setdefault(instance["repo"], []).append(instance)

    rng = random.Random(seed)
    repositories = sorted(by_repo)
    rng.shuffle(repositories)
    for repository in repositories:
        rng.shuffle(by_repo[repository])

    selected = []
    offsets = {repository: 0 for repository in repositories}
    while len(selected) < limit:
        made_progress = False
        for repository in repositories:
            offset = offsets[repository]
            candidates = by_repo[repository]
            if offset >= len(candidates):
                continue
            selected.append(candidates[offset])
            offsets[repository] = offset + 1
            made_progress = True
            if len(selected) == limit:
                break
        if not made_progress:
            break
    if len(selected) != limit:
        raise RuntimeError(f"Could only select {len(selected)} of {limit} tasks")
    return selected

