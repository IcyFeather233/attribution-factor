"""Resume patch-transport regrading without repeating task-agent rollouts."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys
import time

import requests
from qwen_gpt55_experiments.common.dataset import load_failed_instances
from qwen_gpt55_experiments.common.evaluator import evaluate_patch, normalize_patch_transport
from qwen_gpt55_experiments.common.result_validity import retryable_result
from qwen_gpt55_experiments.meta_harness.run_native import RUN_ROOT, UDOCKER_EXECUTABLE, UDOCKER_REPO


def atomic_json(path: Path, value: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--ablation-group", required=True)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    run_dir = RUN_ROOT / args.ablation_group / args.run_name
    benchmark = run_dir / "candidate_benchmark"
    archive = run_dir / "evaluation_before_patch_transport_fix"
    archive.mkdir(exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    progress_path = run_dir / ("evaluation_recovery_" + stamp + ".json")
    instances = {item["instance_id"]: item for item in load_failed_instances()}
    # SWE-bench's environment-spec builder may fetch requirements from GitHub.
    # A transient read timeout must not discard all other recovered evaluations.
    original_request = requests.Session.request
    def bounded_request(self, method, url, **kwargs):
        if kwargs.get("timeout") is None:
            kwargs["timeout"] = 60
        return original_request(self, method, url, **kwargs)
    requests.Session.request = bounded_request
    targets = []
    for directory in sorted(benchmark.iterdir()):
        result_path, patch_path = directory / "result.json", directory / "patch.diff"
        if not result_path.is_file() or not patch_path.is_file():
            continue
        result = json.loads(result_path.read_text())
        evaluation = result.get("evaluation", {})
        patch = patch_path.read_text()
        transport_failure = evaluation.get("error") == "patch_apply_failed" and evaluation.get("patch_apply_returncode") == 128
        if transport_failure and normalize_patch_transport(patch).startswith("diff --git "):
            targets.append(directory)
    summary = run_dir / "summary.json"
    if targets and summary.exists():
        summary.rename(run_dir / ("summary.before_recovery_" + stamp + ".json"))
    state = {"state": "running", "started_at": stamp, "target_count": len(targets),
             "completed_count": 0, "resolved_count": 0, "task_ids": [p.name for p in targets], "failures": {}}
    atomic_json(progress_path, state)

    def one(directory):
        result_path = directory / "result.json"
        result = json.loads(result_path.read_text())
        current, archived = directory / "evaluation", archive / directory.name
        if current.exists():
            destination = archived if not archived.exists() else directory / ("evaluation.before_" + stamp)
            current.rename(destination)
        for attempt in range(3):
            pending = directory / f"evaluation.recovery_{stamp}_{attempt}"
            try:
                evaluation = evaluate_patch(instance=instances[directory.name], patch=(directory / "patch.diff").read_text(),
                                            output_dir=pending, udocker_repo=UDOCKER_REPO, udocker_executable=UDOCKER_EXECUTABLE)
                if not isinstance(evaluation.get("resolved"), bool) or retryable_result({"evaluation": evaluation}):
                    raise RuntimeError("Non-gradeable recovery evaluation")
                pending.rename(current)
                result.update(evaluation=evaluation, evaluation_recovered_from_returncode_128=True,
                              evaluation_recovery_attempt=stamp, evaluation_archive=str(archived.relative_to(run_dir)))
                atomic_json(result_path, result)
                return bool(evaluation["resolved"])
            except Exception:
                if attempt == 2:
                    raise
                time.sleep(2)
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(one, path): path.name for path in targets}
        for future in as_completed(futures):
            iid = futures[future]
            try:
                state["resolved_count"] += int(future.result())
                state["completed_count"] += 1
            except Exception as exc:
                state["failures"][iid] = type(exc).__name__ + ": " + str(exc)
            atomic_json(progress_path, state)
    if state["failures"]:
        state["state"] = "needs_attention"
        atomic_json(progress_path, state)
        raise RuntimeError(f"Recovery incomplete; see {progress_path}")
    # Rebuild the summary from graded artifacts only; never silently rerun agents.
    command = [sys.executable, "-B", "-m", "qwen_gpt55_experiments.meta_harness.resume_native",
               "--ablation-group", args.ablation_group, "--run-name", args.run_name,
               "--adaptation-tasks", "32", "--adaptation-seed", "20260902", "--workers", str(args.workers), "--offline-only"]
    subprocess.run(command, check=True)
    state["state"] = "complete"
    atomic_json(progress_path, state)
    atomic_json(run_dir / "evaluation_recovery.json", state)


if __name__ == "__main__":
    main()
