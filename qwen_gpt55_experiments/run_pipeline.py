"""Wait for original baselines, then run six ablation groups with bounded parallelism."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
import sys
from datetime import datetime, timezone
from pathlib import Path
from qwen_gpt55_experiments.common.udocker_runtime import persistent_environment

from qwen_gpt55_experiments.common.analysis_payloads import GROUPS, GROUP_BASELINE
from qwen_gpt55_experiments.common.model_config import (
    OPTIMIZER_MODEL,
    OPTIMIZER_BASE_URL,
    TASK_MODEL,
    TASK_BASE_URL,
    TASK_API_KEY_FILE,
    TASK_API_KEY_SOURCE,
    TASK_STEP_LIMIT,
    task_api_key,
)


ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "qwen_gpt55_experiments"
PYTHON = Path(sys.executable)
MINI_SOURCE = ROOT / "qwen_failure_comparison" / "mini-swe-agent-v2.4.6" / "src"
SIDECAR_ENV = Path("/volume/med/work/users/mzchen/skills/llm-sidecar/.env")
LAUNCH_ROOT = PACKAGE / "parallel_launch"
BASELINE_MANIFEST = LAUNCH_ROOT / "baseline_manifest.json"
ABLATION_MANIFEST = LAUNCH_ROOT / "ablation_manifest.json"
PIPELINE_STATUS = LAUNCH_ROOT / "pipeline_status.json"
MANAGER_MANIFEST = LAUNCH_ROOT / "pipeline_manager.json"
MANAGER_LOG = LAUNCH_ROOT / "pipeline_manager.log"
SUMMARY_NAMES = {
    "gepa": "final_summary.json",
    "scope": "summary.json",
    "recreate": "summary.json",
    "meta_harness": "summary.json",
    "harnessfix": "summary.json",
}
METHODS = {
    "gepa": "qwen_gpt55_experiments.gepa.run_native",
    "scope": "qwen_gpt55_experiments.scope.run_native",
    "recreate": "qwen_gpt55_experiments.recreate.run_native",
    "meta_harness": "qwen_gpt55_experiments.meta_harness.run_native",
    "harnessfix": "qwen_gpt55_experiments.harnessfix.native_profile",
}
ABLATION_GROUP_CONCURRENCY = 3


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--detach", action="store_true")
    parser.add_argument("--supervise", action="store_true")
    parser.add_argument("--poll-seconds", type=int, default=60)
    return parser.parse_args()


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
    temporary.replace(path)


def load_sidecar_key() -> str:
    if os.environ.get("SIDECAR_API_KEY"):
        return os.environ["SIDECAR_API_KEY"]
    for line in SIDECAR_ENV.read_text().splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        name, value = stripped.split("=", 1)
        if name.strip() == "KEY":
            key = value.strip().strip(chr(34)).strip(chr(39))
            if key:
                return key
    raise RuntimeError(f"KEY is missing or empty in {SIDECAR_ENV}")


def run_dir(method: str, group: str) -> Path:
    return PACKAGE / method / group / "native_round1"


def summary_path(method: str, group: str) -> Path:
    return run_dir(method, group) / SUMMARY_NAMES[method]


def process_is_running(job: dict) -> bool:
    pid = int(job["pid"])
    stat_path = Path(f"/proc/{pid}/stat")
    cmdline_path = Path(f"/proc/{pid}/cmdline")
    if not stat_path.is_file() or not cmdline_path.is_file():
        return False
    if stat_path.read_text().split()[2] == "Z":
        return False
    expected_module = str(job["command"][2])
    command_line = cmdline_path.read_bytes().replace(b"\0", b" ").decode(errors="replace")
    return expected_module in command_line


def baseline_state() -> tuple[list[str], list[str], list[str]]:
    manifest = json.loads(BASELINE_MANIFEST.read_text())
    complete = []
    running = []
    failed = []
    for job in manifest["jobs"]:
        name = job["job"]
        if summary_path(job["method"], GROUP_BASELINE).is_file():
            complete.append(name)
        elif process_is_running(job):
            running.append(name)
        else:
            failed.append(name)
    return complete, running, failed


def child_environment(optimizer_key: str, task_key: str) -> dict[str, str]:
    inherited = os.environ.get("PYTHONPATH", "")
    pythonpath = str(MINI_SOURCE)
    if inherited:
        pythonpath += os.pathsep + inherited
    return {
        **os.environ,
        "SIDECAR_API_KEY": optimizer_key,
        "QWEN_TASK_API_KEY": task_key,
        "PYTHONPATH": pythonpath,
        "PYTHONUNBUFFERED": "1",
        "QWEN_MIN_REQUEST_INTERVAL_SECONDS": os.environ.get("QWEN_MIN_REQUEST_INTERVAL_SECONDS", "0.5"),
        "QWEN_TPM_LIMIT": os.environ.get("QWEN_TPM_LIMIT", "10000000"),
        "QWEN_TPM_UTILIZATION": os.environ.get("QWEN_TPM_UTILIZATION", "0.9"),
        "QWEN_TOKEN_RESERVATION": os.environ.get("QWEN_TOKEN_RESERVATION", "35000"),
        "QWEN_MAX_IN_FLIGHT": os.environ.get("QWEN_MAX_IN_FLIGHT", "192"),
        "QWEN_SLOT_LEASE_SECONDS": os.environ.get("QWEN_SLOT_LEASE_SECONDS", "900"),
        "MSWEA_MODEL_RETRY_STOP_AFTER_ATTEMPT": os.environ.get("MSWEA_MODEL_RETRY_STOP_AFTER_ATTEMPT", "10"),
        **persistent_environment(),
        "UDOCKER_CREATE_CONCURRENCY": os.environ.get("UDOCKER_CREATE_CONCURRENCY", "2"),
        "UDOCKER_MAX_CONTAINERS": os.environ.get("UDOCKER_MAX_CONTAINERS", "16"),
        "UDOCKER_MAX_EVALUATORS": os.environ.get("UDOCKER_MAX_EVALUATORS", "4"),
    }


def command(method: str, group: str) -> list[str]:
    result = [
        str(PYTHON),
        "-m",
        METHODS[method],
        "--ablation-group",
        group,
    ]
    if method == "gepa":
        result.extend([
            "--adaptation-tasks", "148",
            "--full-trace-pass", "--max-candidate-proposals", "10",
            "--reflection-batch-size", "15",
            "--adaptation-seed", "20260902",
            "--evolution-workers", "4",
            "--final-workers", "32",
        ])
    elif method == "scope":
        result.extend([
            "--adaptation-tasks", "148",
            "--adaptation-seed", "20260902",
            "--adaptation-batch-size", "15",
            "--adaptation-workers", "8",
            "--quality-analysis-frequency", "4",
            "--final-workers", "32",
        ])
    elif method == "meta_harness":
        result.extend([
            "--adaptation-tasks", "148",
            "--adaptation-seed", "20260902",
            "--benchmark-workers", "32",
        ])
    elif method == "harnessfix":
        result.extend([
            "--max-iterations", "1",
            "--max-promotion-failures", "2",
            "--analysis-tasks", "148",
            "--validation-tasks", "32",
            "--analysis-workers", "16",
            "--validation-workers", "24",
            "--final-workers", "24",
            "--validation-seed", "20260902",
        ])
    elif method == "recreate":
        result.extend([
            "--adaptation-tasks", "148",
            "--adaptation-batch-size", "15",
            "--adaptation-repeats", "1",
            "--adaptation-workers", "16",
            "--final-workers", "24",
            "--adaptation-seed", "20260902",
        ])
    return result


def wait_for_baselines(poll_seconds: int) -> bool:
    while True:
        complete, running, failed = baseline_state()
        write_json(PIPELINE_STATUS, {
            "updated_at": datetime.now(timezone.utc).isoformat(),
            "state": "waiting_for_baselines",
            "baseline_complete": complete,
            "baseline_running": running,
            "baseline_failed": failed,
            "ablation_groups_complete": [],
            "current_ablation_group": None,
        })
        if failed:
            return False
        if len(complete) == len(METHODS):
            return True
        time.sleep(poll_seconds)


def write_ablation_manifest(jobs: list[dict], state: str) -> None:
    write_json(ABLATION_MANIFEST, {
        "schema_version": "2.0",
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "phase": "ablation",
        "state": state,
        "scheduling": "three attribution groups at a time; five baselines per group",
        "ablation_group_concurrency": ABLATION_GROUP_CONCURRENCY,
        "planned_job_count": (len(GROUPS) - 1) * len(METHODS),
        "launched_job_count": len(jobs),
        "task_count_per_job": 148,
        "task_model": TASK_MODEL,
        "task_step_limit": TASK_STEP_LIMIT,
        "task_base_url": TASK_BASE_URL,
        "task_key_source": TASK_API_KEY_SOURCE,
        "task_key_recorded": False,
        "optimizer_model": OPTIMIZER_MODEL,
        "optimizer_base_url": OPTIMIZER_BASE_URL,
        "sidecar_key_source": str(SIDECAR_ENV),
        "sidecar_key_recorded": False,
        "jobs": jobs,
    })


def run_ablation_groups(poll_seconds: int) -> bool:
    groups = [group for group in GROUPS if group != GROUP_BASELINE]
    occupied = [
        str(run_dir(method, group))
        for group in groups
        for method in METHODS
        if run_dir(method, group).exists()
    ]
    if occupied:
        raise FileExistsError(
            "Refusing to overwrite existing formal ablation directories:\n"
            + "\n".join(occupied)
        )
    if ABLATION_MANIFEST.exists():
        raise FileExistsError(f"Ablation manifest already exists: {ABLATION_MANIFEST}")

    optimizer_key = load_sidecar_key()
    task_key = task_api_key()
    env = child_environment(optimizer_key, task_key)
    all_jobs: list[dict] = []
    completed_groups: list[str] = []
    logs_root = LAUNCH_ROOT / "ablation_logs"
    logs_root.mkdir(parents=True, exist_ok=True)

    for wave_start in range(0, len(groups), ABLATION_GROUP_CONCURRENCY):
        wave_groups = groups[
            wave_start : wave_start + ABLATION_GROUP_CONCURRENCY
        ]
        wave_jobs = []
        for group in wave_groups:
            group_logs = logs_root / group
            group_logs.mkdir()
            for method, module in METHODS.items():
                cmd = command(method, group)
                log_path = group_logs / f"{method}.log"
                with log_path.open("w") as log:
                    process = subprocess.Popen(
                        cmd,
                        cwd=ROOT,
                        env=env,
                        stdin=subprocess.DEVNULL,
                        stdout=log,
                        stderr=subprocess.STDOUT,
                        start_new_session=True,
                    )
                job = {
                    "job": f"{method}__{group}",
                    "method": method,
                    "ablation_group": group,
                    "pid": process.pid,
                    "command": cmd,
                    "log": str(log_path),
                    "run_dir": str(run_dir(method, group)),
                }
                wave_jobs.append(job)
                all_jobs.append(job)
        write_ablation_manifest(all_jobs, "running")

        while True:
            complete = [
                job["job"]
                for job in wave_jobs
                if summary_path(
                    job["method"], job["ablation_group"]
                ).is_file()
            ]
            failed = [
                job["job"]
                for job in wave_jobs
                if job["job"] not in complete and not process_is_running(job)
            ]
            running = [
                job["job"]
                for job in wave_jobs
                if job["job"] not in complete and job["job"] not in failed
            ]
            wave_complete_groups = [
                group for group in wave_groups
                if all(
                    summary_path(method, group).is_file()
                    for method in METHODS
                )
            ]
            write_json(PIPELINE_STATUS, {
                "updated_at": datetime.now(timezone.utc).isoformat(),
                "state": "running_ablations",
                "baseline_complete": [f"{method}__baseline" for method in METHODS],
                "baseline_running": [],
                "baseline_failed": [],
                "ablation_groups_complete": [
                    *completed_groups,
                    *wave_complete_groups,
                ],
                "current_ablation_group": None,
                "current_ablation_groups": wave_groups,
                "current_complete": complete,
                "current_running": running,
                "current_failed": failed,
            })
            if failed:
                write_ablation_manifest(all_jobs, "failed")
                return False
            if len(complete) == len(wave_jobs):
                completed_groups.extend(wave_groups)
                break
            time.sleep(poll_seconds)

    write_ablation_manifest(all_jobs, "complete")
    write_json(PIPELINE_STATUS, {
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "state": "complete",
        "baseline_complete": [f"{method}__baseline" for method in METHODS],
        "baseline_running": [],
        "baseline_failed": [],
        "ablation_groups_complete": completed_groups,
        "current_ablation_group": None,
        "current_ablation_groups": [],
    })
    return True


def supervise(poll_seconds: int) -> int:
    if not BASELINE_MANIFEST.is_file():
        raise FileNotFoundError(f"Missing baseline manifest: {BASELINE_MANIFEST}")
    if not wait_for_baselines(poll_seconds):
        return 2
    return 0 if run_ablation_groups(poll_seconds) else 3


def detach(poll_seconds: int) -> None:
    if MANAGER_MANIFEST.is_file():
        previous = json.loads(MANAGER_MANIFEST.read_text())
        previous_job = {
            "pid": previous["pid"],
            "command": previous["command"],
        }
        if process_is_running(previous_job):
            raise RuntimeError("Pipeline manager is already running with PID {}".format(previous["pid"]))
    LAUNCH_ROOT.mkdir(parents=True, exist_ok=True)
    command_line = [
        str(PYTHON),
        "-m",
        "qwen_gpt55_experiments.run_pipeline",
        "--supervise",
        "--poll-seconds",
        str(poll_seconds),
    ]
    log = MANAGER_LOG.open("a")
    process = subprocess.Popen(
        command_line,
        cwd=ROOT,
        env={**os.environ, **persistent_environment()},
        stdin=subprocess.DEVNULL,
        stdout=log,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    write_json(MANAGER_MANIFEST, {
        "launched_at": datetime.now(timezone.utc).isoformat(),
        "pid": process.pid,
        "command": command_line,
        "log": str(MANAGER_LOG),
        "sidecar_key_recorded": False,
    })
    print(json.dumps({
        "status": "pipeline_manager_launched",
        "pid": process.pid,
        "log": str(MANAGER_LOG),
    }, indent=2))


def main() -> None:
    args = parse_args()
    if args.poll_seconds < 10:
        raise ValueError("poll-seconds must be at least 10")
    if args.detach == args.supervise:
        raise ValueError("Choose exactly one of --detach or --supervise")
    if args.detach:
        detach(args.poll_seconds)
        return
    raise SystemExit(supervise(args.poll_seconds))


if __name__ == "__main__":
    main()
