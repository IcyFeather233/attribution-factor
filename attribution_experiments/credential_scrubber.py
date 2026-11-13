"""Continuously remove the sidecar key from in-progress formal experiment artifacts."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from attribution_experiments.common.udocker_runtime import persistent_environment

from attribution_experiments.common.analysis_payloads import GROUPS
from attribution_experiments.common.model_config import task_api_key, optimizer_api_key


ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "attribution_experiments"
PYTHON = ROOT / "native_baseline_runs" / ".venv" / "bin" / "python"
LAUNCH_ROOT = PACKAGE / "parallel_launch"
PIPELINE_STATUS = LAUNCH_ROOT / "pipeline_status.json"
SCRUBBER_STATUS = LAUNCH_ROOT / "credential_scrubber_status.json"
SCRUBBER_MANIFEST = LAUNCH_ROOT / "credential_scrubber.json"
SCRUBBER_LOG = LAUNCH_ROOT / "credential_scrubber.log"
METHODS = ("gepa", "scope", "recreate", "meta_harness", "harnessfix")
SUFFIXES = {".json", ".jsonl", ".txt", ".log", ".yaml", ".yml"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--detach", action="store_true")
    parser.add_argument("--supervise", action="store_true")
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--poll-seconds", type=int, default=20)
    return parser.parse_args()


def write_json(path: Path, value: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
    temporary.replace(path)


def load_keys() -> tuple[str, str]:
    return optimizer_api_key(), task_api_key()


def formal_roots() -> list[Path]:
    return [
        PACKAGE / method / group / "native_round1"
        for method in METHODS
        for group in GROUPS
        if (PACKAGE / method / group / "native_round1").exists()
    ]


def iter_files(root: Path):
    """Yield files while tolerating temporary directories disappearing mid-scan."""
    for directory, _, filenames in os.walk(root, onerror=lambda _: None):
        parent = Path(directory)
        for filename in filenames:
            yield parent / filename


def scrub_once(secrets: tuple[str, ...]) -> dict:
    secrets = tuple(secret for secret in secrets if secret and secret != "EMPTY")
    files_scanned = 0
    files_rewritten = 0
    occurrences_removed = 0
    for root in formal_roots():
        for path in iter_files(root):
            if not path.is_file() or path.suffix not in SUFFIXES:
                continue
            files_scanned += 1
            try:
                content = path.read_text()
            except (OSError, UnicodeDecodeError):
                continue
            updated = content
            removed = 0
            for secret in secrets:
                removed += updated.count(secret)
                updated = updated.replace(secret, "<REDACTED>")
            if removed:
                path.write_text(updated)
                files_rewritten += 1
                occurrences_removed += removed
    result = {
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "files_scanned": files_scanned,
        "files_rewritten": files_rewritten,
        "occurrences_removed": occurrences_removed,
        "api_keys_recorded": False,
    }
    write_json(SCRUBBER_STATUS, result)
    return result


def pipeline_is_terminal() -> bool:
    if not PIPELINE_STATUS.is_file():
        return False
    status = json.loads(PIPELINE_STATUS.read_text())
    return (
        status.get("state") == "complete"
        or bool(status.get("baseline_failed"))
        or bool(status.get("current_failed"))
    )


def supervise(poll_seconds: int) -> None:
    secrets = load_keys()
    while True:
        result = scrub_once(secrets)
        print(json.dumps(result), flush=True)
        if pipeline_is_terminal():
            return
        time.sleep(poll_seconds)


def process_is_running(pid: int) -> bool:
    stat = Path(f"/proc/{pid}/stat")
    return stat.is_file() and stat.read_text().split()[2] != "Z"


def detach(poll_seconds: int) -> None:
    if SCRUBBER_MANIFEST.is_file():
        previous = json.loads(SCRUBBER_MANIFEST.read_text())
        if process_is_running(int(previous["pid"])):
            raise RuntimeError(
                "Credential scrubber is already running with PID {}".format(previous["pid"])
            )
    command = [
        str(PYTHON),
        "-m",
        "attribution_experiments.credential_scrubber",
        "--supervise",
        "--poll-seconds",
        str(poll_seconds),
    ]
    log = SCRUBBER_LOG.open("a")
    process = subprocess.Popen(
        command,
        cwd=ROOT,
        env={**os.environ, **persistent_environment()},
        stdin=subprocess.DEVNULL,
        stdout=log,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    write_json(SCRUBBER_MANIFEST, {
        "launched_at": datetime.now(timezone.utc).isoformat(),
        "pid": process.pid,
        "command": command,
        "log": str(SCRUBBER_LOG),
        "api_keys_recorded": False,
    })
    print(json.dumps({
        "status": "credential_scrubber_launched",
        "pid": process.pid,
        "log": str(SCRUBBER_LOG),
    }, indent=2))


def main() -> None:
    args = parse_args()
    if args.poll_seconds < 10:
        raise ValueError("poll-seconds must be at least 10")
    selected = sum((args.detach, args.supervise, args.once))
    if selected != 1:
        raise ValueError("Choose exactly one of --detach, --supervise, or --once")
    if args.once:
        print(json.dumps(scrub_once(load_keys()), indent=2))
    elif args.detach:
        detach(args.poll_seconds)
    else:
        supervise(args.poll_seconds)


if __name__ == "__main__":
    main()
