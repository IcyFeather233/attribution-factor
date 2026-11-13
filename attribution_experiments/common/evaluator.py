"""Official SWE-bench grading executed inside a fresh udocker container."""

from __future__ import annotations

import json
import re
import time
import uuid
from pathlib import Path

from swebench.harness.constants import KEY_INSTANCE_ID, KEY_PREDICTION
from swebench.harness.grading import get_eval_report
from swebench.harness.test_spec.test_spec import make_test_spec

from .dataset import image_for_instance
from .execution_environment import ExecutionEnvironment as UdockerEnvironment
from .execution_environment import backend_name
from .remote_runtime import RemoteInfrastructureError

_HUNK_HEADER = re.compile(
    r"^@@ -\d+(?:,(?P<old>\d+))? \+\d+(?:,(?P<new>\d+))? @@"
)


def normalize_patch_transport(patch: str) -> str:
    """Repair only EOF whitespace lost while transporting a unified diff."""
    if not patch:
        return patch
    lines = patch.splitlines()
    last_hunk = None
    for index, line in enumerate(lines):
        if _HUNK_HEADER.match(line):
            last_hunk = index
    suffix = ""
    if last_hunk is not None:
        match = _HUNK_HEADER.match(lines[last_hunk])
        old_expected = int(match.group("old") or "1")
        new_expected = int(match.group("new") or "1")
        old_seen = new_seen = 0
        for line in lines[last_hunk + 1 :]:
            if line.startswith("diff --git ") or _HUNK_HEADER.match(line):
                break
            if line.startswith("\\"):
                continue
            if line.startswith("-"):
                old_seen += 1
            elif line.startswith("+"):
                new_seen += 1
            elif line.startswith(" "):
                old_seen += 1
                new_seen += 1
        old_missing = old_expected - old_seen
        new_missing = new_expected - new_seen
        if old_missing == new_missing and old_missing > 0:
            suffix = " \n" * old_missing
    normalized = patch if patch.endswith("\n") else patch + "\n"
    return normalized + suffix


def evaluate_patch(
    *,
    instance: dict,
    patch: str,
    output_dir: Path,
    udocker_repo: Path,
    udocker_executable: Path,
    timeout: int = 1800,
) -> dict:
    """Apply a prediction to a fresh official image and grade its full test spec."""
    patch = normalize_patch_transport(patch)
    output_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    timings = {}
    instance_id = instance[KEY_INSTANCE_ID]
    prediction = {
        KEY_INSTANCE_ID: instance_id,
        "model_name_or_path": "native-baseline",
        KEY_PREDICTION: patch,
    }
    test_spec = make_test_spec(instance)

    result = {
        'execution_backend': backend_name(),
        "instance_id": instance_id,
        "resolved": False,
        "patch_applied": False,
        "timed_out": False,
        "timing_seconds": timings,
    }
    if not patch.strip():
        result["error"] = "empty_patch"
        timings["total"] = time.perf_counter() - started
        (output_dir / "report.json").write_text(json.dumps(result, indent=2))
        return result

    stage_started = time.perf_counter()
    env = UdockerEnvironment(
        image=image_for_instance(instance),
        repo=udocker_repo,
        executable=udocker_executable,
        cwd="/testbed",
        timeout=120,
        pull_timeout=3600,
        container_name=f"eval-{uuid.uuid4().hex[:12]}",
        env={"PAGER": "cat", "MANPAGER": "cat", "PIP_PROGRESS_BAR": "off", "TQDM_DISABLE": "1"},
    )
    timings["environment_setup_including_queue"] = time.perf_counter() - stage_started
    try:
        stage_started = time.perf_counter()
        apply_result = env.execute(
            {"command": "git apply --verbose -"},
            input_text=patch,
            timeout=180,
        )
        timings["patch_apply"] = time.perf_counter() - stage_started
        (output_dir / "patch_apply.log").write_text(apply_result["output"])
        if apply_result["returncode"] != 0:
            result["error"] = "patch_apply_failed"
            result["patch_apply_returncode"] = apply_result["returncode"]
            return result
        result["patch_applied"] = True

        stage_started = time.perf_counter()
        eval_result = env.execute(
            {"command": "/bin/bash -s"},
            input_text=test_spec.eval_script,
            timeout=timeout,
        )
        timings["test_execution"] = time.perf_counter() - stage_started
        test_log = output_dir / "test_output.txt"
        test_log.write_text(eval_result["output"])
        result["eval_returncode"] = eval_result["returncode"]
        result["exception_info"] = eval_result["exception_info"]
        result["timed_out"] = "TimeoutExpired" in eval_result["exception_info"]

        report = get_eval_report(
            test_spec=test_spec,
            prediction=prediction,
            test_log_path=str(test_log),
            include_tests_status=True,
        )
        result["official_report"] = report
        result["resolved"] = bool(report.get(instance_id, {}).get("resolved", False))
        return result
    except RemoteInfrastructureError as exc:
        result['resolved'] = None
        result['error'] = 'remote_infrastructure_error'
        result['exception_info'] = str(exc)
        raise
    finally:
        stage_started = time.perf_counter()
        env.cleanup()
        timings["cleanup"] = time.perf_counter() - stage_started
        timings["total"] = time.perf_counter() - started
        (output_dir / "report.json").write_text(json.dumps(result, indent=2))
