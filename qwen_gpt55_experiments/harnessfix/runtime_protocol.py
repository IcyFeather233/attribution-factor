"""Prevent reuse of benchmarks that bypassed candidate environment behavior."""

import json
from qwen_gpt55_experiments.common.execution_environment import backend_name
from pathlib import Path


PROTOCOL = 'remote_docker_candidate_v1' if backend_name() == 'remote_docker' else 'persistent_udocker_candidate_v1'


def require_compatible_runtime(run_dir: Path) -> None:
    marker = run_dir / "runtime_protocol.json"
    has_results = (run_dir / "summary.json").exists() or any(run_dir.glob("iteration_*"))
    if marker.is_file():
        if json.loads(marker.read_text()).get("environment_transport") == PROTOCOL:
            return
    if has_results:
        raise RuntimeError(
            "HarnessFix run predates candidate-environment transport or has an incompatible protocol. "
            "Keep its artifacts for audit and use a fresh --run-dir; old validation/final results cannot be reused."
        )
