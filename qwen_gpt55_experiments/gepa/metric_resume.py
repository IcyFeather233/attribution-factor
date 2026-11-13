"""Reuse each completed metric once while rebuilding an interrupted GEPA engine."""

import json
import shutil
import threading
from collections import defaultdict, deque
from pathlib import Path
from qwen_gpt55_experiments.common.execution_environment import backend_name


class MetricReplay:
    def __init__(self, source: Path, target: Path, *, model: str, step_limit: int):
        def transport(path, default):
            marker = path / 'execution_backend.json'
            return json.loads(marker.read_text()).get('backend') if marker.exists() else default
        if transport(source, 'udocker') != transport(target, backend_name()):
            raise ValueError('Metric replay requires identical execution backends')
        self.entries = defaultdict(deque)
        self.lock = threading.Lock()
        self.reused = 0
        old = {p.name: p.read_bytes() for p in (source / "provided_analysis").glob("*.json")}
        new = {p.name: p.read_bytes() for p in (target / "provided_analysis").glob("*.json")}
        if not old or old != new:
            raise ValueError("Metric replay requires identical provided-analysis snapshots")
        for path in sorted((source / "metric_calls").glob("*")):
            try:
                result = json.loads((path / "result.json").read_text())
                report = json.loads((path / "evaluation" / "report.json").read_text())
                trajectory = json.loads((path / "trajectory.json").read_text())
                candidate = (path / "candidate_scaffold.yaml").read_text()
                evaluation = result.get("evaluation", {})
                if (
                    result.get("model") != model or result.get("step_limit") != step_limit
                    or result.get("temperature") != 1.0 or result.get("api_calls", 0) <= 0
                    or result.get("exception")
                    or result.get("exit_status") not in {"Submitted", "LimitsExceeded", "RepeatedFormatError"}
                    or not isinstance(evaluation.get("resolved"), bool)
                    or report.get("resolved") != evaluation["resolved"]
                    or report.get("instance_id") != result.get("instance_id")
                    or not trajectory.get("messages")
                    or not (path / "patch.diff").is_file()
                ):
                    continue
                self.entries[(candidate, result["instance_id"])].append(path)
            except (OSError, ValueError, KeyError, TypeError, AttributeError):
                continue
        self.available = sum(map(len, self.entries.values()))

    def restore(self, candidate: str, instance_id: str, destination: Path) -> dict | None:
        with self.lock:
            entries = self.entries.get((candidate, instance_id))
            if not entries:
                return None
            source = entries.popleft()
        shutil.copytree(source, destination, dirs_exist_ok=True)
        result = json.loads((destination / "result.json").read_text())
        result["resumed_from_metric_artifact"] = str(source)
        (destination / "result.json").write_text(json.dumps(result, indent=2))
        with self.lock:
            self.reused += 1
        print(f"Reused completed GEPA metric: {instance_id}", flush=True)
        return result
