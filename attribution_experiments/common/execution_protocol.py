"""Prevent silently mixing transport results when resuming a run."""
import json
import os
from pathlib import Path
from .execution_environment import backend_name


def record_execution_backend(run_dir, *, check_only=False):
    run_dir = Path(run_dir)
    current = {'backend': backend_name(), 'version': 1}
    if current['backend'] == 'remote_docker':
        current['host'] = os.environ.get('EXPERIMENT_REMOTE_HOST', 'server241')
    marker = run_dir / 'execution_backend.json'
    if marker.exists():
        if json.loads(marker.read_text()) != current:
            raise RuntimeError('Execution backend changed; use a fresh run name instead of mixing results')
        return current
    # Budget registration precedes execution and contains no rollout artifacts.
    # Continue rejecting every unmarked directory with actual experiment data.
    has_artifacts = run_dir.exists() and any(
        entry.name != 'optimization_budget.json' or not entry.is_file()
        for entry in run_dir.iterdir()
    )
    if current['backend'] == 'remote_docker' and has_artifacts:
        raise RuntimeError('Unmarked historical run cannot resume on remote Docker; use a fresh run name')
    if check_only:
        return current
    run_dir.mkdir(parents=True, exist_ok=True)
    marker.write_text(json.dumps(current, indent=2))
    return current
