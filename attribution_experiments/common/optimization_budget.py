"""Pin optimization budgets before execution; reject incompatible resumes."""
import json
from pathlib import Path


def pin_budget(run_dir: Path, specification: dict, *, has_progress: bool) -> None:
    path = run_dir / 'optimization_budget.json'
    if path.exists():
        if json.loads(path.read_text()) != specification:
            raise ValueError('Optimization budget/protocol changed; use a new run directory')
    elif has_progress:
        raise ValueError('Unmarked historical optimization cannot use the new protocol; use a new run directory')
    else:
        run_dir.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix('.tmp')
        temporary.write_text(json.dumps(specification, indent=2, sort_keys=True) + '\n')
        temporary.replace(path)


def adaptation_batches(instances, batch_size: int, repeats: int):
    if batch_size < 1 or repeats < 1:
        raise ValueError('batch_size and repeats must be positive')
    batch_index = 0
    for repeat in range(repeats):
        for start in range(0, len(instances), batch_size):
            yield batch_index, repeat, instances[start:start + batch_size]
            batch_index += 1


def committed_resume_boundary(completed, total, batch_size):
    """A committed final partial batch is complete, not an interrupted batch."""
    if not 0 <= completed <= total or batch_size < 1:
        raise ValueError('Invalid adaptation progress')
    return completed if completed == total else (completed // batch_size) * batch_size
