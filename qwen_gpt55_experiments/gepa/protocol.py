"""Budget and completion checks for bounded multi-proposal GEPA search."""

import json
import math
from collections import Counter


def one_pass_proposals(task_count, minibatch_size):
    if task_count < 1 or minibatch_size < 1:
        raise ValueError('Task count and minibatch size must be positive')
    return math.ceil(task_count / minibatch_size)


def one_pass_metric_budget(validation_size, task_count, minibatch_size):
    if validation_size < 1:
        raise ValueError('Validation set must be nonempty')
    proposals = one_pass_proposals(task_count, minibatch_size)
    return validation_size + 2 * task_count + proposals * validation_size


def reflection_coverage(engine_dir, train_ids):
    seen, presentations, completed = set(), 0, 0
    counts, batches = Counter(), []
    for path in (engine_dir / 'iterations').glob('*/meta.json'):
        if path.parent.name == 'seed' or not (path.parent / 'reflective_dataset.json').exists():
            continue
        metadata = json.loads(path.read_text())
        batches.append((metadata['trace_i'], len(metadata.get('subsample_ids', []))))
        for index in metadata.get('subsample_ids', []):
            if not isinstance(index, int) or not 0 <= index < len(train_ids):
                raise ValueError('Reflection metadata references an unknown training example')
            seen.add(train_ids[index])
            counts[train_ids[index]] += 1
            presentations += 1
        completed += 1
    return {'unique_reflection_tasks': len(seen), 'reflection_task_ids': sorted(seen),
            'task_presentations': presentations, 'completed_reflection_proposals': completed,
            'batch_sizes': [size for _, size in sorted(batches)],
            'repeated_task_ids': sorted(iid for iid, count in counts.items() if count > 1),
            'missing_task_ids': sorted(set(train_ids) - seen)}


def split_adaptation(instances, train_tasks: int, *, validation_offset=None, validation_tasks=None):
    if validation_offset is not None or validation_tasks is not None:
        if validation_offset is None or validation_tasks is None or validation_offset < 0 or validation_tasks < 1:
            raise ValueError('Specify a nonnegative validation offset and positive validation size together')
        end = validation_offset + validation_tasks
        remaining = instances[:validation_offset] + instances[end:]
        if end > len(instances) or not 1 <= train_tasks <= len(remaining):
            raise ValueError('Insufficient examples for the requested disjoint train/validation split')
        return remaining[:train_tasks], instances[validation_offset:end]
    if not 1 <= train_tasks < len(instances):
        raise ValueError('train_tasks must leave at least one independent validation task')
    return instances[:train_tasks], instances[train_tasks:]


def prepare_engine_resume(run_dir):
    """Keep the upstream checkpoint, pool, counters and validation history intact."""
    engine = run_dir / 'engine'
    if engine.exists() and any(engine.iterdir()) and not (engine / 'gepa_state.bin').is_file():
        raise ValueError('GEPA engine has artifacts but no checkpoint; do not silently rebuild search')
    calls = run_dir / 'metric_calls'
    indices = [int(p.name.split('_', 1)[0]) for p in calls.iterdir()
               if p.is_dir() and p.name.split('_', 1)[0].isdigit()] if calls.exists() else []
    return max(indices, default=-1) + 1


def proposal_budget(
    validation_size: int,
    reflection_batch_size: int,
    max_candidate_proposals: int = 1,
) -> int:
    if validation_size < 1 or reflection_batch_size < 1:
        raise ValueError("GEPA validation and reflection batches must be nonempty")
    if max_candidate_proposals < 1:
        raise ValueError("GEPA must allow at least one candidate proposal")
    return validation_size + max_candidate_proposals * (
        2 * reflection_batch_size + validation_size
    )


def require_completed_evolution(summary: dict) -> None:
    if not summary.get("optimization_completed") or summary.get("new_candidates_evaluated", 0) < 1:
        raise RuntimeError(
            "GEPA has not evaluated a new candidate. Do not treat seed-only results "
            "as optimized results; rerun evolution with a sufficient proposal budget."
        )
