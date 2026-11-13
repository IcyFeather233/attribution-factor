"""Regrade an already submitted patch, then finalize SCOPE without new model calls."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import subprocess
import sys

from attribution_experiments.common.dataset import load_failed_instances
from attribution_experiments.common.evaluator import evaluate_patch
from attribution_experiments.common.result_validity import retryable_result
from attribution_experiments.scope.run_native import RUN_ROOT, UDOCKER_REPO, UDOCKER_EXECUTABLE, completed_final_results
from attribution_experiments.supervise_parallel import atomic_json


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--group', required=True)
    p.add_argument('--run-name', required=True)
    p.add_argument('--instance-id', required=True)
    args = p.parse_args()
    run = RUN_ROOT / args.group / args.run_name
    directory = run / 'final' / args.instance_id
    instances = load_failed_instances()
    cached = completed_final_results(run / 'final', instances)
    pending = [x['instance_id'] for x in instances if x['instance_id'] not in cached]
    if pending not in ([], [args.instance_id]):
        raise RuntimeError(f'Unexpected additional missing tasks: {pending}')
    if pending:
        patch = (directory / 'patch.diff').read_text()
        trajectory = json.loads((directory / 'trajectory.json').read_text())
        if not patch.strip() or trajectory.get('info', {}).get('exit_status') != 'Submitted':
            raise RuntimeError('Saved patch must come from a completed submission')
        stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
        backup = run / ('saved_patch_recovery_' + stamp)
        backup.mkdir()
        for name in ('result.json', 'trajectory.json', 'patch.diff'):
            shutil.copy2(directory / name, backup / name)
        if (directory / 'evaluation').exists():
            (directory / 'evaluation').rename(backup / 'evaluation.before')
        result = json.loads((directory / 'result.json').read_text())
        # The old result.json predates the newer completed trajectory, because
        # an evaluator exception prevented the runner from saving its new result.
        info = trajectory['info']
        for key in ('exception', 'traceback', 'error'):
            result.pop(key, None)
        result.update(exit_status='Submitted', api_calls=info['model_stats']['api_calls'],
                      patch_chars=len(patch), saved_patch_recovery=str(backup))
        instance = next(x for x in instances if x['instance_id'] == args.instance_id)
        evaluation = evaluate_patch(instance=instance, patch=patch, output_dir=directory / 'evaluation',
                                    udocker_repo=UDOCKER_REPO, udocker_executable=UDOCKER_EXECUTABLE)
        result['evaluation'] = evaluation
        if retryable_result(result) or not isinstance(evaluation.get('resolved'), bool):
            raise RuntimeError('Regrading did not produce a valid evaluation')
        atomic_json(directory / 'result.json', result)
    cached = completed_final_results(run / 'final', instances)
    if len(cached) != len(instances):
        raise RuntimeError('Refusing to launch any missing task rollout')
    # Resume reuses all 32 adaptation tasks and all 148 final results, and writes
    # the native summary. No alternate scoring implementation is introduced.
    subprocess.run([sys.executable, '-B', '-m', 'attribution_experiments.scope.run_native',
                    '--ablation-group', args.group, '--run-name', args.run_name, '--resume',
                    '--adaptation-tasks', '32', '--adaptation-seed', '20260902',
                    '--adaptation-batch-size', '8', '--adaptation-workers', '4',
                    '--quality-analysis-frequency', '4', '--final-workers', '8'], check=True)


if __name__ == '__main__':
    main()
