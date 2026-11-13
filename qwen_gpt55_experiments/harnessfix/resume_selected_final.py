"""Finish evaluation of a recorded single-round selection without optimizing again."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path

from .recover_final_infrastructure import gradeable
from ..common.udocker_runtime import persistent_environment
from ..supervise_parallel import atomic_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--check-only', action='store_true')
    parser.add_argument('--retained-original', action='store_true',
                        help='Require a saved rejected round and selected_iteration=0 summary')
    args = parser.parse_args()
    run = args.run_dir.resolve()
    if not run.is_relative_to('/volume/med'):
        raise ValueError('Persistent run directory required')
    budget = json.loads((run / 'optimization_budget.json').read_text())
    reports = [json.loads(f.read_text()) for f in sorted(run.glob('iteration_*/iteration_report.json'))]
    if budget['max_iterations'] != 1 or len(reports) != 1:
        raise ValueError('This recovery requires exactly one completed optimization round')
    report = reports[0]
    if args.retained_original:
        previous = json.loads((run / 'summary.json').read_text())
        candidate = (run / 'original_agent').resolve()
        if (report.get('promoted') is not False or previous.get('selected_iteration') != 0
                or Path(previous['selected_agent']).resolve() != candidate):
            raise ValueError('No matching recorded retained-original selection')
        selected_iteration = 0
    else:
        if not (report.get('promoted') and report['audit']['passed'] and report['gate']['passed']):
            raise ValueError('No recorded promoted candidate; do not guess a selection')
        candidate = Path(report['candidate_agent']).resolve()
        selected_iteration = report['iteration']
    if not candidate.is_relative_to(run) or not candidate.is_dir():
        raise ValueError('Recorded candidate is not available in this run')
    ids = (run / 'instance_ids.txt').read_text().split()
    expected = set((Path(__file__).resolve().parents[1] / 'task_ids.txt').read_text().split())
    if len(ids) != 148 or set(ids) != expected:
        raise ValueError('Unexpected evaluation task set')
    reused = [iid for iid in ids if gradeable(run / 'final' / iid)]
    if args.check_only:
        print(json.dumps({'selected_iteration': selected_iteration, 'candidate': str(candidate),
                          'reused': len(reused), 'remaining': 148 - len(reused)}))
        return
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    archive = run / ('final_reboot_recovery_' + stamp)
    archive.mkdir()
    for iid in ids:
        output = run / 'final' / iid
        if iid not in reused and output.exists():
            output.rename(archive / iid)
    atomic_json(archive / 'receipt.json', {'selected_candidate': str(candidate),
        'reused': reused, 'retry': [iid for iid in ids if iid not in reused]})
    os.environ.update(persistent_environment())
    os.environ['HARNESSFIX_NATIVE_PROFILE'] = '1'
    from . import run_native as adapter
    from .native_profile import install
    install(adapter)
    optimizer_key, task_key = adapter.optimizer_api_key(), adapter.task_api_key()
    env = adapter.optimizer_env(optimizer_key, task_key)
    print(f'Reusing {len(reused)}/148 final evaluations; completing {148-len(reused)} tasks', flush=True)
    rows = adapter.run_benchmark(run / 'final', candidate_dir=candidate, ids=ids,
                                 workers=args.workers, env=env, api_key=task_key, resume=True)
    if len(rows) != 148 or not all(gradeable(run / 'final' / iid) for iid in ids):
        raise RuntimeError('Final recovery is not complete; preserve artifacts for another resume')
    resolved = [row['instance_id'] for row in rows if row['evaluation']['resolved']]
    manifest = json.loads((run / 'manifest.json').read_text())
    group = budget['group']
    summary = {'method': 'HarnessFix', 'status': 'complete', 'ablation_group': group,
               'analysis_json_injected': group != 'baseline',
               'analysis_injection_point': 'native per-trace diagnosis and aggregation',
               'selected_iteration': selected_iteration, 'selected_agent': str(candidate),
               'max_iterations': 1, 'completed_iterations': 1, 'iterations': reports,
               'validation_protocol': manifest.get('validation_protocol'),
               'promotion_policy': 'upstream_step_promotion_decision',
               'validation_task_count': len((run / 'validation_ids.txt').read_text().split()),
               'validation_ids': (run / 'validation_ids.txt').read_text().split(),
               'analysis_task_count': len((run / 'analysis_ids.txt').read_text().split()),
               'optimization_budget': budget,
               'final_recovery': {'receipt': str(archive / 'receipt.json'), 'reused': len(reused)},
               'final': {'status': 'complete', 'total': 148, 'resolved': len(resolved),
                         'resolved_ids': resolved, 'results': rows}}
    if args.retained_original:
        summary.update(retained_original_evaluated=True, final_score_is_incomplete=False)
    atomic_json(run / 'summary.json', summary)
    print(json.dumps({'complete': True, 'resolved': len(resolved), 'total': 148}), flush=True)


if __name__ == '__main__':
    main()
