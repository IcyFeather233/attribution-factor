"""Register one-pass task-coverage alignment, or explicitly execute one job.

Default operation writes only a plan. No scheduler, models, or containers start.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

from .common.analysis_payloads import GROUPS

ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / 'qwen_gpt55_experiments'
PLAN = PACKAGE / 'runtime/ten_batches_unpadded_20261015/plan.json'
PYTHON = Path(sys.executable)


def jobs(groups):
    from .common.analysis_payloads import task_ids
    total = len(task_ids())
    if total < 3:
        raise ValueError('At least 3 tasks are required for training and candidate selection')
    val_size = min(22, total - 1)
    val_offset = min(10, total - val_size)
    rows = []
    def add(method, group, variant, module, flags, profile=None):
        name = 'ten_batches_unpadded_v6_' + variant + '_20261015'
        run = PACKAGE / (profile or method) / group / name
        command = [str(PYTHON), '-B', '-m', module, '--ablation-group', group]
        command += ['--run-dir', str(run)] if method == 'harnessfix' else ['--run-name', name]
        rows.append({'id': method + '__' + group + '__' + variant, 'method': method,
                     'group': group, 'variant': variant, 'run_dir': str(run),
                     'command': command + flags, 'state': 'planned'})
    for group in groups:
        add('harnessfix', group, '1iter_full148', 'qwen_gpt55_experiments.harnessfix.native_profile', [
            '--max-iterations', '1', '--max-promotion-failures', '2',
            '--analysis-tasks', str(total),
            '--validation-tasks', str(min(32, total)), '--validation-seed', '20260902',
            '--analysis-workers', '4', '--validation-workers', '8', '--final-workers', '8'])
        add('meta_harness', group, 'corpus148', 'qwen_gpt55_experiments.meta_harness.run_native', [
            '--adaptation-tasks', str(total), '--adaptation-seed', '20260902', '--benchmark-workers', '8'])
        add('gepa', group, 'p10_full148', 'qwen_gpt55_experiments.gepa.run_native', [
            '--adaptation-tasks', str(total), '--full-trace-pass',
            '--validation-offset', str(val_offset), '--validation-tasks', str(val_size),
            '--adaptation-seed', '20260902', '--reflection-batch-size', str(min(15, total)),
            '--max-candidate-proposals', '10', '--evolution-workers', '4', '--final-workers', '8'])
        add('recreate', group, 'b15_r1_full148', 'qwen_gpt55_experiments.recreate.run_shared_v2', [
            '--adaptation-tasks', str(total), '--adaptation-batch-size', '15',
            '--adaptation-repeats', '1', '--adaptation-seed', '20260902',
            '--adaptation-workers', '4', '--final-workers', '8'], profile='recreate_shared_v246')
        add('scope', group, 'b15_r1_full148', 'qwen_gpt55_experiments.scope.run_native', [
            '--adaptation-tasks', str(total), '--adaptation-batch-size', '15',
            '--adaptation-seed', '20260902', '--quality-analysis-frequency', '4',
            '--adaptation-workers', '4', '--final-workers', '8'])
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--groups', nargs='+', choices=GROUPS,
                        default=['baseline', 'group1_1_causing', 'group1_2_inevitable'])
    parser.add_argument('--execute-job', help='Explicitly launch exactly one job ID from the pinned plan')
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    files = [Path(__file__), PACKAGE / 'common/optimization_budget.py',
             PACKAGE / 'run_pipeline.py', PACKAGE / 'run_remote.py',
             PACKAGE / 'gepa/run_native.py', PACKAGE / 'gepa/protocol.py',
             PACKAGE / 'gepa/single_pass_sampler.py',
             PACKAGE / 'harnessfix/native_profile.py', PACKAGE / 'harnessfix/upstream_bridge.py',
             PACKAGE / 'meta_harness/run_native.py', PACKAGE / 'meta_harness/corpus_index.py',
             PACKAGE / 'meta_harness/resume_native.py',
             PACKAGE / 'recreate/run_native.py',
             PACKAGE / 'scope/run_native.py',
             ROOT / 'baselines/gepa/src/gepa/core/engine.py',
             ROOT / 'baselines/harnessfix/failure_analysis/validation_metrics.py',
             ROOT / 'baselines/harnessfix/failure_analysis/check_val_gate.py',
             ROOT / 'baselines/harnessfix/failure_analysis/plan_diff_audit.py']
    hashes = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    if not args.execute_job:
        if args.resume:
            parser.error('--resume requires --execute-job')
        from .common.dataset import load_failed_instances, select_repository_balanced_instances
        instances = load_failed_instances()
        val22 = select_repository_balanced_instances(instances, limit=32, seed=20260902)[10:]
        state = {'protocol': 'ten_batch_unpadded_v6', 'jobs': jobs(args.groups),
                 'gepa_task_ids': {'training148': [x['instance_id'] for x in instances],
                                   'validation22': [x['instance_id'] for x in val22]},
                 'source_sha256': hashes, 'state': 'planned_only',
                 'selection': 'original promotion gates; report every registered outcome',
                 'alignment_unit': '148 distinct tasks; GEPA/ReCreate/SCOPE use 10 batches, HF/Meta use 1 outer iteration; not equal tokens or calls',
                 'gepa_validation': '22 fixed selection IDs overlap training; 9 minibatches of 15 and a final batch of 13 cover 148 tasks exactly once',
                 'harnessfix_limitation': '148 historical failure diagnoses; 32 validation tasks overlap; subsequent diagnoses use actual current-candidate failures; final148 is not held-out testing',
                 'meta_harness_coverage': '148 historical records searchable by the native proposer; not a claim that all records are deeply read'}
        if PLAN.exists() and json.loads(PLAN.read_text()) != state:
            raise RuntimeError('An existing pilot plan differs; preserve it and use a new plan version')
        PLAN.parent.mkdir(parents=True, exist_ok=True)
        PLAN.write_text(json.dumps(state, indent=2) + '\n')
        print(json.dumps({'state': 'planned_only', 'jobs': len(state['jobs']), 'path': str(PLAN)}))
        return
    state = json.loads(PLAN.read_text())
    if any(not (ROOT / name).is_file() or hashlib.sha256((ROOT / name).read_bytes()).hexdigest() != expected
           for name, expected in state['source_sha256'].items()):
        raise RuntimeError('Source changed after planning; review and register a new plan')
    matches = [row for row in state['jobs'] if row['id'] == args.execute_job]
    if len(matches) != 1:
        parser.error('Unknown job ID in saved plan')
    row = matches[0]
    command = list(row['command'])
    if args.resume:
        if row['method'] == 'gepa' and (Path(row['run_dir']) / 'evolution_summary.json').exists():
            command = [str(PYTHON), '-B', '-m', 'qwen_gpt55_experiments.gepa.resume_final',
                       '--ablation-group', row['group'], '--run-name', Path(row['run_dir']).name,
                       '--workers', '8']
        elif row['method'] == 'meta_harness':
            search = Path(row['run_dir']) / 'search'
            if (search / 'agents/candidate_agent.py').exists() and (search / 'agents/candidate_scaffold.yaml').exists():
                command = [str(PYTHON), '-B', '-m', 'qwen_gpt55_experiments.meta_harness.resume_native',
                           '--ablation-group', row['group'], '--run-name', Path(row['run_dir']).name,
                           '--adaptation-tasks', '148', '--adaptation-seed', '20260902', '--workers', '8']
            else:
                command += ['--resume-analysis-proposal']
        else:
            command += ['--resume']
    elif Path(row['run_dir']).exists():
        raise RuntimeError('Run exists; use an explicit compatible resume')
    from .run_pipeline import child_environment, load_sidecar_key
    from .common.model_config import task_api_key
    env = child_environment(load_sidecar_key(), task_api_key())
    env.update(EXPERIMENT_CONTAINER_BACKEND='remote_docker', EXPERIMENT_REMOTE_HOST='server241',
               EXPERIMENT_ALIGNED_TASK_BUDGET='1', PYTHONDONTWRITEBYTECODE='1')
    raise SystemExit(subprocess.call(command, env=env, cwd=ROOT))


if __name__ == '__main__':
    main()
