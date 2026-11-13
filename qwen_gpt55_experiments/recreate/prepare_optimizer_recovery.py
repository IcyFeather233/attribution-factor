"""Fork an affected ReCreate run at the first incomplete optimizer batch."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import subprocess

from qwen_gpt55_experiments.run_pipeline import ROOT, PYTHON
from qwen_gpt55_experiments.supervise_parallel import atomic_json


def main():
    package = ROOT / 'qwen_gpt55_experiments'
    group = 'group3_2_inevitable_flow_context'
    old_name = 'remote_group3_native_flow_v2_20260920'
    new_name = 'remote_group3_native_flow_v2_recreate_repair_20260922'
    source = package / 'recreate' / group / old_name
    target = package / 'recreate' / group / new_name
    bad_id = 'pytest-dev__pytest-6197'
    assert not target.exists(), 'Recovery destination already exists'
    first_batch = source / 'workspace/batch_000'
    cached = []
    for task in sorted(first_batch.iterdir()):
        if not task.is_dir() or not (task / 'batch_result.json').exists() or task.name == bad_id:
            continue
        result = json.loads((task / 'batch_result.json').read_text())
        assert not result.get('error'), task.name
        traces = list((source / 'runs_recreate').glob('parallel_' + task.name + '_*/*/*.traj.json'))
        assert len(traces) == 1 and json.loads(traces[0].read_text())['exit_status'] in ('Submitted', 'Resolved'), task.name
        cached.append((task, traces[0]))
    assert len(cached) == 7
    (target / 'workspace/batch_000').mkdir(parents=True)
    signature = json.loads((source / 'execution_backend.json').read_text())
    assert signature == {'backend': 'remote_docker', 'version': 1, 'host': 'server241'}
    shutil.copy2(source / 'execution_backend.json', target / 'execution_backend.json')
    shutil.copytree(source / 'workspace/global_v000', target / 'workspace/global_v000', symlinks=True)
    for task, trace in cached:
        destination = target / 'workspace/batch_000' / task.name
        shutil.copytree(task, destination, symlinks=True)
        atomic_json(destination / 'evolution_status.json', {
            'completed': True, 'stage': 'per_case_optimizer', 'reused_from': str(task),
            'trajectory': str(trace), 'trajectory_sha256': hashlib.sha256(trace.read_bytes()).hexdigest(),
            'exit_status': json.loads(trace.read_text())['exit_status'],
        })
    checkpoint = {'created_at': datetime.now(timezone.utc).isoformat(), 'source_run': str(source),
                  'checkpoint_type': 'offline completed-run fork before batch-0 synthesis',
                  'reused_optimizer_tasks': [task.name for task, _ in cached],
                  'rerun_task': bad_id, 'downstream_batches_require_regeneration': [1, 2, 3],
                  'reuse_final_results': False}
    atomic_json(target / 'resume32_drained.json', checkpoint)
    atomic_json(target / 'recovery_checkpoint.json', checkpoint)
    original_summary = source / 'summary.json'
    backup = source / 'summary.before_optimizer_integrity_note_20260922.json'
    shutil.copy2(original_summary, backup)
    summary = json.loads(original_summary.read_text())
    summary.update(adaptation_complete=False, incomplete_optimizer_tasks=[bad_id], replacement_run_dir=str(target))
    atomic_json(original_summary, summary)
    manifest_path = package / 'runtime/remote_sweeps' / old_name / 'manifest.json'
    manifest = json.loads(manifest_path.read_text())
    job = next(j for j in manifest['jobs'] if j['group'] == group and j['method'] == 'recreate')
    job.update(adaptation_complete=False, replacement_run_dir=str(target),
               replacement_recovery='recreate_optimizer_repair_20260922')
    atomic_json(manifest_path, manifest)
    cmd = [str(PYTHON), '-B', '-m', 'qwen_gpt55_experiments.run_tracked_recovery',
           '--name', 'recreate_optimizer_repair_20260922', '--summary', str(target / 'summary.json'), '--',
           str(PYTHON), '-B', '-m', 'qwen_gpt55_experiments.run_remote', '--method', 'recreate',
           '--group', group, '--run-name', new_name, '--workers', '8', '--resume']
    subprocess.run(cmd, cwd=ROOT, check=True)
    print(json.dumps(checkpoint))


if __name__ == '__main__':
    main()
