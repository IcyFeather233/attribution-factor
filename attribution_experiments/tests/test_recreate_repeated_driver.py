"""Run the real batching/resume driver with fake task and optimizer calls."""
import json
import sys
from types import SimpleNamespace


def test_driver_finishes_two_passes_and_reuses_them_on_resume(tmp_path, monkeypatch):
    from attribution_experiments.recreate import run_native as runner
    from attribution_experiments.common import execution_protocol, flow_evidence
    tasks = [SimpleNamespace(instance_id=f'task-{i}') for i in range(4)]
    monkeypatch.setattr(runner, 'RUN_ROOT', tmp_path)
    monkeypatch.setattr(runner, 'optimizer_api_key', lambda: 'unused')
    monkeypatch.setattr(runner, 'task_api_key', lambda: 'unused')
    for name in ['install_dual_model_router', 'install_aligned_task_budget',
                 'install_redacted_traj_saver', 'install_local_environment_wrapper',
                 'snapshot_payloads', 'redact_api_key']:
        monkeypatch.setattr(runner, name, lambda *a, **k: None)
    monkeypatch.setattr(execution_protocol, 'record_execution_backend', lambda *a, **k: None)
    monkeypatch.setattr(flow_evidence, 'write_receipt', lambda *a, **k: None)
    monkeypatch.setattr(runner, 'load_payloads', lambda *a: {t.instance_id: {} for t in tasks})
    monkeypatch.setattr(runner, 'load_instances', lambda: tasks)
    monkeypatch.setattr(runner, 'select_adaptation_instances', lambda items, **k: items)
    monkeypatch.setattr(runner, 'UdockerSWEBenchAdapter', lambda **k: SimpleNamespace(
        get_recreate_agent_config=lambda: {}))
    def initialize(**kwargs):
        base = kwargs['workspace'] / 'global_v000'
        base.mkdir(parents=True)
        (base / 'scaffold.yaml').write_text('seed: true\n')
        return {'domain_config': {}, 'ablations': {}}
    monkeypatch.setattr(runner, 'initialize_workspace', initialize)
    visits, syntheses = [], []
    def evolve(**kwargs):
        output = kwargs['output_dir']; output.mkdir(parents=True, exist_ok=True)
        visits.append((kwargs['instance'].instance_id, kwargs['global_scaffold_dir'].name))
        return runner.BatchResult(kwargs['instance'].instance_id, False, 0., True, False, False)
    def synthesize(**kwargs):
        syntheses.append(kwargs['batch_idx'])
        kwargs['new_global_dir'].mkdir()
        (kwargs['new_global_dir'] / 'scaffold.yaml').write_text(f'batch: {kwargs["batch_idx"]}\n')
    monkeypatch.setattr(runner, 'run_single_evolution', evolve)
    monkeypatch.setattr(runner, 'run_batch_synthesis', synthesize)
    monkeypatch.setattr(runner, 'run_final', lambda **k: {'total': 4, 'resolved': 0, 'results': []})
    args = ['run', '--ablation-group', 'baseline', '--run-name', 'repeat',
            '--adaptation-tasks', '4', '--adaptation-batch-size', '2',
            '--adaptation-repeats', '2', '--adaptation-workers', '1']
    monkeypatch.setattr(sys, 'argv', args)
    runner.main()
    assert syntheses == [0, 1, 2, 3]
    assert visits == [('task-0', 'global_v000'), ('task-1', 'global_v000'),
                      ('task-2', 'global_v001'), ('task-3', 'global_v001'),
                      ('task-0', 'global_v002'), ('task-1', 'global_v002'),
                      ('task-2', 'global_v003'), ('task-3', 'global_v003')]
    run = tmp_path / 'baseline/repeat'
    summary = json.loads((run / 'summary.json').read_text())
    assert summary['global_versions'] == 5
    assert summary['adaptation_task_visits'] == 8
    (run / 'resume32_drained.json').write_text('{}')
    monkeypatch.setattr(sys, 'argv', args + ['--resume'])
    runner.main()
    assert len(visits) == 8 and len(syntheses) == 4
    assert (run / 'workspace/current').resolve().name == 'global_v004'
