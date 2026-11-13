import json
from pathlib import Path


def test_saved_submission_is_regraded_without_another_task_rollout(tmp_path, monkeypatch):
    from qwen_gpt55_experiments.scope import recover_saved_patch as recovery
    monkeypatch.setattr(recovery, 'RUN_ROOT', tmp_path)
    monkeypatch.setattr(recovery, 'load_failed_instances', lambda: [{'instance_id': 'saved'}, {'instance_id': 'cached'}])
    run = tmp_path / 'group' / 'run'
    task = run / 'final/saved'
    (task / 'evaluation').mkdir(parents=True)
    old = {'instance_id': 'saved', 'exit_status': 'RemoteInfrastructureError', 'exception': 'previous failure', 'api_calls': 12}
    (task / 'result.json').write_text(json.dumps(old))
    (task / 'trajectory.json').write_text(json.dumps({'info': {'exit_status': 'Submitted', 'model_stats': {'api_calls': 20}}}))
    (task / 'patch.diff').write_text('saved candidate patch\n')
    (task / 'evaluation/report.json').write_text('{"resolved": null}')
    cached = run / 'final/cached'
    (cached / 'evaluation').mkdir(parents=True)
    (cached / 'result.json').write_text(json.dumps({'instance_id': 'cached', 'api_calls': 4, 'evaluation': {'resolved': False}}))
    (cached / 'evaluation/report.json').write_text('{"resolved": false}')
    before = (cached / 'result.json').read_bytes()
    calls = []
    def evaluate(**kwargs):
        assert kwargs['instance']['instance_id'] == 'saved'
        assert kwargs['patch'] == 'saved candidate patch\n'
        kwargs['output_dir'].mkdir()
        (kwargs['output_dir'] / 'report.json').write_text('{"resolved": true}')
        calls.append('evaluate')
        return {'resolved': True}
    def finalize(command, check):
        assert '--resume' in command
        assert len(recovery.completed_final_results(run / 'final', recovery.load_failed_instances())) == 2
        calls.append('finalize_all_cached')
    monkeypatch.setattr(recovery, 'evaluate_patch', evaluate)
    monkeypatch.setattr(recovery.subprocess, 'run', finalize)
    monkeypatch.setattr('sys.argv', ['recover', '--group', 'group', '--run-name', 'run', '--instance-id', 'saved'])
    recovery.main()
    after = json.loads((task / 'result.json').read_text())
    assert after['exit_status'] == 'Submitted' and after['api_calls'] == 20
    assert 'exception' not in after and after['evaluation']['resolved'] is True
    assert (cached / 'result.json').read_bytes() == before
    assert json.loads((Path(after['saved_patch_recovery']) / 'result.json').read_text()) == old
    assert calls == ['evaluate', 'finalize_all_cached']
