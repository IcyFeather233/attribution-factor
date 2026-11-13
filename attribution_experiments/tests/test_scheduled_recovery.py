import json
from types import SimpleNamespace

from attribution_experiments import scheduled_recovery as scheduled


def test_plan_covers_all_ten_jobs_without_rerunning_valid_baselines():
    jobs = scheduled.plan_jobs()
    assert len(jobs) == 10 and len({j['id'] for j in jobs}) == 10
    assert sum(j['group'] == 'group3_3_both_flow_context' for j in jobs) == 5
    assert [j['method'] for j in jobs if j['group'] == 'baseline'] == ['gepa']
    scope = next(j for j in jobs if j['id'] == 'scope__group3_2_inevitable_flow_context')
    assert 'attribution_experiments.scope.recover_saved_patch' in scope['command']
    assert all('--resume' in j['command'] for j in jobs if j['group'].startswith('group2_'))


def test_no_launch_before_deadline_and_failure_does_not_block_queue(tmp_path, monkeypatch):
    clock = [0]
    jobs = [{'id': 'a', 'state': 'queued', 'command': ['fake', 'a']},
            {'id': 'b', 'state': 'queued', 'command': ['fake', 'b']}]
    (tmp_path / 'manifest.json').write_text(json.dumps({'not_before_epoch': 60, 'max_active_jobs': 1, 'jobs': jobs,
        'request_gate_environment': {'TASK_MODEL_MIN_REQUEST_INTERVAL_SECONDS': '0.1', 'TASK_MODEL_TPM_LIMIT': '20000000'}}))
    monkeypatch.setattr(scheduled, 'persistent_environment', lambda: {})
    monkeypatch.setattr(scheduled, 'child_environment', lambda *a: {})
    monkeypatch.setattr(scheduled, 'load_sidecar_key', lambda: 'unused')
    monkeypatch.setattr(scheduled, 'task_api_key', lambda: 'unused')
    monkeypatch.setattr(scheduled, 'find_existing_job', lambda job: None)
    launched = []
    done = set()
    def popen(command, **kwargs):
        assert clock[0] >= 60
        assert kwargs['env']['TASK_MODEL_MIN_REQUEST_INTERVAL_SECONDS'] == '0.1'
        assert kwargs['env']['TASK_MODEL_TPM_LIMIT'] == '20000000'
        launched.append(command[-1])
        done.add(command[-1])
        return SimpleNamespace(pid=100 + len(launched), poll=lambda: 1 if command[-1] == 'a' else 0)
    monkeypatch.setattr(scheduled, 'valid_summary', lambda job: job['id'] == 'b' and 'b' in done)
    def pause(seconds):
        clock[0] += seconds
        assert clock[0] < 1000
    scheduled.supervise(tmp_path, clock=lambda: clock[0], pause=pause,
                         admission=lambda _: (True, None), popen=popen)
    state = json.loads((tmp_path / 'manifest.json').read_text())
    assert launched == ['a', 'b']
    assert [j['state'] for j in state['jobs']] == ['failed', 'complete']
    assert state['state'] == 'needs_attention'


def test_existing_valid_result_is_reused_without_execution(tmp_path, monkeypatch):
    (tmp_path / 'manifest.json').write_text(json.dumps({'not_before_epoch': 0, 'max_active_jobs': 10,
        'jobs': [{'id': 'done', 'state': 'queued'}]}))
    monkeypatch.setattr(scheduled, 'persistent_environment', lambda: {})
    monkeypatch.setattr(scheduled, 'child_environment', lambda *a: {})
    monkeypatch.setattr(scheduled, 'load_sidecar_key', lambda: '')
    monkeypatch.setattr(scheduled, 'task_api_key', lambda: '')
    monkeypatch.setattr(scheduled, 'valid_summary', lambda _: True)
    def forbidden(*args, **kwargs):
        raise AssertionError('Should not launch existing complete result')
    scheduled.supervise(tmp_path, popen=forbidden, admission=forbidden)
    assert json.loads((tmp_path / 'manifest.json').read_text())['state'] == 'complete'
