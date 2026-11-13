import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

from attribution_experiments.common.remote_runtime import (
    RemoteRuntime, RemoteInfrastructureError, remote_status_command, decode_remote_status,
)
from attribution_experiments.harnessfix.candidate_worker import finish_candidate
from attribution_experiments.recreate.resume_state import recovered_batch_result


@pytest.mark.parametrize('code', [0, 1, 124, 137, 255])
def test_real_process_exit_is_separate_from_transport_and_stdin_preserved(code):
    cmd = [sys.executable, '-c', 'import sys;sys.stdout.write(sys.stdin.read());sys.exit(' + str(code) + ')']
    response = subprocess.run(remote_status_command(cmd, 'testnonce'), input='no final newline',
                              text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=True)
    result = decode_remote_status(response, cmd, 'testnonce')
    assert result.returncode == code and result.stdout == 'no final newline'


def test_transport_failure_and_missing_receipt_are_not_task_exit_codes():
    for response in [subprocess.CompletedProcess([], 255, 'Connection reset'),
                     subprocess.CompletedProcess([], 0, 'partial without confirmation'),
                     subprocess.CompletedProcess([], 0, '\n__FA_EXEC_STATUS_wrong__:0\n')]:
        with pytest.raises(RemoteInfrastructureError):
            decode_remote_status(response, [], 'right')


def test_runtime_allows_program_255_and_keeps_container_usable(tmp_path):
    runtime = RemoteRuntime.__new__(RemoteRuntime)
    runtime.local_dir = tmp_path
    runtime.mounts = {}
    runtime.sync_failed = False
    runtime.failed = False
    runtime.container_name = 'own'
    runtime.config = SimpleNamespace(cwd='/testbed')
    def ssh(args, **kwargs):
        if args[:2] == ['docker', 'inspect']:
            return subprocess.CompletedProcess(args, 0, '{"Running":true}')
        return subprocess.CompletedProcess(args, 0, 'error: invalid option: --path\n__FA_EXEC_STATUS_' + args[3] + '__:255\n')
    runtime.ssh = ssh
    result = runtime.run(['run', '--nobanner', 'own', 'git', 'diff', '--path'], timeout=1)
    assert result.returncode == 255 and not runtime.failed
    assert 'invalid option' in result.stdout
    runtime.run(['run', '--nobanner', 'own', 'git', 'diff', '--path'], timeout=1)


@pytest.mark.parametrize('save_fails', [False, True])
def test_candidate_diff_failure_keeps_diagnostics_and_always_cleans(tmp_path, save_fails):
    events = []
    class Env:
        def execute(self, *args, **kwargs):
            raise RemoteInfrastructureError('connection lost')
        def cleanup(self):
            events.append('cleanup')
    class Agent:
        n_calls = 111
        def save(self, path, data):
            events.append('save')
            if save_fails:
                raise OSError('save failed')
            path.write_text(json.dumps(data))
    result = {'instance_id': 'task', 'exit_status': 'Submitted'}
    finish_candidate(Agent(), Env(), tmp_path, 'secret-test', result, '')
    saved = json.loads((tmp_path / 'result.json').read_text())
    assert saved['exit_status'] == 'RemoteInfrastructureError'
    assert saved['exception'] == 'connection lost'
    assert saved['api_calls'] == 111 and 'evaluation' not in saved
    assert events == ['save', 'cleanup']


def test_candidate_does_not_issue_recovery_command_after_failure(tmp_path):
    env = SimpleNamespace(execute=lambda *a, **k: pytest.fail('Must not repeat commands after failure'), cleanup=lambda: None)
    agent = SimpleNamespace(n_calls=2, save=lambda *args: None)
    result = {'instance_id': 'task', 'exit_status': 'RemoteInfrastructureError'}
    finish_candidate(agent, env, tmp_path, 'test', result, '')
    assert json.loads((tmp_path / 'result.json').read_text())['exit_status'] == 'RemoteInfrastructureError'


def test_recreate_failed_optimizer_is_not_a_successful_no_change_checkpoint(tmp_path):
    data = {'instance_id': 'task', 'success': True, 'scaffold_changed': False, 'error': ''}
    (tmp_path / 'batch_result.json').write_text(json.dumps(data))
    (tmp_path / 'evolution_status.json').write_text(json.dumps({'completed': False, 'exit_status': 'APIError: provider rejection'}))
    assert recovered_batch_result(tmp_path, tmp_path, 'task') is None
    (tmp_path / 'evolution_status.json').write_text(json.dumps({'completed': True}))
    assert recovered_batch_result(tmp_path, tmp_path, 'task') == data
    data['error'] = 'optimizer incomplete'
    (tmp_path / 'batch_result.json').write_text(json.dumps(data))
    assert recovered_batch_result(tmp_path, tmp_path, 'task') is None
