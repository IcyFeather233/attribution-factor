from pathlib import Path
import shutil
import subprocess
import sys
from types import SimpleNamespace

import pytest

from qwen_gpt55_experiments.common.execution_environment import backend_name
from qwen_gpt55_experiments.common.remote_runtime import merge_workspace, snapshot, RemoteRuntime


def test_remote_observation_accepts_non_utf8_and_preserves_exit_marker(tmp_path, monkeypatch):
    from qwen_gpt55_experiments.common.remote_runtime import decode_remote_status
    real_run = subprocess.run
    def local_process(command, **kwargs):
        return real_run([sys.executable, '-c',
                         "import sys; sys.stdout.buffer.write(b'raw\\xdf\\n__FA_EXEC_STATUS_bytes__:0\\n')"], **kwargs)
    monkeypatch.setattr('qwen_gpt55_experiments.common.remote_runtime.subprocess.run', local_process)
    monkeypatch.setattr('qwen_gpt55_experiments.common.remote_runtime.persistent_environment', lambda: {})
    runtime = RemoteRuntime.__new__(RemoteRuntime)
    runtime.host, runtime.local_dir = 'unused', tmp_path
    result = runtime.ssh(['unused'])
    decoded = decode_remote_status(result, [], 'bytes')
    assert decoded.returncode == 0 and decoded.stdout == 'raw\ufffd'


def test_remote_archive_stream_is_not_decoded(tmp_path, monkeypatch):
    real_run = subprocess.run
    def local_process(command, **kwargs):
        return real_run([sys.executable, '-c',
                         'import sys; sys.stdout.buffer.write(sys.stdin.buffer.read())'], **kwargs)
    monkeypatch.setattr('qwen_gpt55_experiments.common.remote_runtime.subprocess.run', local_process)
    monkeypatch.setattr('qwen_gpt55_experiments.common.remote_runtime.persistent_environment', lambda: {})
    runtime = RemoteRuntime.__new__(RemoteRuntime)
    runtime.host, runtime.local_dir = 'unused', tmp_path
    raw = b'archive\x00\xdf\xff'
    path = tmp_path / 'archive'
    path.write_bytes(raw)
    with path.open('rb') as handle:
        assert runtime.ssh(['unused'], stdin=handle).stdout == raw


def test_workspace_preserves_host_logs_and_remote_tools(tmp_path):
    local, stage = tmp_path / 'local', tmp_path / 'stage'
    local.mkdir()
    (local / 'tool.py').write_text('old')
    (local / 'removed').write_text('old')
    before = snapshot(local)
    shutil.copytree(local, stage)
    (stage / 'tool.py').write_text('new tool')
    (stage / 'removed').unlink()
    (stage / 'memory').mkdir()
    (stage / 'memory/note').write_text('learned')
    (local / 'trajectory.json').write_text('host update during command')
    merge_workspace(local, stage, before)
    assert (local / 'trajectory.json').read_text() == 'host update during command'
    assert (local / 'tool.py').read_text() == 'new tool'
    assert (local / 'memory/note').read_text() == 'learned'
    assert not (local / 'removed').exists()


def test_workspace_conflict_and_escaping_symlink_are_not_silently_overwritten(tmp_path):
    local, stage = tmp_path / 'local', tmp_path / 'stage'
    local.mkdir()
    (local / 'memory').write_text('old')
    before = snapshot(local)
    shutil.copytree(local, stage)
    (local / 'memory').write_text('host')
    (stage / 'memory').write_text('container')
    with pytest.raises(RuntimeError, match='Concurrent'):
        merge_workspace(local, stage, before)
    assert (local / 'memory').read_text() == 'host'
    (stage / 'escape').symlink_to('/etc/passwd')
    with pytest.raises(ValueError, match='escapes'):
        snapshot(stage)


def test_unknown_backend_is_rejected(monkeypatch):
    monkeypatch.setenv('EXPERIMENT_CONTAINER_BACKEND', 'typo')
    with pytest.raises(ValueError):
        backend_name()


def test_remote_timeout_preserves_partial_output_and_bounds_remote_process(tmp_path):
    runtime = RemoteRuntime.__new__(RemoteRuntime)
    runtime.local_dir = tmp_path
    runtime.mounts = {}
    runtime.sync_failed = False
    runtime.container_name = 'own'
    runtime.config = SimpleNamespace(cwd='/testbed')
    calls = []
    def ssh(args, **kwargs):
        calls.append((args, kwargs))
        return subprocess.CompletedProcess(args, 0, 'partial\n__FA_EXEC_STATUS_' + args[3] + '__:124\n', None)
    runtime.ssh = ssh
    with pytest.raises(subprocess.TimeoutExpired) as caught:
        runtime.run(['run', '--nobanner', 'own', 'bash', '-c', 'sleep 100'], timeout=1)
    assert caught.value.output == 'partial'
    assert calls[0][0][-6:] == ['--signal=TERM', '--kill-after=5', '1', 'bash', '-c', 'sleep 100']
    assert calls[0][1]['timeout'] == 31
    with pytest.raises(ValueError, match='private'):
        runtime.run(['run', '--nobanner', 'someone-else', 'true'], timeout=1)


def test_pressure_prevents_admission():
    runtime = RemoteRuntime.__new__(RemoteRuntime)
    runtime.ssh = lambda *a, **kw: SimpleNamespace(stdout='[1, 1000000000000]')
    with pytest.raises(RuntimeError, match='disk'):
        runtime.guard()


def test_remote_resource_removal_retries_transient_daemon_error(monkeypatch):
    runtime = RemoteRuntime.__new__(RemoteRuntime)
    replies = iter([
        subprocess.CompletedProcess([], 1, 'removal of container is already in progress', None),
        subprocess.CompletedProcess([], 0, 'owned', None),
    ])
    runtime.ssh = lambda *args, **kwargs: next(replies)
    monkeypatch.setattr('qwen_gpt55_experiments.common.remote_runtime.time.sleep', lambda _: None)
    runtime._remove_remote_resource(
        ['docker', 'rm', '-f', 'owned'], missing_markers=('no such container',)
    )
