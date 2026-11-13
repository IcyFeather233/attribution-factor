from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

from qwen_gpt55_experiments.common.remote_runtime import RemoteRuntime


def runtime_for(tmp_path, source):
    runtime = RemoteRuntime.__new__(RemoteRuntime)
    runtime.local_dir = tmp_path
    runtime.container_name = 'test-owned-container'
    return runtime


def test_push_tolerates_entry_disappearing_between_scan_and_delete(tmp_path):
    source, target = tmp_path / 'source', tmp_path / 'target'
    source.mkdir()
    target.mkdir()
    (source / 'new.txt').write_text('new')
    stale = target / 'old.txt'
    stale.write_text('old')
    runtime = runtime_for(tmp_path, source)
    copied = []
    def ssh(args, **kwargs):
        if args[3] == 'find':
            command = args[3:]
            # Deterministically remove the discovered file immediately before
            # GNU find tries its own deletion, reproducing the observed race.
            command = command[:-1] + ['-exec', sys.executable, '-c',
                'from pathlib import Path; import sys; Path(sys.argv[1]).unlink()', '{}', ';', command[-1]]
            return subprocess.run(command, text=True, capture_output=True, check=True)
        copied.append(kwargs['stdin'].read())
        return SimpleNamespace(returncode=0)
    runtime.ssh = ssh
    runtime.push(str(source), str(target))
    assert not stale.exists()
    assert len(copied) == 1 and len(copied[0]) > 0
    assert not (tmp_path / 'workspace-upload.tar').exists()


def test_push_still_rejects_missing_workspace_root(tmp_path):
    source = tmp_path / 'source'
    source.mkdir()
    runtime = runtime_for(tmp_path, source)
    def ssh(args, **kwargs):
        assert args[3] == 'find', 'Must not upload after cleanup failure'
        return subprocess.run(args[3:], text=True, capture_output=True, check=True)
    runtime.ssh = ssh
    with pytest.raises(subprocess.CalledProcessError):
        runtime.push(str(source), str(tmp_path / 'missing-root'))
    assert (tmp_path / 'workspace-upload.tar').exists()
