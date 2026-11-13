import json
import subprocess
from types import SimpleNamespace

import pytest

from attribution_experiments.common.remote_runtime import RemoteRuntime, RemoteInfrastructureError


def test_stopped_container_invalidates_run(tmp_path):
    runtime = RemoteRuntime.__new__(RemoteRuntime)
    runtime.local_dir, runtime.mounts, runtime.sync_failed = tmp_path, {}, False
    runtime.config = SimpleNamespace(cwd='/testbed')
    runtime.container_name = 'own'
    def ssh(args, **kwargs):
        if args[1] == 'inspect':
            return SimpleNamespace(returncode=0, stdout='{"Running": false, "OOMKilled": true}')
        return subprocess.CompletedProcess(args, 0, 'container is not running\n__FA_EXEC_STATUS_' + args[3] + '__:1\n', None)
    runtime.ssh = ssh
    with pytest.raises(RemoteInfrastructureError, match='stopped'):
        runtime.run(['run', '--nobanner', 'own', 'true'], timeout=1)
    assert runtime.failed
    with pytest.raises(RemoteInfrastructureError, match='invalidated'):
        runtime.run(['run', '--nobanner', 'own', 'true'], timeout=1)


def test_infrastructure_failure_is_not_a_negative_grade(tmp_path, monkeypatch):
    from attribution_experiments.common import evaluator
    class Environment:
        def __init__(self, **kwargs): pass
        def execute(self, *args, **kwargs):
            raise RemoteInfrastructureError('container killed by resource guard')
        def cleanup(self): pass
    monkeypatch.setattr(evaluator, 'UdockerEnvironment', Environment)
    monkeypatch.setattr(evaluator, 'make_test_spec', lambda x: SimpleNamespace(eval_script='true'))
    with pytest.raises(RemoteInfrastructureError):
        evaluator.evaluate_patch(instance={'instance_id': 'test__task-1'}, patch='nonempty',
                                 output_dir=tmp_path, udocker_repo=tmp_path, udocker_executable=tmp_path / 'udocker')
    report = json.loads((tmp_path / 'report.json').read_text())
    assert report['resolved'] is None
    assert report['error'] == 'remote_infrastructure_error'
    assert 'official_report' not in report
