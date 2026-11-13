import json
import os
import sys
from types import SimpleNamespace

import pytest

from qwen_gpt55_experiments.common import remote_runtime as remote
from qwen_gpt55_experiments.meta_harness.timed_task import run_bounded
from qwen_gpt55_experiments.meta_harness.input_audit import audit_proposer_inputs


def test_admission_waits_for_real_remote_capacity(tmp_path, monkeypatch):
    monkeypatch.setattr(remote, 'runtime_root', lambda: tmp_path / 'containers')
    monkeypatch.setenv('UDOCKER_MAX_EVALUATORS', '1')
    monkeypatch.setattr(remote.time, 'sleep', lambda _: None)
    runtime = remote.RemoteRuntime.__new__(remote.RemoteRuntime)
    runtime.role = 'evaluator'
    counts = iter(['occupied\n', 'occupied\n', ''])
    calls = []
    def ssh(command):
        calls.append(command)
        return SimpleNamespace(stdout=next(counts))
    runtime.ssh = ssh
    with runtime.admission():
        assert len(calls) == 3


def test_admission_timeout_does_not_create_container(tmp_path, monkeypatch):
    monkeypatch.setattr(remote, 'runtime_root', lambda: tmp_path / 'containers')
    monkeypatch.setenv('EXPERIMENT_CAPACITY_WAIT_SECONDS', '0')
    runtime = remote.RemoteRuntime.__new__(remote.RemoteRuntime)
    runtime.role = 'agent'
    with pytest.raises(remote.RemoteInfrastructureError, match='capacity wait timed out'):
        with runtime.admission():
            pytest.fail('Must not admit')


def test_timeout_cleanup_runs_while_child_is_frozen(tmp_path):
    pidfile = tmp_path / 'pid'
    command = [sys.executable, '-c',
               'import os,time,pathlib,sys; pathlib.Path(sys.argv[1]).write_text(str(os.getpid())); time.sleep(60)', str(pidfile)]
    states = []
    def cleanup():
        from pathlib import Path
        pid = int(pidfile.read_text())
        # SIGSTOP delivery can lag very briefly.
        import time
        for _ in range(100):
            state = Path(f'/proc/{pid}/stat').read_text().rsplit(')', 1)[1].split()[0]
            if state == 'T': break
            time.sleep(.01)
        states.append(state)
    assert run_bounded(command, env=os.environ.copy(), log=tmp_path/'log', timeout=.5,
                       before_terminate=cleanup) == 124
    assert states == ['T']


def test_ascii_escaped_frozen_evidence(tmp_path):
    search = tmp_path / 'search'
    (search/'traces').mkdir(parents=True)
    (search/'logs').mkdir()
    (tmp_path/'provided_analysis').mkdir()
    (search/'traces/task.traj.json').write_text('{}')
    evidence = '冻结的分析材料说明 gold_patch 的差异，这里只是引用已授权分析文本，不读取其他材料。'
    data = {'provided_information': {'evidence': evidence}}
    (tmp_path/'provided_analysis/task.json').write_text(json.dumps(data))
    def check(output):
        messages = [{'role':'assistant', 'extra':{'actions':[{'command':'cat provided_analysis/task.json','tool_call_id':'x'}]}},
                    {'role':'tool','tool_call_id':'x','content':output,'extra':{'raw_output':output}}]
        (search/'logs/analysis.traj.json').write_text(json.dumps({'messages':messages}))
        return audit_proposer_inputs(search)['passed']
    assert check(json.dumps(data, ensure_ascii=True))
    assert not check(json.dumps(data, ensure_ascii=True) + ' additional gold_patch material')


def test_stop_task_without_asset_mounts(tmp_path, monkeypatch):
    from qwen_gpt55_experiments.meta_harness import timed_task
    monkeypatch.setattr(timed_task, 'runtime_root', lambda: tmp_path/'containers')
    registry = tmp_path/'remote_registry'
    registry.mkdir()
    task = tmp_path/'task'
    task.mkdir()
    name = 'fa-' + 'a'*32
    (registry/'owned.json').write_text(json.dumps(dict(container_name=name, task_dir=str(task), mounts={})))
    (registry/'other.json').write_text(json.dumps(dict(container_name='other', task_dir=str(tmp_path/'other'), mounts={})))
    calls = []
    runtime = SimpleNamespace(container_name=name)
    def ssh(args, **kwargs):
        calls.append(args)
        if args[1:3] == ['inspect', '--format']:
            return SimpleNamespace(returncode=0, stdout='false\n')
        if args[1] == 'inspect':
            return SimpleNamespace(returncode=0, stdout=json.dumps([{'Config':{'Labels':{remote.LABEL:name}}}]))
        return SimpleNamespace(returncode=0, stdout=name)
    runtime.ssh = ssh
    monkeypatch.setattr(remote.RemoteRuntime, 'attach', lambda n: runtime if n==name else pytest.fail('Wrong task'))
    timed_task.stop_owned(task)
    assert calls[1] == ['docker','stop','-t','5',name]
    assert calls[2][-1] == name
