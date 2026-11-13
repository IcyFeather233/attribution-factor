from types import SimpleNamespace
import pytest


def test_budget_starts_at_construction_and_only_bounds_task_environments(monkeypatch):
    from qwen_gpt55_experiments.recreate import run_native as runner
    from minisweagent.agents import default as native
    base = native.DefaultAgent
    monkeypatch.setattr(native, 'DefaultAgent', base)
    monkeypatch.setattr(base, 'query', lambda self: 'native_query')
    monkeypatch.setenv('EXPERIMENT_ALIGNED_TASK_BUDGET', '1')
    clock = [0]
    monkeypatch.setattr(runner.time, 'monotonic', lambda: clock[0])
    class TaskEnvironment:
        pass
    monkeypatch.setattr(runner, 'V1UdockerEnvironment', TaskEnvironment)
    runner.install_aligned_task_budget()
    task = native.DefaultAgent(SimpleNamespace(), TaskEnvironment())
    optimizer = native.DefaultAgent(SimpleNamespace(), object())
    assert task.query() == 'native_query'
    clock[0] = 3601
    with pytest.raises(native.TerminatingException, match='3600'):
        task.query()
    assert optimizer.query() == 'native_query'
    same = native.DefaultAgent
    runner.install_aligned_task_budget()
    assert native.DefaultAgent is same
