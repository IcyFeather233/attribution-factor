import json
from pathlib import Path
from types import SimpleNamespace

from qwen_gpt55_experiments.common.remote_capacity import agent_capacity
from qwen_gpt55_experiments.recreate.resume_state import recovered_batch_result


def test_capacity_32_uses_same_shared_slots(tmp_path, monkeypatch):
    from qwen_gpt55_experiments.common import udocker_runtime as runtime
    profile = tmp_path / 'capacity.json'
    profile.write_text('{"agent_containers": 32}')
    monkeypatch.setenv('EXPERIMENT_CONTAINER_BACKEND', 'remote_docker')
    monkeypatch.setenv('EXPERIMENT_REMOTE_CAPACITY_FILE', str(profile))
    monkeypatch.setenv('UDOCKER_RUNTIME_ROOT', str(tmp_path / 'containers'))
    assert agent_capacity() == 32
    handles = []
    try:
        for _ in range(32):
            handles.append(runtime._acquire_slot('resident', 16))
        assert len({h.name for h in handles}) == 32
        assert handles[-1].name.endswith('slot-031.lock')
    finally:
        for handle in handles:
            handle.close()


def test_recreate_completed_update_is_recovered_without_new_model_calls(tmp_path):
    old, task = tmp_path / 'global', tmp_path / 'task'
    for root in (old, task):
        (root / 'agent_tools').mkdir(parents=True)
        (root / 'agent_memory').mkdir()
        (root / 'agent_memory/memories.yaml').write_text('old')
    (task / 'agent_memory/memories.yaml').write_text('learned')
    (task / 'scaffold.yaml').write_text('agent: {}')
    (task / 'scaffold_diff.txt').write_text('changed')
    (task / 'agent_tools/new_tool').mkdir()
    (task / 'eval_result.json').write_text('{"success": false, "score": 0, "error": ""}')
    (task / 'case.traj.json').write_text('{"info": {"exit_status": "Submitted"}}')
    result = recovered_batch_result(task, old, 'case')
    assert result['success'] is False
    assert result['scaffold_changed'] and result['has_new_tools'] and result['has_new_memories']
    (task / 'recreate_temp').mkdir()
    import pytest
    with pytest.raises(RuntimeError, match='unfinished'):
        recovered_batch_result(task, old, 'case')


def test_scope_partial_batch_keeps_completed_branch_and_merges_once(tmp_path, monkeypatch):
    from qwen_gpt55_experiments.scope import run_native as scope
    run = tmp_path / 'group1_1_causing' / 'resume'
    memory = run / 'adaptation_batches/batch_0000/a/scope_memory'
    memory.mkdir(parents=True)
    (memory / 'learned.txt').write_text('keep this guideline')
    result = run / 'adaptation/a/result.json'
    result.parent.mkdir(parents=True)
    result.write_text(json.dumps(dict(instance_id='a', api_calls=4, guideline_count=1)))
    instances = [dict(instance_id=k) for k in ['a', 'b']]
    monkeypatch.setattr(scope, 'RUN_ROOT', tmp_path)
    monkeypatch.setattr(scope, 'optimizer_api_key', lambda: 'test')
    monkeypatch.setattr(scope, 'task_api_key', lambda: 'test')
    monkeypatch.setattr(scope, 'load_failed_instances', lambda: instances)
    monkeypatch.setattr(scope, 'select_repository_balanced_instances', lambda items, **kw: items)
    monkeypatch.setattr(scope, 'load_payloads', lambda *a: dict(a={}, b={}))
    monkeypatch.setattr(scope, 'snapshot_payloads', lambda *a: None)
    called, merges = [], []
    def adapt(**kwargs):
        called.append(kwargs['instance']['instance_id'])
        return dict(instance_id='b', api_calls=4, guideline_count=1)
    def merge(**kwargs):
        assert (memory / 'learned.txt').read_text() == 'keep this guideline'
        merges.append([i for i, _ in kwargs['branches']])
        return {}
    monkeypatch.setattr(scope, 'run_adaptation_task', adapt)
    monkeypatch.setattr(scope, 'merge_batch_memories', merge)
    monkeypatch.setattr(scope, 'make_optimizer', lambda **kw: SimpleNamespace(get_strategic_rules_for_agent=lambda _: 'rules'))
    monkeypatch.setattr(scope, 'run_task', lambda **kw: dict(api_calls=1, evaluation=dict(resolved=False)))
    monkeypatch.setattr('sys.argv', ['scope', '--ablation-group', 'group1_1_causing', '--run-name', 'resume',
                                  '--adaptation-tasks', '2', '--adaptation-batch-size', '2', '--resume'])
    scope.main()
    assert called == ['b']
    assert merges == [['a', 'b']]
    assert json.loads((run / 'adaptation_batches/batch_0000/commit.json').read_text())['instance_ids'] == ['a', 'b']
