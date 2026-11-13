import json

from attribution_experiments.common import analysis_payloads as legacy
from attribution_experiments.common import position_evidence as position


def test_all_conditions_and_shifted_positions(tmp_path, monkeypatch):
    task = 'synthetic-1'
    trace = {'messages': [{'role': 'system', 'content': 'task'}] + [
        {'role': 'assistant', 'content': f'step-{i}'} for i in range(1, 8)
    ]}
    trace_path = tmp_path / 'trace.json'
    trace_path.write_text(json.dumps(trace))
    a, b = position.ANCHORS
    anchors = {a: {'step': 3, 'message_index': 3, 'what_went_wrong': 'diagnosis-a'},
               b: {'step': 5, 'message_index': 5, 'what_went_wrong': 'diagnosis-b'}}
    analysis = {'schema_version': '2.0', 'classification': 'confirmed_agent_failure',
                'trace': {'path': str(trace_path)}, 'step_attribution': anchors}
    (tmp_path / f'{task}.analysis.json').write_text(json.dumps(analysis))
    def window(center):
        return {'center_step': center, 'window_steps': list(range(max(1, center - 3), min(7, center + 3) + 1)),
                'control_flow': {'nodes': [], 'edges': []}, 'data_flow': {'nodes': [], 'edges': []}}
    graph = {'schema_version': '1.0', 'task_id': task, 'canonical_scaffold_version': 'test-agent-v1',
             'windows': {a: window(3), b: window(5)}}
    (tmp_path / f'{task}.graph.json').write_text(json.dumps(graph))
    monkeypatch.setattr(legacy, 'ANALYSIS_ROOT', tmp_path)
    monkeypatch.setattr(legacy, 'GRAPH_ANALYSIS_ROOT', tmp_path)
    payloads = legacy.build_task_payloads(task)
    assert set(payloads) == set(legacy.GROUPS)
    assert payloads['baseline']['provided_information'] == {}
    for group, payload in payloads.items():
        converted = position.convert_payload(payload, trace, 'synthetic-sha')
        position.verify_payload(converted, source_trace=trace)
        info = converted['provided_information']
        assert 'diagnosis-a' not in json.dumps(info) and 'diagnosis-b' not in json.dumps(info)
        if 'shifted' in group:
            for key in set(info).intersection(position.ANCHORS):
                assert info[key]['step'] != anchors[key]['step']
                assert f"step-{info[key]['step']}" in info['historical_step_evidence'][key]['raw_excerpt']
        if 'flow_context' in group:
            assert any(key.endswith('_context') for key in info)
    ids = tmp_path / 'task_ids.txt'
    ids.write_text(task + '\n')
    monkeypatch.setattr(legacy, 'FAILED_IDS', ids)
    legacy_root = tmp_path / 'legacy'
    legacy.generate_payloads(legacy_root)
    monkeypatch.setattr(legacy, 'DEFAULT_PAYLOAD_ROOT', legacy_root)
    position_root = tmp_path / 'position'
    manifest = position.generate(position_root)
    assert len(manifest['groups']) == 10
    for group in legacy.GROUPS:
        assert set(position.read_payloads(group, position_root)) == {task}


def test_plan_uses_current_task_count(monkeypatch):
    from attribution_experiments import plan_budget_repair
    monkeypatch.setattr(legacy, 'task_ids', lambda: [f'task-{i}' for i in range(20)])
    jobs = plan_budget_repair.jobs(list(legacy.GROUPS))
    assert len(jobs) == 50
    assert {job['method'] for job in jobs} == {'gepa', 'scope', 'recreate', 'meta_harness', 'harnessfix'}
    for job in jobs:
        flag = '--analysis-tasks' if job['method'] == 'harnessfix' else '--adaptation-tasks'
        assert job['command'][job['command'].index(flag) + 1] == '20'


def test_model_configuration_uses_generic_environment(monkeypatch):
    from attribution_experiments.common import model_config
    monkeypatch.setenv('TASK_MODEL_API_KEY', 'task-test-key')
    monkeypatch.setenv('OPTIMIZER_API_KEY', 'optimizer-test-key')
    monkeypatch.setenv('TASK_MODEL_EXTRA_BODY', '{"custom_option": true}')
    assert model_config.task_api_key() == 'task-test-key'
    assert model_config.optimizer_api_key() == 'optimizer-test-key'
    assert model_config.task_model_kwargs(api_key='test')['extra_body'] == {'custom_option': True}
