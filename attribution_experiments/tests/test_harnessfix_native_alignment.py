import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from attribution_experiments.harnessfix.native_profile import analysis_ids, native_promotion, install
from attribution_experiments.harnessfix.upstream_bridge import validate_plan_consistency, attribution_summary, install_flow_diagnosis


def test_resolved_cases_cannot_be_diagnosed_as_failures_or_regressions(tmp_path):
    path = tmp_path / 'eval.json'
    path.write_text(json.dumps({'resolved_ids': ['fixed'], 'unresolved_ids': ['failed'], 'error_ids': ['error']}))
    assert analysis_ids(path, ['fixed', 'failed']) == ['failed']
    assert analysis_ids(path, None) == ['error', 'failed']
    assert analysis_ids(path, ['fixed']) == []


def test_plan_rejects_required_child_of_forbidden_directory_without_mutating_spec():
    spec = {'fixes': [{'id': 'f', 'target_files': ['src/config/task.yaml']}],
            'edit_budget': {'forbidden_paths': ['src/config/']}}
    before = json.dumps(spec)
    with pytest.raises(ValueError, match='Required fixes target forbidden'):
        validate_plan_consistency(spec)
    assert json.dumps(spec) == before
    spec['fixes'][0]['required'] = False
    validate_plan_consistency(spec)


def test_native_promotion_requires_target_metrics_and_error_bounds(tmp_path):
    gate = {'net_change': 2, 'target_metric_results': {'improved_metric_count': 1},
            'baseline_metrics': {'metrics': {'error_rate': 0, 'invalid_submission_rate': 0}},
            'current_metrics': {'metrics': {'error_rate': .1, 'invalid_submission_rate': 0}}}
    assert native_promotion(tmp_path, gate)['promoted']
    gate['target_metric_results']['improved_metric_count'] = 0
    assert not native_promotion(tmp_path, gate)['promoted']
    gate['target_metric_results']['improved_metric_count'] = 1
    gate['current_metrics']['metrics']['error_rate'] = .3
    assert not native_promotion(tmp_path, gate)['promoted']


def test_native_anchor_interface_retains_shifted_indices_and_full_diagnosis(tmp_path, monkeypatch):
    monkeypatch.setenv('HARNESSFIX_NATIVE_PROFILE', '1')
    payload = {'failure_causing_wrong_step': {'step': 29, 'message_index': 58, 'role': 'implementation',
                                            'what_went_wrong': 'ignored null branch', 'evidence': 'trace line'}}
    file = tmp_path / 'task.json'
    file.write_text(json.dumps({'provided_information': payload}))
    class Agent:
        def __init__(self): self.config = SimpleNamespace(instance_template='native')
        def run(self, **kwargs): self.kwargs = kwargs; return {}
    module = SimpleNamespace(DefaultAgent=Agent)
    install_flow_diagnosis(module, tmp_path)
    agent = Agent();agent.run(instance_id='task')
    assert json.loads(agent.kwargs['historical_anchor_evidence']) == payload
    before = attribution_summary(tmp_path, [{'instance_id': 'task'}])
    payload['failure_causing_wrong_step']['step'] = 31
    payload['failure_causing_wrong_step']['message_index'] = 62
    file.write_text(json.dumps({'provided_information': payload}))
    after = attribution_summary(tmp_path, [{'instance_id': 'task'}])
    assert before != after and 'ignored null branch' in after


def test_native_config_removes_clamp_and_filtered_diagnosis_keeps_native_entry(tmp_path, monkeypatch):
    def configure(run, inject):
        path = run / 'config.yaml'
        path.write_text(yaml.safe_dump({'agent': {'instance_template': 'STRICT CONTRACT\n## Your Assignment\nread evidence', 'step_limit':40}}))
        return path
    called = []
    def analyze(directory, **kwargs): called.append(kwargs['selected_ids']);return 'native-called'
    adapter = SimpleNamespace(prepare_analysis_config=configure, run_analysis=analyze, run_gate=lambda *a,**k:None,
                              run_benchmark=lambda *a,**k:[], redact_api_key=lambda *a:None,
                              prepare_run=lambda *a,**k:None,
                              write_json=lambda p,v:p.write_text(json.dumps(v)))
    install(adapter)
    config = yaml.safe_load(adapter.prepare_analysis_config(tmp_path, False).read_text())
    assert config['agent']['instance_template'].startswith('## Your Assignment')
    assert config['agent']['step_limit']==20
    evaluation = tmp_path/'eval.json';evaluation.write_text('{"resolved_ids":["a"],"unresolved_ids":["b"]}')
    assert adapter.run_analysis(tmp_path,eval_path=evaluation,selected_ids=['a','b'])=='native-called'
    assert called==[['b']]
