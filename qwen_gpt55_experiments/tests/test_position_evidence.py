import copy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from jinja2 import Environment, StrictUndefined

from qwen_gpt55_experiments.common import position_evidence as p
from qwen_gpt55_experiments.harnessfix.position_bridge import install_diagnosis, install_aggregate


def trace():
    return {'info': {'api_key': 'secret-not-for-model'}, 'messages': [
        {'role': 'system', 'content': 'task system'},
        {'role': 'assistant', 'content': 'OLD_CORRECT_STEP', 'extra': {'actions': [{'command': 'edit original'}]}},
        {'role': 'tool', 'content': 'OLD_OBSERVATION'},
        {'role': 'assistant', 'reasoning_content': 'SELECTED_SHIFTED_STEP',
         'extra': {'actions': [{'command': 'inspect shifted'}], 'api_key': 'secret-extra'}},
        {'role': 'tool', 'content': 'SHIFTED_OBSERVATION'},
        {'role': 'assistant', 'content': 'NEXT_ASSISTANT_NOT_INCLUDED'},
        {'role': 'exit', 'content': 'submitted'}]}


def payload(step=2, index=3):
    legacy = {'task_id': 'task-1', 'provided_information': {
        p.ANCHORS[0]: {'step': step, 'message_index': index, 'role': 'implementation',
                       'what_went_wrong': 'ANNOTATED_DIAGNOSIS', 'evidence': 'step 1 GOLD',
                       'recoverability': 'SOLUTION'}}}
    return p.convert_payload(legacy, trace(), 'trace-hash')


def test_shift_resolves_shifted_content_not_correct_diagnosis():
    x = payload()
    p.verify_payload(x, source_trace=trace())
    info = x['provided_information']; text = json.dumps(info)
    assert info[p.ANCHORS[0]] == {'step': 2, 'message_index': 3}
    assert 'SELECTED_SHIFTED_STEP' in text and 'SHIFTED_OBSERVATION' in text
    for absent in ['OLD_CORRECT_STEP', 'OLD_OBSERVATION', 'NEXT_ASSISTANT_NOT_INCLUDED',
                   'ANNOTATED_DIAGNOSIS', 'SOLUTION', 'GOLD', 'secret-extra', 'secret-not-for-model']:
        assert absent not in text
    assert info['historical_step_evidence'][p.ANCHORS[0]]['message_indices'] == [3, 4]
    assert ' --step 2 --message-index 3 ' in text


@pytest.mark.parametrize('anchor', [
    {'step': 1, 'message_index': 3}, {'step': 0, 'message_index': 0},
    {'step': True, 'message_index': 1}, {'step': 1, 'message_index': 1, 'role': 'wrong'},
])
def test_inconsistent_indices_rejected(anchor):
    with pytest.raises(ValueError):
        p.resolve_step(trace(), anchor)


def test_truncation_and_terminal_step_have_explicit_boundaries():
    t = trace();t['messages'][3]['reasoning_content'] = 'x' * 20000
    r = p.resolve_step(t, {'step': 2, 'message_index': 3}, excerpt_chars=300)
    assert r['truncated'] and len(r['raw_excerpt']) == 300 and r['full_chars'] > 20000
    assert 'truncated' in r['raw_excerpt']
    r = p.resolve_step(trace(), {'step': 3, 'message_index': 5})
    assert r['message_indices'] == [5, 6] and 'submitted' in r['raw_excerpt']


def test_raw_excerpt_and_extra_diagnosis_tampering_rejected():
    x = payload(); x['provided_information']['historical_step_evidence'][p.ANCHORS[0]]['raw_excerpt'] = 'old content'
    with pytest.raises(ValueError, match='mechanical'):
        p.verify_payload(x, source_trace=trace())
    x = payload();x['provided_information'][p.ANCHORS[0]]['what_went_wrong'] = 'leak'
    with pytest.raises(ValueError):
        p.verify_payload(x)


def test_baseline_stays_empty_and_both_resolves_each_position():
    x = p.convert_payload({'task_id': 'task-1', 'provided_information': {}}, trace(), 'hash')
    assert x['provided_information'] == {};p.verify_payload(x)
    info = {p.ANCHORS[0]: {'step': 1, 'message_index': 1}, p.ANCHORS[1]: {'step': 2, 'message_index': 3}}
    x = p.convert_payload({'task_id': 'task-1', 'provided_information': info}, trace(), 'hash')
    p.verify_payload(x, source_trace=trace())
    assert 'OLD_CORRECT_STEP' in x['provided_information']['historical_step_evidence'][p.ANCHORS[0]]['raw_excerpt']
    assert 'OLD_CORRECT_STEP' not in x['provided_information']['historical_step_evidence'][p.ANCHORS[1]]['raw_excerpt']


def test_harnessfix_raw_evidence_reaches_initial_diagnosis(tmp_path):
    (tmp_path / 'task-1.json').write_text(json.dumps(payload()))
    class Agent:
        def __init__(self): self.config = SimpleNamespace(instance_template='native task {{instance_id}}')
        def run(self, **kwargs):
            self.prompt = Environment(undefined=StrictUndefined).from_string(self.config.instance_template).render(**kwargs)
            return {}
    module = SimpleNamespace(DefaultAgent=Agent)
    install_diagnosis(module, tmp_path);agent = Agent();agent.run(instance_id='task-1')
    assert 'SELECTED_SHIFTED_STEP' in agent.prompt and 'SHIFTED_OBSERVATION' in agent.prompt
    assert 'ANNOTATED_DIAGNOSIS' not in agent.prompt


def test_harnessfix_aggregate_receives_diagnosis_before_first_plan_and_isolates_files(tmp_path):
    (tmp_path / 'task-1.json').write_text(json.dumps(payload()))
    captured = {}
    def call(**kwargs): captured.update(kwargs);return 'plan'
    module = SimpleNamespace(call_aggregate_agent=call)
    install_aggregate(module, tmp_path)
    result = module.call_aggregate_agent(mode_system_prompt='native', output_path=tmp_path / 'a/plan.md',
        results=[{'instance_id': 'task-1', 'failure_reason': 'NATIVE_DIAGNOSIS', 'agent_design_issue': 'NATIVE_DESIGN'}])
    assert result == 'plan'
    assert 'NATIVE_DIAGNOSIS' in captured['mode_system_prompt'] and 'message_index' in captured['mode_system_prompt']
    assert module._aggregate_context_dir(tmp_path / 'a/plan.md') != module._aggregate_context_dir(tmp_path / 'b/plan.md')
    assert json.loads((tmp_path / 'a/position_aggregate_delivery.json').read_text())['diagnoses_in_initial_prompt'] == 1


def test_gepa_actual_reflection_input_keeps_shifted_raw_content(monkeypatch, tmp_path):
    from qwen_gpt55_experiments.gepa import run_native as gepa
    def run(**kwargs):
        d = kwargs['output_dir'];(d / 'trajectory.json').write_text(json.dumps({'messages': []}))
        (d / 'patch.diff').write_text('diff --git a/x b/x\n')
        return {'exit_status': 'Submitted', 'api_calls': 1, 'evaluation': {'resolved': True}}
    monkeypatch.setattr(gepa, 'run_task', run)
    evaluator = gepa.NativeEvaluator(output_dir=tmp_path, api_key='unused', step_limit=250,
                                     analysis_payloads={'task-1': payload()})
    _, info = evaluator(gepa.build_seed_candidate(), {'instance_id': 'task-1', 'problem_statement': 'task'})
    evidence = info['Provided Failure Analysis (Trusted Evidence)']
    assert evidence[p.ANCHORS[0]]['message_index'] == 3
    assert 'SELECTED_SHIFTED_STEP' in json.dumps(evidence) and 'ANNOTATED_DIAGNOSIS' not in json.dumps(evidence)


def test_scope_actual_native_hook_gets_selected_historical_step():
    from qwen_gpt55_experiments.scope.run_native import ScopeAgent
    captured = {}
    class Optimizer:
        async def on_step_complete(self, **kwargs): captured.update(kwargs);return None
    fake = SimpleNamespace(env=SimpleNamespace(execute=lambda a: {'returncode': 0, 'output': 'current result'}),
        model=SimpleNamespace(format_observation_messages=lambda *a: [{'role': 'tool', 'content': 'current result'}]),
        get_template_vars=lambda: {}, add_messages=lambda *a: None, optimizer=Optimizer(),
        analysis_payload=payload(), task_id='task-1', n_calls=90, problem_statement='task',
        messages=[{'content': 'task prompt'}], guideline_events=[])
    ScopeAgent.execute_actions(fake, {'content': 'current reasoning', 'extra': {'actions': [{'command': 'current command'}]}})
    assert 'SELECTED_SHIFTED_STEP' in captured['task']
    assert 'current command' in captured['tool_calls'] and fake.messages[0]['content'] == 'task prompt'


def test_meta_actual_workspace_contains_resolved_message(monkeypatch, tmp_path):
    from qwen_gpt55_experiments.meta_harness import run_native as meta
    source = tmp_path / 'task-1.traj.json';source.write_text(json.dumps(trace()))
    monkeypatch.setattr(meta, 'trace_path', lambda iid: source)
    run = tmp_path / 'run';run.mkdir()
    search = meta.prepare_search_workspace(run, [{'instance_id': 'task-1', 'repo': 'test/repo'}], False,
                                           {'task-1': payload()}, True)
    actual = json.loads((search / 'provided_analysis/task-1.json').read_text())
    assert actual[p.ANCHORS[0]] == {'step': 2, 'message_index': 3}
    assert 'SELECTED_SHIFTED_STEP' in json.dumps(actual)
    assert (search / 'traces/task-1.traj.json').exists()


def test_recreate_native_optimizer_template_receives_position_evidence():
    from qwen_gpt55_experiments.recreate.position_adapter import instance_template, synthesis_template
    template = Path('baselines/recreate/src/recreate_agent/prompts/meta_instance.jinja2').read_text()
    evidence = json.dumps(payload()['provided_information'])
    success = dict(instance_id='task-1', problem_summary='task', n_steps=3, total_cost=0,
                   good_decisions=[], provided_failure_analysis=evidence)
    rendered = Environment(undefined=StrictUndefined).from_string(instance_template(template)).render(recent_failures=[], recent_successes=[success])
    assert 'SELECTED_SHIFTED_STEP' in rendered and 'ANNOTATED_DIAGNOSIS' not in rendered
    batch = Path('baselines/recreate/src/recreate_agent/prompts/meta_batch_synthesis.jinja2').read_text()
    mod = dict(instance_id='task-1', success=True, score=1, summary='native proposal', diff='diff',
               historical_flow_evidence={'text': evidence, 'handoff': {'protocol': p.PROTOCOL}})
    rendered = Environment(undefined=StrictUndefined).from_string(synthesis_template(batch)).render(
        batch_idx=0, num_instances=1, num_modifications=1, modifications=[mod])
    assert 'SELECTED_SHIFTED_STEP' in rendered and 'native proposal' in rendered


def test_new_common_boundary_keeps_flow_raw_steps_and_freezes_receipt(monkeypatch, tmp_path):
    from qwen_gpt55_experiments.common import analysis_payloads, flow_evidence
    from qwen_gpt55_experiments.position_native import install_common_interfaces
    from qwen_gpt55_experiments.tests.test_native_flow_evidence import payload as graph_payload
    for module, names in [(analysis_payloads, ['load_payloads', 'snapshot_payloads', 'render_payload']),
                          (flow_evidence, ['packet', 'render'])]:
        for name in names:
            monkeypatch.setattr(module, name, getattr(module, name))
    old = graph_payload();old['provided_information'][p.ANCHORS[0]]['message_index'] = 3
    converted = p.convert_payload(old, trace(), 'hash')
    install_common_interfaces()
    evidence = flow_evidence.packet(converted)
    assert 'SELECTED_SHIFTED_STEP' in json.dumps(evidence['historical_step_evidence'])
    assert 'negative test' in flow_evidence.render(converted)
    assert 'SELECTED_SHIFTED_STEP' in flow_evidence.render(converted)
    analysis_payloads.snapshot_payloads(tmp_path, 'group3_1_causing_flow_context', {'task-1': converted})
    receipt = json.loads((tmp_path / 'position_evidence_protocol.json').read_text())
    assert receipt['protocol'] == p.PROTOCOL
    assert json.loads((tmp_path / 'provided_analysis/task-1.json').read_text()) == converted
