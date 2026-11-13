import json
from types import SimpleNamespace

from qwen_gpt55_experiments.tests.test_native_flow_evidence import payload


def test_gepa_reflection_keeps_historical_failure_separate_from_current_success(monkeypatch, tmp_path):
    from qwen_gpt55_experiments.gepa import run_native as gepa

    def fake_run(**kwargs):
        directory = kwargs['output_dir']
        (directory / 'trajectory.json').write_text(json.dumps({'messages': []}))
        (directory / 'patch.diff').write_text('diff --git a/a b/a\n')
        return {'exit_status': 'Submitted', 'api_calls': 1, 'evaluation': {'resolved': True}}

    monkeypatch.setattr(gepa, 'run_task', fake_run)
    evaluator = gepa.NativeEvaluator(output_dir=tmp_path, api_key='unused', step_limit=250,
                                     analysis_payloads={'task-1': payload()})
    score, info = evaluator(gepa.build_seed_candidate(), {'instance_id': 'task-1', 'problem_statement': 'task'})
    evidence = info['Provided Failure Analysis (Trusted Evidence)']
    assert score == 1 and evidence['current_candidate_rollout']['resolved'] is True
    assert evidence['historical_original_failure']['windows']['failure_causing_wrong_step']['anchor']['step'] == 2
    assert 'Historical step numbers' in evidence['current_candidate_rollout']['instruction']
    assert 'Agent Trace' in info['Generated Outputs']


def test_scope_native_hook_receives_current_action_with_distinct_historical_graph():
    from qwen_gpt55_experiments.scope.run_native import ScopeAgent
    captured = {}

    class Optimizer:
        async def on_step_complete(self, **kwargs):
            captured.update(kwargs)
            return None

    model = SimpleNamespace(format_observation_messages=lambda *args: [{'role': 'tool', 'content': 'current test passed'}])
    fake = SimpleNamespace(
        env=SimpleNamespace(execute=lambda action: {'returncode': 0, 'output': 'current test passed'}),
        model=model, get_template_vars=lambda: {}, add_messages=lambda *messages: None,
        optimizer=Optimizer(), analysis_payload=payload(), task_id='task-1', n_calls=80,
        problem_statement='current task', messages=[{'content': 'unchanged current prompt'}], guideline_events=[])
    ScopeAgent.execute_actions(fake, {'content': 'verify', 'extra': {'actions': [{'command': 'pytest test_current'}]}})
    assert 'current model call=80' in captured['task']
    assert 'original step=2' in captured['task']
    assert 'NOT indices' in captured['task']
    assert 'pytest test_current' in captured['tool_calls']
    assert 'current test passed' in captured['observations']
    assert fake.messages[0]['content'] == 'unchanged current prompt'
