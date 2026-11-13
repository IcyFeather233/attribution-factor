import json
from pathlib import Path

import pytest

from attribution_experiments.common.optimization_budget import pin_budget, adaptation_batches
from attribution_experiments.gepa.protocol import split_adaptation, prepare_engine_resume, proposal_budget
from attribution_experiments.harnessfix.upstream_bridge import validate_plan_consistency, validate_swe_targets
from baselines.harnessfix.failure_analysis.validation_metrics import compute_swe_metrics, evaluate_target_metrics
from baselines.harnessfix.failure_analysis.plan_diff_audit import audit_candidate


@pytest.mark.parametrize('blocked', ['src/config/...', 'src/config/**', 'src/config/', 'src/config/*.yaml'])
def test_precheck_rejects_same_paths_as_postcheck(blocked):
    plan = {'edit_budget': {'forbidden_paths': [blocked]},
            'fixes': [{'id': 'required', 'target_files': ['src/config/task.yaml']}]}
    with pytest.raises(ValueError, match='forbidden'):
        validate_plan_consistency(plan)
    plan['fixes'][0]['optional'] = True
    validate_plan_consistency(plan)


def test_real_three_iteration_plan_is_rejected_before_modification():
    root = Path(__file__).resolve().parents[2]
    path = root / 'attribution_experiments/harnessfix/baseline/native_alignment_3iter_20261002/iteration_01/repair/improvement_plan.json'
    with pytest.raises(ValueError, match='swebench.yaml'):
        validate_plan_consistency(json.loads(path.read_text()))


def test_unsupported_metrics_are_missing_not_zero_and_not_averaged():
    spec = {'fixes': [{'target_metrics': ['accuracy', 'resolved_rate']}]}
    with pytest.raises(ValueError, match='Unsupported SWE'):
        validate_swe_targets(spec)
    result = evaluate_target_metrics(spec, {'metrics': {'resolved_rate': .2}}, {'metrics': {'resolved_rate': .3}})
    assert result['unsupported_metrics'] == ['accuracy']
    assert result['metrics'][0]['baseline'] is None
    assert result['improved_metric_count'] == 1
    assert result['aggregate_delta'] == pytest.approx(.1)


def test_v2_metrics_measure_tool_errors_and_submission_rejections(tmp_path):
    traces = tmp_path / 'traces'
    for name, messages in {
        'bad': [{'role': 'tool', 'content': 'Traceback\nSyntaxError: invalid syntax'},
                {'role': 'tool', 'content': '<output>Submission rejected: missing patch</output>'}],
        'good': [{'role': 'tool', 'content': 'class SyntaxError(Exception): pass'}],
    }.items():
        folder = traces / name; folder.mkdir(parents=True)
        (folder / (name + '.traj.json')).write_text(json.dumps({'messages': messages}))
    evaluation = tmp_path / 'eval.json'
    evaluation.write_text(json.dumps({'error_ids': [], 'resolved_ids': ['good'], 'unresolved_ids': ['bad']}))
    metrics = compute_swe_metrics(traces, evaluation)['metrics']
    assert metrics['syntax_error_rate'] == .5
    assert metrics['invalid_submission_rate'] == .5
    assert metrics['error_rate'] == 0


def test_compilable_but_misnested_cleanup_is_rejected(tmp_path):
    original, candidate = tmp_path / 'original', tmp_path / 'candidate'
    rel = 'src/minisweagent/environments/docker.py'
    for base in [original, candidate]:
        (base / rel).parent.mkdir(parents=True)
    (original / rel).write_text('class DockerEnvironment:\n    def cleanup(self): pass\n')
    (candidate / rel).write_text('class DockerEnvironment: pass\ndef helper():\n    return\n    def cleanup(self): pass\n')
    report = audit_candidate(original, candidate, {'edit_budget': {'allowed_paths': ['src/...']}})
    assert not report['passed']
    assert report['syntax_failures'] == []
    assert 'candidate_interface_failures' in report['violations']


def test_repeats_use_distinct_batches_and_same_task_order():
    batches = list(adaptation_batches(list(range(32)), 4, 2))
    assert [x[0] for x in batches] == list(range(16))
    assert [x[1] for x in batches] == [0] * 8 + [1] * 8
    assert [x[2] for x in batches[:8]] == [x[2] for x in batches[8:]]


def test_expanded_training_keeps_the_identical_independent_validation_set():
    small_train, val = split_adaptation(list(range(32)), 10, validation_offset=10, validation_tasks=22)
    large_train, large_val = split_adaptation(list(range(42)), 20, validation_offset=10, validation_tasks=22)
    assert val == large_val
    assert large_train[:10] == small_train
    assert set(large_train).isdisjoint(val)
    with pytest.raises(ValueError):
        split_adaptation(list(range(32)), 20, validation_offset=10, validation_tasks=22)


def test_budget_pin_does_not_allow_overwrite_or_legacy_resume(tmp_path):
    pin_budget(tmp_path, {'repeats': 1}, has_progress=False)
    pin_budget(tmp_path, {'repeats': 1}, has_progress=True)
    with pytest.raises(ValueError, match='changed'):
        pin_budget(tmp_path, {'repeats': 2}, has_progress=True)
    assert json.loads((tmp_path / 'optimization_budget.json').read_text()) == {'repeats': 1}
    with pytest.raises(ValueError, match='Unmarked'):
        pin_budget(tmp_path / 'legacy', {}, has_progress=True)


def test_real_gepa_resume_keeps_completed_proposals_and_seed_validation(tmp_path):
    from gepa.optimize_anything import EngineConfig, GEPAConfig, ReflectionConfig, optimize_anything
    reflections, calls = [], []
    engine_dir = tmp_path / 'engine'
    def evaluator(candidate, example):
        calls.append((candidate, example))
        return 0.0, {'Feedback': 'Try a different instruction'}
    def reflect(prompt):
        reflections.append(prompt)
        return f'```\ncandidate_{len(reflections)}\n```'
    def run(stop=None):
        return optimize_anything(seed_candidate='seed', evaluator=evaluator,
            dataset=list(range(10)), valset=list(range(10, 32)),
            config=GEPAConfig(
                engine=EngineConfig(run_dir=str(engine_dir), max_candidate_proposals=4,
                                    max_metric_calls=proposal_budget(22, 3, 4),
                                    parallel=False, write_agent_state=True),
                reflection=ReflectionConfig(reflection_lm=reflect, reflection_minibatch_size=3),
                stop_callbacks=stop, merge=None, refiner=None))
    run(lambda state: state.i >= 1)
    assert len(reflections) == 2
    checkpoint = tmp_path / 'engine/gepa_state.bin'
    saved = checkpoint.read_bytes()
    assert prepare_engine_resume(tmp_path) == 0
    assert checkpoint.read_bytes() == saved
    run()
    assert len(reflections) == 4
    assert len(list((tmp_path / 'engine/iterations').glob('*/meta.json'))) == 5
    assert sum(candidate == 'seed' and example >= 10 for candidate, example in calls) == 22
    resumed_calls = list(calls)
    reflections.clear(); calls.clear()
    engine_dir = tmp_path / 'uninterrupted'
    run()
    assert calls == resumed_calls


def test_budget_plan_has_no_score_driven_jobs_and_distinct_paths():
    from attribution_experiments.plan_budget_repair import jobs
    rows = jobs(['baseline', 'group1_1_causing', 'group1_2_inevitable'])
    assert len(rows) == 15
    assert len({r['run_dir'] for r in rows}) == 15
    assert all(r['state'] == 'planned' for r in rows)
    assert all('native_profile' in ' '.join(r['command']) for r in rows if r['method'] == 'harnessfix')
