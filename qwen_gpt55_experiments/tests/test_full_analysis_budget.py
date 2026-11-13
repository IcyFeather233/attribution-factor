import json
import sys
from types import SimpleNamespace
import pytest

from qwen_gpt55_experiments.harnessfix.native_profile import install
from qwen_gpt55_experiments.meta_harness.corpus_index import write_corpus_index
from qwen_gpt55_experiments.plan_budget_repair import jobs


def test_full_diagnosis_keeps_validation_separate_and_current_failures_current(tmp_path, monkeypatch):
    all_ids = [f'task-{i:03d}' for i in range(148)]
    validation = all_ids[:32]
    calls = []
    def write(path, data):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data))
    def prepare(run, **kwargs):
        run.mkdir(exist_ok=True)
        (run / 'instance_ids.txt').write_text('\n'.join(all_ids))
        (run / 'validation_ids.txt').write_text('\n'.join(validation))
        write(run / 'baseline/eval.json', {'resolved_ids': [], 'unresolved_ids': all_ids})
        return {'paths': {'validation_ids': str(run / 'validation_ids.txt')}}
    def analyze(iteration, **kwargs):
        calls.append(kwargs)
        return 'native-analysis'
    def forbidden(*args, **kwargs):
        raise AssertionError('prepare-only must not invoke benchmarks or load API keys')
    adapter = SimpleNamespace(write_json=write, prepare_run=prepare,
        select_validation_ids=lambda ids, limit, seed: ids[:limit],
        run_analysis=analyze, run_benchmark=forbidden, optimizer_api_key=forbidden,
        prepare_analysis_config=lambda *a: None, run_gate=lambda *a, **k: None,
        redact_api_key=lambda *a: None)
    install(adapter, analysis_tasks=148)
    monkeypatch.setattr(sys, 'argv', ['native_profile', '--prepare-only'])
    manifest = adapter.prepare_run(tmp_path, validation_seed=20260902)
    assert manifest['analysis_task_count'] == 148
    assert (tmp_path / 'validation_ids.txt').read_text().split() == validation
    iteration = tmp_path / 'iteration_01'; iteration.mkdir()
    adapter.run_analysis(iteration, eval_path=tmp_path / 'baseline/eval.json', selected_ids=validation)
    assert calls[-1]['selected_ids'] == all_ids
    adapter.write_json(iteration / 'iteration_report.json', {'analysis_count': 32})
    assert json.loads((iteration / 'iteration_report.json').read_text())['analysis_count'] == 148
    current = iteration / 'validation/eval.json'
    write(current, {'resolved_ids': validation[:2], 'unresolved_ids': validation[2:]})
    second = tmp_path / 'iteration_02'; second.mkdir()
    adapter.run_analysis(second, eval_path=current, selected_ids=validation)
    assert calls[-1]['selected_ids'] == validation[2:]
    adapter.run_analysis(second, eval_path=current, selected_ids=validation[3:5], label='val_regressions')
    assert calls[-1]['selected_ids'] == validation[3:5]


def test_meta_index_contains_all_records_without_invented_diagnoses(tmp_path):
    (tmp_path / 'traces').mkdir()
    instances = [{'instance_id': f'task-{i}', 'repo': 'example', 'gold_patch': 'must not expose'} for i in range(148)]
    path = write_corpus_index(tmp_path, instances, True)
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    assert len(rows) == 148 and len({r['instance_id'] for r in rows}) == 148
    assert 'gold_patch' not in path.read_text() and 'must not expose' not in path.read_text()
    assert rows[0]['provided_analysis'] == 'provided_analysis/task-0.json'
    write_corpus_index(tmp_path, instances, False)
    assert all(json.loads(line)['provided_analysis'] is None for line in path.read_text().splitlines())


def test_full_analysis_settings_survive_onepass_alignment():
    rows = jobs(['baseline'])
    hf = next(r['command'] for r in rows if r['method'] == 'harnessfix')
    assert hf[hf.index('--analysis-tasks') + 1] == '148'
    assert hf[hf.index('--validation-tasks') + 1] == '32'
    assert hf[hf.index('--max-iterations') + 1] == '1'
    meta = next(r['command'] for r in rows if r['method'] == 'meta_harness')
    assert meta[meta.index('--adaptation-tasks') + 1] == '148'
    for row in rows:
        if row['method'] == 'recreate':
            assert row['command'][row['command'].index('--adaptation-tasks') + 1] == '148'
    from qwen_gpt55_experiments.run_pipeline import command
    hf_default = command('harnessfix', 'baseline')
    assert 'qwen_gpt55_experiments.harnessfix.native_profile' in hf_default
    assert hf_default[hf_default.index('--analysis-tasks') + 1] == '148'
    assert hf_default[hf_default.index('--validation-tasks') + 1] == '32'
    meta_default = command('meta_harness', 'baseline')
    assert meta_default[meta_default.index('--adaptation-tasks') + 1] == '148'


@pytest.mark.parametrize('inject', [False, True])
def test_meta_workspace_exposes_all_traces_and_preserves_existing_search(tmp_path, monkeypatch, inject):
    from qwen_gpt55_experiments.meta_harness import run_native as runner
    source = tmp_path / 'source'; source.mkdir()
    instances = [{'instance_id': f'task-{i}', 'repo': 'example'} for i in range(148)]
    for instance in instances:
        (source / (instance['instance_id'] + '.traj.json')).write_text('{"messages": []}')
    monkeypatch.setattr(runner, 'trace_path', lambda iid: source / (iid + '.traj.json'))
    monkeypatch.setattr(runner, 'load_original_config', lambda: {
        'agent': {k: 'test' for k in runner.AGENT_TEXT_FIELDS},
        'model': {k: 'test' for k in runner.MODEL_TEXT_FIELDS}})
    payloads = {instance['instance_id']: {'provided_information': {}} for instance in instances}
    run = tmp_path / 'run'
    search = runner.prepare_search_workspace(run, instances, False, payloads, inject)
    assert len(list((search / 'traces').glob('*.traj.json'))) == 148
    assert len((search / 'traces/_index.jsonl').read_text().splitlines()) == 148
    assert len(list((search / 'provided_analysis').glob('*.json'))) == (148 if inject else 0)
    with pytest.raises(RuntimeError, match='already exists'):
        runner.prepare_search_workspace(run, instances, False, payloads, inject)
    assert len(list((search / 'traces').glob('*.traj.json'))) == 148
