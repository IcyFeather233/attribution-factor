import json

from qwen_gpt55_experiments.launch_reviewed_reruns import jobs
from qwen_gpt55_experiments import scheduled_recovery as scheduled


def test_only_incompatible_cells_are_fresh_runs_and_baseline_is_final_only():
    groups = {
        'gepa': ['group1_1', 'group1_2', 'group1_3', 'group2_3'],
        'scope': ['group1_1', 'group1_2', 'group1_3', 'group2_1', 'group2_2', 'group2_3'],
        'meta_harness': ['group1_1', 'group1_2', 'group1_3', 'group2_1', 'group2_2', 'group2_3'],
        'recreate': ['group1_2', 'group1_3', 'group2_1', 'group2_2', 'group2_3', 'group3_1', 'group3_2', 'group3_3'],
    }
    rows = [{'method': m, 'group': g, 'verdict': 'requires_review', 'run': 'old/run',
             'profile_differences': ['wall_time'], 'errors': []} for m, values in groups.items() for g in values]
    rows.append({'method': 'scope', 'group': 'group3_1', 'verdict': 'verified_reuse'})
    planned = jobs({'cells': rows})
    assert len(planned) == len({j['id'] for j in planned}) == 25
    assert 'qwen_gpt55_experiments.recover_remaining_finals' in planned[0]['command']
    assert planned[0]['run_name'] == 'aligned_baseline_20261002'
    assert {(j['method'], j['group']) for j in planned[1:]} == {(r['method'], r['group']) for r in rows[:-1]}
    assert all('--resume' not in j['command'] and j['run_name'] == 'aligned_reaudit_20261013' for j in planned[1:])


def test_scheduler_rejects_complete_summary_with_one_missing_recreate_report(tmp_path, monkeypatch):
    monkeypatch.setattr(scheduled, 'ROOT', tmp_path)
    package = tmp_path / 'qwen_gpt55_experiments'
    package.mkdir()
    ids = [f'task-{i}' for i in range(148)]
    (package / 'task_ids.txt').write_text('\n'.join(ids))
    run = package / 'recreate'
    rows = [{'instance_id': iid, 'resolved': False} for iid in ids]
    for row in rows[:-1]:
        report = run / 'final' / row['instance_id'] / 'evaluation/eval_result.json'
        report.parent.mkdir(parents=True)
        report.write_text(json.dumps(row))
    summary = run / 'summary.json'
    summary.write_text(json.dumps({'final': {'total': 148, 'resolved': 0, 'results': rows}}))
    job = {'method': 'recreate', 'run_dir': str(run), 'summary': str(summary)}
    assert scheduled.valid_summary(job) is False
    report = run / 'final' / ids[-1] / 'evaluation/eval_result.json'
    report.parent.mkdir(parents=True)
    report.write_text(json.dumps(rows[-1]))
    assert scheduled.valid_summary(job) is True
