import json

from attribution_experiments.recover_remaining_finals import valid_paths


def test_scope_requires_matching_report_and_excludes_ssh_failures(tmp_path):
    for iid, score, error in [('valid', False, None), ('infra', None, 'RemoteInfrastructureError')]:
        d = tmp_path / 'final' / iid
        (d / 'evaluation').mkdir(parents=True)
        row = {'instance_id': iid, 'evaluation': {'resolved': score}}
        if error:
            row['exit_status'] = error
        (d / 'result.json').write_text(json.dumps(row))
        (d / 'evaluation/report.json').write_text(json.dumps({'resolved': score}))
    assert list(valid_paths(tmp_path, 'scope', ['valid', 'infra', 'missing'])) == ['valid']


def test_recreate_keeps_valid_unresolved_and_excludes_null_or_wrong_identity(tmp_path):
    for iid, row in [('ok', {'instance_id': 'ok', 'resolved': False}),
                     ('bad', {'instance_id': 'bad', 'resolved': None}),
                     ('wrong', {'instance_id': 'other', 'resolved': True})]:
        d = tmp_path / 'final' / iid / 'evaluation'
        d.mkdir(parents=True)
        (d / 'eval_result.json').write_text(json.dumps(row))
    assert list(valid_paths(tmp_path, 'recreate', ['ok', 'bad', 'wrong'])) == ['ok']
