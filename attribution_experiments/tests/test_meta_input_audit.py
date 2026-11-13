import json
import pytest
from attribution_experiments.meta_harness.input_audit import audit_proposer_inputs

EVIDENCE = 'The supplied analysis mentions gold_patch while describing the original incorrect implementation.'


def setup_run(tmp_path, command='cat provided_analysis/task.json', output=EVIDENCE, phase='analysis'):
    search = tmp_path / 'search'
    (search / 'logs').mkdir(parents=True)
    (search / 'traces').mkdir()
    (search / 'traces/task.traj.json').write_text('{}')
    (tmp_path / 'provided_analysis').mkdir()
    (tmp_path / 'provided_analysis/task.json').write_text(json.dumps({'provided_information': {'evidence': EVIDENCE}}))
    messages = [
        {'role': 'assistant', 'extra': {'actions': [{'command': command, 'tool_call_id': 'call-1'}]}},
        {'role': 'tool', 'tool_call_id': 'call-1', 'content': output, 'extra': {'raw_output': output}},
    ]
    (search / f'logs/{phase}.traj.json').write_text(json.dumps({'messages': messages}))
    return search


def test_exact_frozen_quotation_is_allowed_and_recorded(tmp_path):
    result = audit_proposer_inputs(setup_run(tmp_path))
    assert result['passed']
    assert len(result['authorized_quotation_hits']) == 1


@pytest.mark.parametrize('command,output,phase', [
    ('cat gold_patch', EVIDENCE, 'analysis'),
    ('cat provided_analysis/task.json', EVIDENCE + '\nother gold_patch material', 'analysis'),
    ('cat provided_analysis/task.json', 'different gold_patch evidence', 'analysis'),
    ('cat unrelated_file', EVIDENCE, 'analysis'),
    ('cat provided_analysis/task.json', EVIDENCE, 'implementation'),
])
def test_forbidden_access_or_unverified_output_still_rejected(tmp_path, command, output, phase):
    assert not audit_proposer_inputs(setup_run(tmp_path, command, output, phase))['passed']


def test_workspace_copy_cannot_grant_itself_an_exception(tmp_path):
    search = setup_run(tmp_path)
    (search.parent / 'provided_analysis/task.json').unlink()
    (search / 'provided_analysis').mkdir()
    (search / 'provided_analysis/task.json').write_text(json.dumps({'evidence': EVIDENCE}))
    assert not audit_proposer_inputs(search)['passed']
