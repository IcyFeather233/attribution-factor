import json
from attribution_experiments.meta_harness.input_audit import audit_proposer_inputs, frozen_analysis_view


def test_derived_view_requires_exact_frozen_inputs(tmp_path):
    search = tmp_path / 'search'
    (search / 'traces').mkdir(parents=True)
    (search / 'logs').mkdir()
    (tmp_path / 'provided_analysis').mkdir()
    (search / 'traces/task.traj.json').write_text('{}')
    evidence = 'The provided analysis mentions gold_patch in its explanation of the failed repair.'
    data = {'provided_information': {key: {'step': 1, 'evidence': evidence}
             for key in ('failure_causing_wrong_step', 'failure_inevitable_step')}}
    (tmp_path / 'provided_analysis/task.json').write_text(json.dumps(data))
    expected = frozen_analysis_view(search)
    (search / 'pa_full.txt').write_text(expected)
    messages = [{'role': 'assistant', 'extra': {'actions': [{'command': "sed -n '1,100p' pa_full.txt", 'tool_call_id': 'x'}]}},
                {'role': 'tool', 'tool_call_id': 'x', 'content': evidence}]
    (search / 'logs/analysis.traj.json').write_text(json.dumps({'messages': messages}))
    assert audit_proposer_inputs(search)['passed']
    (search / 'pa_full.txt').write_text(expected + 'extra material')
    assert not audit_proposer_inputs(search)['passed']
