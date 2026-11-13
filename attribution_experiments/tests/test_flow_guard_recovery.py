import json
from pathlib import Path
import shlex

import pytest

from minisweagent.environments.local import LocalEnvironment
from minisweagent.exceptions import Submitted
from attribution_experiments.common.flow_evidence import normalize_reference_spelling, packet, records
from attribution_experiments.meta_harness.flow_views import make_views, environment_class
from attribution_experiments.meta_harness.recover_analysis_proposal import recover_proposal
from attribution_experiments.tests.test_native_flow_evidence import payload


def test_actual_shell_can_write_marker_in_json_before_submission_audit(tmp_path):
    manifest = make_views(tmp_path, {'task-1': payload(), 'task-2': payload('task-2')})
    env = environment_class(LocalEnvironment, manifest)(cwd=str(tmp_path))
    proposal = {'hypothesis': 'Handle COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT safely',
                'flow_evidence_refs': [r for r in manifest['references'] if '-edge-' in r]}
    command = "cat > proposal.json <<'JSON'\n" + json.dumps(proposal) + '\nJSON'
    assert env.execute({'command': command})['returncode'] == 0
    assert json.loads((tmp_path / 'proposal.json').read_text()) == proposal
    # A real submission without read evidence is still rejected.
    submit = {'command': 'echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT'}
    assert env.execute(submit)['returncode'] == 2
    for page in manifest['pages']:
        assert env.execute({'command': 'cat ' + shlex.quote(page)})['returncode'] == 0
    with pytest.raises(Submitted):
        env.execute(submit)


def test_only_known_separator_spelling_is_corrected():
    evidence = packet(payload())
    original = 'task-1/failure_causing_wrong_step/control-flow-edge-0'
    fixed, corrections = normalize_reference_spelling(evidence, [original])
    assert fixed == ['task-1/failure_causing_wrong_step/control_flow-edge-0']
    assert corrections[0]['original'] == original
    assert fixed[0] in records(evidence)
    for ref in ['task-1/failure_causing_wrong_step/control-flow-edge-99',
                'other/failure_causing_wrong_step/control-flow-edge-0', 'invented']:
        with pytest.raises(ValueError):
            normalize_reference_spelling(evidence, [ref])


def test_recovery_uses_first_valid_attempt_and_preserves_original_trace(tmp_path):
    manifest = make_views(tmp_path, {'task-1': payload(), 'task-2': payload('task-2')})
    messages = [{'role': 'tool', 'content': page['content']} for page in manifest['pages'].values()]
    refs = [r for r in manifest['references'] if '-edge-' in r]
    for name in ['original', 'later']:
        proposal = dict(name=name, hypothesis='mechanism', changes=['check'], evidence_summary='evidence',
                        expected_efficiency='neutral', flow_evidence_refs=refs)
        command = "cat > proposal.json <<'JSON'\n" + json.dumps(proposal) + '\nJSON\ntouch DONT_EXECUTE'
        messages.extend([{'role': 'assistant', 'extra': {'actions': [{'command': command}]}},
                         {'role': 'tool', 'content': 'Flow evidence not yet verified'}])
    (tmp_path / 'logs').mkdir()
    trace = tmp_path / 'logs/analysis.traj.json'
    trace.write_text(json.dumps({'messages': messages, 'info': {'exit_status': 'LimitsExceeded'}}))
    before = trace.read_bytes()
    recovered = recover_proposal(tmp_path)
    assert recovered['exit_status'] == 'RecoveredBlockedProposal'
    assert json.loads((tmp_path / 'proposal.json').read_text())['name'] == 'original'
    assert trace.read_bytes() == before
    assert not (tmp_path / 'DONT_EXECUTE').exists()
