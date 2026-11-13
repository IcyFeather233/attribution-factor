import json

from qwen_gpt55_experiments.harnessfix.recover_final_infrastructure import resource_limit_record, gradeable
from qwen_gpt55_experiments.common.result_validity import retryable_text


def test_context_limit_requires_matching_structured_worker_result():
    row = {'instance_id': 'case', 'exit_status': 'ContextWindowExceededError', 'api_calls': 250,
           'exception': 'context exceeded with nested {message}'}
    assert resource_limit_record('noise\n' + json.dumps(row, indent=2), 'case') == row
    assert resource_limit_record(json.dumps(row), 'another') is None
    assert resource_limit_record('ContextWindowExceededError in a test output', 'case') is None
    row['exit_status'] = 'RemoteInfrastructureError'
    assert resource_limit_record(json.dumps(row), 'case') is None


def test_none_evaluation_is_not_gradeable(tmp_path):
    (tmp_path / 'evaluation').mkdir()
    (tmp_path / 'result.json').write_text('{"evaluation": null}')
    (tmp_path / 'evaluation/report.json').write_text('{"resolved": false}')
    assert not gradeable(tmp_path)


def test_new_transport_errors_remain_retryable_not_task_failures():
    assert retryable_text('Remote execution completion unconfirmed (SSH exit=255); output tail: ')
    assert retryable_text('This container run was invalidated by an infrastructure failure')
    assert not retryable_text('ContextWindowExceededError: prompt too long')
