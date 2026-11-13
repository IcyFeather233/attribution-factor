from attribution_experiments.common.result_validity import retryable_result, retryable_text
from attribution_experiments.common.result_validity import recreate_final_problems
import json


def test_404_is_retryable():
    assert retryable_result({"exit_status": "NotFoundError", "exception": "404 Not Found"})


def test_remote_disk_is_retryable():
    assert retryable_text("Remote disk below admission threshold; no new container admitted")


def test_remote_none_resolution_is_retryable():
    assert retryable_result({"evaluation": {"resolved": None, "error": "remote_infrastructure_error"}})


def test_normal_unresolved_is_gradeable():
    assert not retryable_result({"exit_status": "Submitted", "evaluation": {"resolved": False}})


def test_task_timeout_is_not_automatically_retried():
    assert not retryable_result({"exit_status": "TaskTimeout", "exception": "TaskTimeout"})


def test_recreate_exception_without_report_is_not_a_completed_negative_grade(tmp_path):
    row = {'instance_id': 'task', 'resolved': False, 'exit_status': 'UnicodeDecodeError'}
    assert recreate_final_problems(tmp_path, [row]) == {'task': 'missing_or_unreadable_evaluation'}
    report = tmp_path / 'task/evaluation/eval_result.json'
    report.parent.mkdir(parents=True)
    report.write_text(json.dumps({'instance_id': 'task', 'resolved': False}))
    assert recreate_final_problems(tmp_path, [{'instance_id': 'task', 'resolved': False,
                                              'exit_status': 'TimeExceeded'}]) == {}
    assert recreate_final_problems(tmp_path, [{'instance_id': 'task', 'resolved': True}]) == {
        'task': 'evaluation_summary_mismatch'}
