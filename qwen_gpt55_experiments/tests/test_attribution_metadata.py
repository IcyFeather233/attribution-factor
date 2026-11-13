from qwen_failure_comparison.tools.validate_trace_analyses import validate_task_metadata


def test_rejects_placeholder_and_invented_test_name():
    official = {"problem_statement": "Actual issue", "fail_to_pass": ["test_lag_decimalfield (WindowFunctionTests)"]}
    errors = validate_task_metadata({"problem_statement": "SWE-bench Verified 任务", "fail_to_pass": ["test_lag_with_decimal_field"]}, official)
    assert len(errors) == 2


def test_rejects_partial_test_list():
    official = {"problem_statement": "Issue", "fail_to_pass": ["first", "second"]}
    assert validate_task_metadata({"problem_statement": "Issue", "fail_to_pass": ["first"]}, official)


def test_accepts_exact_official_record():
    official = {"problem_statement": "Issue\r\nexample", "fail_to_pass": ["test[a]", "test[b]"]}
    assert validate_task_metadata(dict(official), official) == []
