import json
import subprocess
import sys

from qwen_gpt55_experiments.common.remote_runtime import remote_status_command, decode_remote_status


def test_stderr_is_drained_into_same_stream_before_receipt_with_stdin():
    command = ['bash', '-s']
    script = 'set -x\nprintf "test stdout\\n"\nprintf "cleanup stderr\\n" >&2\nexit 255\n'
    result = subprocess.run(remote_status_command(command, 'mixed'), input=script,
                            text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    assert result.returncode == 0
    assert result.stderr == ''
    decoded = decode_remote_status(result, command, 'mixed')
    assert decoded.returncode == 255
    assert 'test stdout' in decoded.stdout and 'cleanup stderr' in decoded.stdout
    assert result.stdout.endswith('\n__FA_EXEC_STATUS_mixed__:255\n')


def test_output_larger_than_pipe_buffer_is_streamed_without_deadlock():
    command = [sys.executable, '-c', 'import sys;sys.stdout.write("a"*200000);sys.stderr.write("b"*200000)']
    result = subprocess.run(remote_status_command(command, 'large'), text=True,
                            capture_output=True, timeout=10)
    output = decode_remote_status(result, command, 'large')
    assert output.stdout == 'a' * 200000 + 'b' * 200000
    assert not result.stderr


def test_scope_does_not_reuse_null_or_inconsistent_evaluation(tmp_path):
    from qwen_gpt55_experiments.scope.run_native import completed_final_results
    directory = tmp_path / 'task'
    (directory / 'evaluation').mkdir(parents=True)
    instances = [{'instance_id': 'task'}]
    for result_score, report_score in [(None, None), (False, True), (True, False)]:
        (directory / 'result.json').write_text(json.dumps({'instance_id': 'task', 'api_calls': 1,
                                                         'evaluation': {'resolved': result_score}}))
        (directory / 'evaluation/report.json').write_text(json.dumps({'resolved': report_score}))
        assert completed_final_results(tmp_path, instances) == {}
    (directory / 'result.json').write_text(json.dumps({'instance_id': 'task', 'api_calls': 1,
                                                     'evaluation': {'resolved': False}}))
    (directory / 'evaluation/report.json').write_text('{"resolved": false}')
    assert list(completed_final_results(tmp_path, instances)) == ['task']
