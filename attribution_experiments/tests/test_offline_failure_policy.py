import json
import pytest
from attribution_experiments.meta_harness.resume_native import benchmark_remaining


def benchmark(path, tasks):
    return benchmark_remaining(instances=[{'instance_id': task} for task in tasks],
        scaffold={}, agent_class=None, assets_dir=None, output_dir=path, api_key='test',
        workers=1, offline_only=True)[0]


def test_offline_never_runs_missing_task(tmp_path):
    with pytest.raises(RuntimeError, match='Offline finalization refuses'):
        benchmark(tmp_path, ['missing'])


def test_exception_counts_as_failure_retaining_reason(tmp_path):
    task = tmp_path / 'failed-task'
    task.mkdir()
    (task / 'skip.json').write_text(json.dumps(dict(instance_id=task.name, skipped=True,
        count_as_failure=True, exit_status='RemoteInfrastructureError')))
    result = benchmark(tmp_path, [task.name])
    assert result['results'][0]['resolved'] is False
    assert result['results'][0]['status'] == 'RemoteInfrastructureError'
    assert result['total'] == 1 and result['resolved'] == 0 and result['skipped'] == 0
    assert result['exceptions_counted_as_failure'] == 1
    assert not result['score_is_incomplete']
