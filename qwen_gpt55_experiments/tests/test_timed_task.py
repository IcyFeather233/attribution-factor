import os
import sys
import time

from qwen_gpt55_experiments.meta_harness.timed_task import run_bounded


def test_timeout_terminates_worker(tmp_path):
    started = time.monotonic()
    assert run_bounded([sys.executable, '-c', 'import time; time.sleep(60)'],
                       env=os.environ.copy(), log=tmp_path / 'worker.log', timeout=0.1) == 124
    assert time.monotonic() - started < 15


def test_completed_worker_preserves_exit_status(tmp_path):
    assert run_bounded([sys.executable, '-c', 'raise SystemExit(3)'],
                       env=os.environ.copy(), log=tmp_path / 'worker.log', timeout=5) == 3


def test_explicit_skip_is_not_graded_as_unresolved(tmp_path):
    import json
    from qwen_gpt55_experiments.meta_harness.resume_native import benchmark_remaining
    task = tmp_path / 'sympy__sympy-13031'
    task.mkdir()
    (task / 'skip.json').write_text(json.dumps({'instance_id': task.name, 'skipped': True,
                                              'exit_status': 'RemoteInfrastructureError'}))
    result, reused = benchmark_remaining(instances=[{'instance_id': task.name}], scaffold={},
        agent_class=None, assets_dir=None, output_dir=tmp_path, api_key='test', workers=1)
    assert result['results'][0]['resolved'] is None
    assert result['results'][0]['status'] == 'RemoteInfrastructureError'
    assert result['skipped'] == 1 and result['score_is_incomplete']
    assert result['total'] == 1 and result['resolved'] == 0
