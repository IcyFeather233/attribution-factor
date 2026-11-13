import json
import pytest
from attribution_experiments.common.execution_protocol import record_execution_backend
from attribution_experiments.gepa.metric_resume import MetricReplay


def test_remote_resume_rejects_historical_and_changed_backend(tmp_path, monkeypatch):
    monkeypatch.setenv('EXPERIMENT_CONTAINER_BACKEND', 'remote_docker')
    historical = tmp_path / 'old'
    historical.mkdir()
    (historical / 'summary.json').write_text('{}')
    with pytest.raises(RuntimeError, match='historical'):
        record_execution_backend(historical)
    fresh = tmp_path / 'fresh'
    record_execution_backend(fresh)
    record_execution_backend(fresh)
    monkeypatch.setenv('EXPERIMENT_CONTAINER_BACKEND', 'udocker')
    with pytest.raises(RuntimeError, match='changed'):
        record_execution_backend(fresh)


def test_gepa_replay_cannot_cross_backends(tmp_path, monkeypatch):
    monkeypatch.setenv('EXPERIMENT_CONTAINER_BACKEND', 'remote_docker')
    with pytest.raises(ValueError, match='backends'):
        MetricReplay(tmp_path / 'old', tmp_path / 'new', model='qwen', step_limit=250)


def test_registered_budget_is_not_mistaken_for_a_historical_run(tmp_path, monkeypatch):
    monkeypatch.setenv('EXPERIMENT_CONTAINER_BACKEND', 'remote_docker')
    (tmp_path / 'optimization_budget.json').write_text('{"protocol":"new"}')
    assert record_execution_backend(tmp_path, check_only=True)['backend'] == 'remote_docker'
    assert not (tmp_path / 'execution_backend.json').exists()
    record_execution_backend(tmp_path)
    assert (tmp_path / 'execution_backend.json').exists()
    old = tmp_path / 'old'; old.mkdir()
    (old / 'optimization_budget.json').write_text('{}')
    (old / 'summary.json').write_text('{}')
    with pytest.raises(RuntimeError, match='historical'):
        record_execution_backend(old)
