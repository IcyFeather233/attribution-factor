from attribution_experiments.run_pipeline import child_environment
from attribution_experiments import run_pipeline


def test_explicit_gate_settings_survive_nested_launchers(monkeypatch):
    monkeypatch.setattr(run_pipeline, 'persistent_environment', lambda: {})
    settings = {'TASK_MODEL_MIN_REQUEST_INTERVAL_SECONDS': '0.1', 'TASK_MODEL_TPM_LIMIT': '20000000',
                'TASK_MODEL_MAX_IN_FLIGHT': '192', 'TASK_MODEL_TPM_UTILIZATION': '0.9'}
    for key, value in settings.items():
        monkeypatch.setenv(key, value)
    env = child_environment('test-optimizer', 'test-task')
    assert {key: env[key] for key in settings} == settings


def test_unspecified_gate_settings_keep_legacy_defaults(monkeypatch):
    monkeypatch.setattr(run_pipeline, 'persistent_environment', lambda: {})
    for key in ['TASK_MODEL_MIN_REQUEST_INTERVAL_SECONDS', 'TASK_MODEL_TPM_LIMIT', 'TASK_MODEL_MAX_IN_FLIGHT']:
        monkeypatch.delenv(key, raising=False)
    env = child_environment('test-optimizer', 'test-task')
    assert env['TASK_MODEL_MIN_REQUEST_INTERVAL_SECONDS'] == '0.5'
    assert env['TASK_MODEL_TPM_LIMIT'] == '10000000'
    assert env['TASK_MODEL_MAX_IN_FLIGHT'] == '192'
