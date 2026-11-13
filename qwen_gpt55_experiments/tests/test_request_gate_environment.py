from qwen_gpt55_experiments.run_pipeline import child_environment
from qwen_gpt55_experiments import run_pipeline


def test_explicit_gate_settings_survive_nested_launchers(monkeypatch):
    monkeypatch.setattr(run_pipeline, 'persistent_environment', lambda: {})
    settings = {'QWEN_MIN_REQUEST_INTERVAL_SECONDS': '0.1', 'QWEN_TPM_LIMIT': '20000000',
                'QWEN_MAX_IN_FLIGHT': '192', 'QWEN_TPM_UTILIZATION': '0.9'}
    for key, value in settings.items():
        monkeypatch.setenv(key, value)
    env = child_environment('test-optimizer', 'test-task')
    assert {key: env[key] for key in settings} == settings


def test_unspecified_gate_settings_keep_legacy_defaults(monkeypatch):
    monkeypatch.setattr(run_pipeline, 'persistent_environment', lambda: {})
    for key in ['QWEN_MIN_REQUEST_INTERVAL_SECONDS', 'QWEN_TPM_LIMIT', 'QWEN_MAX_IN_FLIGHT']:
        monkeypatch.delenv(key, raising=False)
    env = child_environment('test-optimizer', 'test-task')
    assert env['QWEN_MIN_REQUEST_INTERVAL_SECONDS'] == '0.5'
    assert env['QWEN_TPM_LIMIT'] == '10000000'
    assert env['QWEN_MAX_IN_FLIGHT'] == '192'
