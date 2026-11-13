"""Verify task-model requests carry a hard timeout and bounded retries."""

import os

from attribution_experiments.common.model_config import task_model_kwargs


def test_task_model_kwargs_has_timeout():
    kwargs = task_model_kwargs(api_key="test")
    assert kwargs["timeout"] == 300


def test_timeout_is_overridable(monkeypatch):
    monkeypatch.setenv("TASK_MODEL_REQUEST_TIMEOUT_SECONDS", "120")
    kwargs = task_model_kwargs(api_key="test")
    assert kwargs["timeout"] == 120


def test_retry_attempts_default_is_bounded():
    assert int(os.environ.get("MSWEA_MODEL_RETRY_STOP_AFTER_ATTEMPT", "10")) <= 10
