import pytest

from qwen_gpt55_experiments.common.runner import bounded_task_wall_time


def test_task_wall_time_defaults_to_one_hour(monkeypatch):
    monkeypatch.delenv("QWEN_TASK_WALL_TIME_SECONDS", raising=False)
    assert bounded_task_wall_time({"wall_time_limit_seconds": 0}) == 3600


def test_task_wall_time_preserves_stricter_limit_and_caps_longer_one(monkeypatch):
    monkeypatch.setenv("QWEN_TASK_WALL_TIME_SECONDS", "3600")
    assert bounded_task_wall_time({"wall_time_limit_seconds": 900}) == 900
    assert bounded_task_wall_time({"wall_time_limit_seconds": 7200}) == 3600


def test_task_wall_time_rejects_nonpositive_global_limit(monkeypatch):
    monkeypatch.setenv("QWEN_TASK_WALL_TIME_SECONDS", "0")
    with pytest.raises(ValueError, match="must be positive"):
        bounded_task_wall_time({})
