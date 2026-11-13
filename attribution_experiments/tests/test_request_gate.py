import json
import multiprocessing
import time

import pytest

from attribution_experiments.common.request_gate import run_task_model_request


def _request_worker(events):
    def call():
        events.put((time.monotonic(), 1))
        time.sleep(0.12)
        events.put((time.monotonic(), -1))
        return {"usage": {"total_tokens": 7}}
    run_task_model_request(call)


def test_shared_gate_preserves_admission_limits(tmp_path, monkeypatch):
    path = tmp_path / "gate.json"
    monkeypatch.setenv("TASK_MODEL_REQUEST_GATE_PATH", str(path))
    monkeypatch.setenv("TASK_MODEL_MAX_IN_FLIGHT", "2")
    monkeypatch.setenv("TASK_MODEL_MIN_REQUEST_INTERVAL_SECONDS", "0.03")
    ctx = multiprocessing.get_context("fork")
    events = ctx.Queue()
    processes = [ctx.Process(target=_request_worker, args=(events,)) for _ in range(4)]
    for process in processes:
        process.start()
    try:
        records = [events.get(timeout=10) for _ in range(8)]
        for process in processes:
            process.join(timeout=10)
            assert process.exitcode == 0
        active = peak = 0
        for _, delta in sorted(records):
            active += delta
            peak = max(peak, active)
        assert 1 <= peak <= 2
        starts = sorted(t for t, delta in records if delta == 1)
        assert all(b - a >= 0.025 for a, b in zip(starts, starts[1:]))
        state = json.loads(path.read_text())
        assert state["leases"] == []
        assert len(state["entries"]) == 4
        assert sum(e["tokens"] for e in state["entries"]) == 28
    finally:
        for process in processes:
            if process.is_alive():
                process.terminate()
            process.join(timeout=5)
        events.close()


def test_failed_request_releases_lease(tmp_path, monkeypatch):
    path = tmp_path / "gate.json"
    monkeypatch.setenv("TASK_MODEL_REQUEST_GATE_PATH", str(path))
    monkeypatch.setenv("TASK_MODEL_MIN_REQUEST_INTERVAL_SECONDS", "0")
    def fail():
        raise RuntimeError("provider failed")
    with pytest.raises(RuntimeError, match="provider failed"):
        run_task_model_request(fail)
    assert json.loads(path.read_text())["leases"] == []
