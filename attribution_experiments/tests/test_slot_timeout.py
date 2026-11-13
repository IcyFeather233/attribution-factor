"""Verify resident slot acquisition fails fast instead of waiting forever."""

import fcntl

import pytest

from attribution_experiments.common import udocker_runtime


def test_slot_timeout_raises_when_all_held(tmp_path, monkeypatch):
    monkeypatch.setattr(udocker_runtime, "runtime_root", lambda: tmp_path)
    from attribution_experiments.common import remote_capacity
    monkeypatch.setattr(remote_capacity, "agent_capacity", lambda default: 1)
    slots_dir = tmp_path / ".resident_slots"
    slots_dir.mkdir()
    holder = (slots_dir / "slot-000.lock").open("a+")
    fcntl.flock(holder.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    try:
        with pytest.raises(RuntimeError, match="Timed out waiting for a resident slot"):
            udocker_runtime._acquire_slot("resident", 1, timeout=0.1)
    finally:
        fcntl.flock(holder.fileno(), fcntl.LOCK_UN)
        holder.close()


def test_slot_acquires_when_free(tmp_path, monkeypatch):
    monkeypatch.setattr(udocker_runtime, "runtime_root", lambda: tmp_path)
    from attribution_experiments.common import remote_capacity
    monkeypatch.setattr(remote_capacity, "agent_capacity", lambda default: 1)
    handle = udocker_runtime._acquire_slot("resident", 1, timeout=1)
    assert handle is not None
    handle.close()
