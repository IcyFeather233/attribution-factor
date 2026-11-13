from __future__ import annotations

import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from attribution_experiments.common.udocker_runtime import (
    container_scratch_volumes,
    persistent_environment,
    runtime_root,
    cleanup_private_containers_dir,
    create_slot,
    make_private_containers_dir,
    udocker_host_environment,
)


def test_private_container_directory_and_environment(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("UDOCKER_RUNTIME_ROOT", str(tmp_path))
    containers_dir = make_private_containers_dir("agent-test")

    assert containers_dir.parent == tmp_path
    assert containers_dir.name.startswith("agent-test-")
    assert udocker_host_environment(containers_dir)["UDOCKER_CONTAINERS"] == str(
        containers_dir
    )

    cleanup_private_containers_dir(containers_dir)
    assert not containers_dir.exists()


def test_create_slot_bounds_concurrency(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("UDOCKER_RUNTIME_ROOT", str(tmp_path))
    monkeypatch.setenv("UDOCKER_CREATE_CONCURRENCY", "2")
    active = 0
    peak = 0
    lock = threading.Lock()

    def worker() -> None:
        nonlocal active, peak
        with create_slot():
            with lock:
                active += 1
                peak = max(peak, active)
            time.sleep(0.05)
            with lock:
                active -= 1

    with ThreadPoolExecutor(max_workers=6) as executor:
        list(executor.map(lambda _: worker(), range(6)))

    assert peak == 2


def test_cleanup_refuses_directory_outside_runtime_root(
    tmp_path: Path, monkeypatch
) -> None:
    runtime = tmp_path / "runtime"
    outside = tmp_path / "outside"
    runtime.mkdir()
    outside.mkdir()
    monkeypatch.setenv("UDOCKER_RUNTIME_ROOT", str(runtime))

    try:
        cleanup_private_containers_dir(outside)
    except ValueError:
        pass
    else:
        raise AssertionError("cleanup accepted a directory outside the runtime root")


def test_resident_limit_covers_container_lifetime(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("UDOCKER_RUNTIME_ROOT", str(tmp_path))
    monkeypatch.setenv("UDOCKER_MAX_CONTAINERS", "1")
    first = make_private_containers_dir("first")
    with ThreadPoolExecutor(max_workers=1) as executor:
        pending = executor.submit(make_private_containers_dir, "second")
        try:
            time.sleep(0.1)
            assert not pending.done()
        finally:
            cleanup_private_containers_dir(first)
        second = pending.result(timeout=3)
    cleanup_private_containers_dir(second)


def test_failed_directory_creation_releases_slot(tmp_path: Path, monkeypatch) -> None:
    import tempfile
    import pytest

    monkeypatch.setenv("UDOCKER_RUNTIME_ROOT", str(tmp_path))
    monkeypatch.setenv("UDOCKER_MAX_CONTAINERS", "1")
    original = tempfile.mkdtemp

    def fail(**kwargs):
        raise OSError("disk full")

    monkeypatch.setattr(tempfile, "mkdtemp", fail)
    with pytest.raises(OSError, match="disk full"):
        make_private_containers_dir("failed")
    monkeypatch.setattr(tempfile, "mkdtemp", original)
    with ThreadPoolExecutor(max_workers=1) as executor:
        path = executor.submit(make_private_containers_dir, "recovered").result(timeout=3)
    cleanup_private_containers_dir(path)


def test_evaluator_can_start_with_full_rollout_pool(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("UDOCKER_RUNTIME_ROOT", str(tmp_path))
    monkeypatch.setenv("UDOCKER_MAX_CONTAINERS", "1")
    monkeypatch.setenv("UDOCKER_MAX_EVALUATORS", "1")
    agent = make_private_containers_dir("recreate-agent-test")
    try:
        with ThreadPoolExecutor(max_workers=1) as executor:
            evaluator = executor.submit(
                make_private_containers_dir, "recreate-eval-test"
            ).result(timeout=3)
        cleanup_private_containers_dir(evaluator)
    finally:
        cleanup_private_containers_dir(agent)


def test_rejects_ephemeral_runtime(monkeypatch) -> None:
    import pytest
    for root in ("/tmp/unsafe-runtime", "/dev/shm/unsafe-runtime"):
        monkeypatch.setenv("UDOCKER_RUNTIME_ROOT", root)
        with pytest.raises(ValueError, match="/volume/med"):
            runtime_root()


def test_cached_environment_does_not_leak_mutations_or_roots(tmp_path, monkeypatch):
    monkeypatch.setenv("UDOCKER_RUNTIME_ROOT", str(tmp_path / "first" / "containers"))
    first = persistent_environment()
    scratch = first["TMPDIR"]
    first["TMPDIR"] = "/tmp/invalid"
    assert persistent_environment()["TMPDIR"] == scratch
    monkeypatch.setenv("UDOCKER_RUNTIME_ROOT", str(tmp_path / "second" / "containers"))
    second = persistent_environment()
    assert second["TMPDIR"] != scratch
    assert Path(second["TMPDIR"]).is_dir()


def test_host_and_container_scratch_are_persistent(tmp_path: Path, monkeypatch) -> None:
    root = tmp_path / "containers"
    monkeypatch.setenv("UDOCKER_RUNTIME_ROOT", str(root))
    monkeypatch.setenv("TMPDIR", "/tmp")
    monkeypatch.setenv("PROOT_TMP_DIR", "/dev/shm")
    env = persistent_environment()
    assert all(Path(value).is_relative_to("/volume/med") for value in env.values())
    container = make_private_containers_dir("agent-storage-test")
    try:
        host_env = udocker_host_environment(container)
        assert host_env["TMPDIR"] == env["TMPDIR"]
        assert host_env["PROOT_TMP_DIR"] == env["PROOT_TMP_DIR"]
        volumes = container_scratch_volumes(container)
        assert set(volumes.values()) == {"/tmp", "/var/tmp", "/dev/shm"}
        assert all(Path(source).is_dir() and Path(source).is_relative_to(container)
                   for source in volumes)
    finally:
        cleanup_private_containers_dir(container)
