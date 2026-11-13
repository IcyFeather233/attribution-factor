"""Host-side runtime isolation and throttling for udocker containers."""

from __future__ import annotations

import fcntl
import os
import shutil
import tempfile
import time
from contextlib import contextmanager
from functools import lru_cache
from pathlib import Path
from typing import Iterator


DEFAULT_RUNTIME_ROOT = Path(__file__).resolve().parents[1] / "runtime" / "containers"
_resident_slots = {}


def runtime_root() -> Path:
    root = Path(os.environ.get("UDOCKER_RUNTIME_ROOT", DEFAULT_RUNTIME_ROOT)).resolve()
    if not root.is_relative_to("/volume/med"):
        raise ValueError(f"Runtime storage must be under /volume/med: {root}")
    return root


def persistent_environment() -> dict[str, str]:
    root = runtime_root()
    # These directories live for the run, not for an individual container.
    # Return a fresh mapping so callers cannot mutate the cached configuration.
    return dict(_persistent_environment_items(root))


@lru_cache(maxsize=32)
def _persistent_environment_items(root: Path) -> tuple[tuple[str, str], ...]:
    scratch = root.parent / "host_tmp"
    cache = root.parent / "cache"
    values = {
        "UDOCKER_RUNTIME_ROOT": str(root),
        **{key: str(scratch) for key in ("TMPDIR", "TMP", "TEMP", "UDOCKER_TMP", "PROOT_TMP_DIR")},
        "XDG_CACHE_HOME": str(cache),
        "XDG_CONFIG_HOME": str(root.parent / "config"),
        "XDG_DATA_HOME": str(root.parent / "data"),
        "XDG_STATE_HOME": str(root.parent / "state"),
        "MSWEA_GLOBAL_CONFIG_DIR": str(root.parent / "config" / "mini-swe-agent"),
        "HF_HOME": str(cache / "huggingface"),
        "HF_HUB_CACHE": str(cache / "huggingface" / "hub"),
        "HF_DATASETS_CACHE": str(cache / "huggingface" / "datasets"),
        "TORCH_HOME": str(cache / "torch"),
        "MPLCONFIGDIR": str(cache / "matplotlib"),
        "NUMBA_CACHE_DIR": str(cache / "numba"),
        "PIP_CACHE_DIR": str(cache / "pip"),
        "UV_CACHE_DIR": str(cache / "uv"),
        "JOBLIB_TEMP_FOLDER": str(scratch / "joblib"),
    }
    for value in set(values.values()):
        Path(value).mkdir(parents=True, exist_ok=True)
    return tuple(values.items())


def container_scratch_volumes(containers_dir: Path) -> dict[str, str]:
    if not containers_dir.resolve().is_relative_to(runtime_root()):
        raise ValueError("Container scratch must be inside the persistent runtime root")
    volumes = {}
    for name, target in (("tmp", "/tmp"), ("var_tmp", "/var/tmp"), ("shm", "/dev/shm")):
        source = containers_dir / ".scratch" / name
        source.mkdir(parents=True, exist_ok=True)
        volumes[str(source.resolve())] = target
    return volumes


def make_private_containers_dir(container_name: str) -> Path:
    root = runtime_root()
    root.mkdir(parents=True, exist_ok=True)
    safe_name = "".join(
        ch if ch.isalnum() or ch in "-_" else "_" for ch in container_name
    )
    # ReCreate grades while retaining its agent container; reserve evaluator slots
    # so a full rollout pool cannot deadlock waiting for fresh grading containers.
    evaluator = container_name.startswith(("eval-", "recreate-eval-"))
    kind = "evaluators" if evaluator else "resident"
    setting = "UDOCKER_MAX_EVALUATORS" if evaluator else "UDOCKER_MAX_CONTAINERS"
    slot = _acquire_slot(kind, int(os.environ.get(setting, "4" if evaluator else "16")))
    try:
        path = Path(tempfile.mkdtemp(prefix=f"{safe_name}-", dir=root))
    except BaseException:
        slot.close()
        raise
    _resident_slots[path.resolve()] = slot
    return path


def cleanup_private_containers_dir(path: Path) -> None:
    root = runtime_root().resolve()
    target = path.resolve()
    if target.parent != root:
        raise ValueError(f"Refusing to clean udocker directory outside {root}: {target}")
    shutil.rmtree(target, ignore_errors=True)
    slot = _resident_slots.pop(target, None)
    if slot is not None:
        slot.close()


def udocker_host_environment(containers_dir: Path) -> dict[str, str]:
    env = os.environ.copy()
    env.update(persistent_environment())
    env["UDOCKER_CONTAINERS"] = str(containers_dir)
    return env


def _acquire_slot(kind: str, count: int, timeout: float | None = None):
    if count < 1:
        raise ValueError(f"udocker {kind} slot count must be at least 1")
    if timeout is None:
        timeout = float(os.environ.get("UDOCKER_SLOT_WAIT_SECONDS", "7200"))

    slots_dir = runtime_root() / f".{kind}_slots"
    slots_dir.mkdir(parents=True, exist_ok=True)
    handle = None
    deadline = time.monotonic() + timeout
    while handle is None:
        from .remote_capacity import agent_capacity
        current_count = agent_capacity(count) if kind == 'resident' else count
        for index in range(current_count):
            candidate = (slots_dir / f"slot-{index:03d}.lock").open("a+")
            try:
                fcntl.flock(candidate.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                candidate.close()
                continue
            handle = candidate
            break
        if handle is None:
            if time.monotonic() >= deadline:
                raise RuntimeError(f"Timed out waiting for a {kind} slot after {timeout:.0f} seconds")
            time.sleep(0.25)

    handle.seek(0)
    handle.truncate()
    handle.write(f"pid={os.getpid()} acquired_at={time.time()}\n")
    handle.flush()
    return handle


@contextmanager
def create_slot() -> Iterator[None]:
    """Bound extraction separately from the lifetime of resident containers."""
    with _acquire_slot("create", int(os.environ.get("UDOCKER_CREATE_CONCURRENCY", "2"))):
        yield
