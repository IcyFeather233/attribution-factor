"""Cross-process concurrency and TPM gate for task-model requests."""

from __future__ import annotations

import fcntl
import json
import os
import time
import uuid
from pathlib import Path
from typing import Callable, TypeVar


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_GATE_PATH = ROOT / "attribution_experiments" / "parallel_launch" / "request_gate.state"
T = TypeVar("T")


def _total_tokens(response, fallback: int) -> int:
    usage = getattr(response, "usage", None)
    if usage is None and isinstance(response, dict):
        usage = response.get("usage")
    total = getattr(usage, "total_tokens", None)
    if total is None and isinstance(usage, dict):
        total = usage.get("total_tokens")
    try:
        return max(1, int(total))
    except (TypeError, ValueError):
        return fallback


def _read_state(gate) -> dict:
    gate.seek(0)
    raw = gate.read().strip()
    try:
        state = json.loads(raw) if raw else {}
    except json.JSONDecodeError:
        state = {}
    return {
        "entries": state.get("entries", []),
        "leases": state.get("leases", []),
        "last_start": float(state.get("last_start", 0.0)),
    }


def _write_state(gate, state: dict) -> None:
    gate.seek(0)
    gate.truncate()
    json.dump(state, gate)
    gate.flush()
    # flock + flush publishes the ledger to the other live processes. This is
    # transient admission state, not a durable experiment result; forcing every
    # update to stable GPFS storage unnecessarily serializes all requesters.


def _locked_state(gate) -> dict:
    fcntl.flock(gate.fileno(), fcntl.LOCK_EX)
    return _read_state(gate)


def _unlock(gate) -> None:
    fcntl.flock(gate.fileno(), fcntl.LOCK_UN)


def run_task_model_request(call: Callable[[], T]) -> T:
    """Run one task model request under shared in-flight and rolling-TPM limits."""
    gate_path = Path(os.environ.get("TASK_MODEL_REQUEST_GATE_PATH", DEFAULT_GATE_PATH))
    interval = float(os.environ.get("TASK_MODEL_MIN_REQUEST_INTERVAL_SECONDS", "0.5"))
    tpm_limit = int(os.environ.get("TASK_MODEL_TPM_LIMIT", "10000000"))
    utilization = float(os.environ.get("TASK_MODEL_TPM_UTILIZATION", "0.9"))
    reservation = int(os.environ.get("TASK_MODEL_TOKEN_RESERVATION", "35000"))
    max_in_flight = int(os.environ.get("TASK_MODEL_MAX_IN_FLIGHT", "120"))
    lease_seconds = float(os.environ.get("TASK_MODEL_SLOT_LEASE_SECONDS", "900"))
    token_budget = int(tpm_limit * utilization)
    if interval < 0:
        raise ValueError("TASK_MODEL_MIN_REQUEST_INTERVAL_SECONDS must be non-negative")
    if token_budget <= 0 or reservation <= 0 or reservation > token_budget:
        raise ValueError("Invalid task model TPM budget or reservation")
    if max_in_flight <= 0 or lease_seconds <= 0:
        raise ValueError("Invalid task model in-flight limit or lease duration")

    gate_path.parent.mkdir(parents=True, exist_ok=True)
    request_id = uuid.uuid4().hex

    with gate_path.open("a+") as gate:
        while True:
            state = _locked_state(gate)
            now = time.time()
            entries = [
                entry
                for entry in state["entries"]
                if now - float(entry.get("timestamp", 0.0)) < 60
            ]
            leases = [
                lease
                for lease in state["leases"]
                if now - float(lease.get("started_at", 0.0)) < lease_seconds
            ]
            interval_wait = interval - (now - state["last_start"])
            slot_wait = 1.0 if len(leases) >= max_in_flight else 0.0
            token_wait = 0.0
            if sum(int(entry.get("tokens", 0)) for entry in entries) + reservation > token_budget:
                oldest = min(float(entry["timestamp"]) for entry in entries)
                token_wait = 60 - (now - oldest)
            wait_for = max(interval_wait, slot_wait, token_wait, 0.0)
            if wait_for <= 0:
                started_at = time.time()
                entries.append(
                    {
                        "request_id": request_id,
                        "timestamp": started_at,
                        "tokens": reservation,
                    }
                )
                leases.append(
                    {
                        "request_id": request_id,
                        "started_at": started_at,
                        "pid": os.getpid(),
                    }
                )
                state = {
                    "entries": entries,
                    "leases": leases,
                    "last_start": started_at,
                }
                _write_state(gate, state)
                _unlock(gate)
                break
            # No reservation changed. Expiration is recomputed by the next
            # reader, so waiting contenders need not rewrite the shared file.
            _unlock(gate)
            time.sleep(min(wait_for, 5.0))

        actual_tokens = reservation
        try:
            response = call()
            actual_tokens = _total_tokens(response, reservation)
            return response
        finally:
            state = _locked_state(gate)
            now = time.time()
            entries = [
                entry
                for entry in state["entries"]
                if now - float(entry.get("timestamp", 0.0)) < 60
            ]
            for entry in entries:
                if entry.get("request_id") == request_id:
                    entry["tokens"] = actual_tokens
                    break
            state = {
                "entries": entries,
                "leases": [
                    lease
                    for lease in state["leases"]
                    if lease.get("request_id") != request_id
                    and now - float(lease.get("started_at", 0.0)) < lease_seconds
                ],
                "last_start": state["last_start"],
            }
            _write_state(gate, state)
            _unlock(gate)
