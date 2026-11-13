"""Load the preserved HarnessFix bounded-validation implementation.

The implementation bytecode is pinned to the experiment Python 3.11 runtime.
Task rollouts inherit the shared 250-step protocol through the environment and
candidate worker.
"""

from __future__ import annotations

import marshal
import os
from pathlib import Path

from qwen_gpt55_experiments.common.model_config import TASK_STEP_LIMIT

_IMPL = Path(__file__).with_name(
    "run_native_impl_bounded_validation.cpython-311.pyc"
)
os.environ.setdefault("HARNESSFIX_STEP_LIMIT", str(TASK_STEP_LIMIT))

with _IMPL.open("rb") as _handle:
    _handle.read(16)
    _code = marshal.load(_handle)

_loader_entrypoint_name = __name__
try:
    __name__ = "qwen_gpt55_experiments.harnessfix._preserved_implementation"
    exec(_code, globals(), globals())
finally:
    __name__ = _loader_entrypoint_name

from qwen_gpt55_experiments.harnessfix.runtime_protocol import PROTOCOL, require_compatible_runtime

_preserved_prepare_run = prepare_run


def prepare_run(run_dir, *args, **kwargs):
    require_compatible_runtime(Path(run_dir))
    from qwen_gpt55_experiments.common.execution_protocol import record_execution_backend
    execution_signature = record_execution_backend(run_dir, check_only=True)
    result = _preserved_prepare_run(run_dir, *args, **kwargs)
    write_json(Path(run_dir) / "runtime_protocol.json", {"environment_transport": PROTOCOL})
    write_json(Path(run_dir) / 'execution_backend.json', execution_signature)
    return result


if __name__ == "__main__":
    main()
