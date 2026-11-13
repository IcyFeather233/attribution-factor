import json

import pytest

from attribution_experiments.harnessfix.runtime_protocol import PROTOCOL, require_compatible_runtime


def test_fresh_run_is_allowed(tmp_path):
    require_compatible_runtime(tmp_path / "new")


def test_legacy_partial_or_complete_run_is_rejected(tmp_path):
    (tmp_path / "iteration_01").mkdir()
    with pytest.raises(RuntimeError, match="cannot be reused"):
        require_compatible_runtime(tmp_path)
    (tmp_path / "summary.json").write_text("{}")
    with pytest.raises(RuntimeError):
        require_compatible_runtime(tmp_path)


def test_matching_protocol_allows_resume(tmp_path):
    (tmp_path / "iteration_01").mkdir()
    (tmp_path / "runtime_protocol.json").write_text(json.dumps({"environment_transport": PROTOCOL}))
    require_compatible_runtime(tmp_path)
