"""Regression tests for HarnessFix plan-to-diff audit requirements."""

from baselines.harnessfix.failure_analysis.plan_diff_audit import audit_candidate


def _spec(fixes):
    return {
        "edit_budget": {
            "max_files_to_modify": 2,
            "allowed_paths": ["src/..."],
            "forbidden_paths": [],
        },
        "fixes": fixes,
    }


def _candidate_dirs(tmp_path):
    original = tmp_path / "original"
    candidate = tmp_path / "candidate"
    (original / "src").mkdir(parents=True)
    (candidate / "src").mkdir(parents=True)
    (original / "src" / "implemented.py").write_text("VALUE = 1\n")
    (candidate / "src" / "implemented.py").write_text("VALUE = 2\n")
    return original, candidate


def test_legacy_p3_follow_up_does_not_block_promotion(tmp_path):
    original, candidate = _candidate_dirs(tmp_path)
    result = audit_candidate(
        original,
        candidate,
        _spec([
            {"id": "required", "priority": "P1", "target_files": ["src/implemented.py"]},
            {"id": "follow_up", "priority": "P3", "target_files": ["src/optional.py"]},
        ]),
    )
    assert result["passed"] is True
    assert result["fix_coverage"]["follow_up"]["required"] is False
    assert result["fix_coverage"]["follow_up"]["covered"] is False


def test_uncovered_required_fix_still_blocks_promotion(tmp_path):
    original, candidate = _candidate_dirs(tmp_path)
    result = audit_candidate(
        original,
        candidate,
        _spec([{"id": "required", "priority": "P1", "target_files": ["src/missing.py"]}]),
    )
    assert result["passed"] is False
    assert "uncovered_fix_targets" in result["violations"]


def test_explicit_optional_fix_is_non_blocking_regardless_of_priority(tmp_path):
    original, candidate = _candidate_dirs(tmp_path)
    result = audit_candidate(
        original,
        candidate,
        _spec([
            {
                "id": "optional",
                "priority": "P1",
                "required": False,
                "target_files": ["src/missing.py"],
            }
        ]),
    )
    assert result["passed"] is True
    assert result["fix_coverage"]["optional"]["required"] is False
