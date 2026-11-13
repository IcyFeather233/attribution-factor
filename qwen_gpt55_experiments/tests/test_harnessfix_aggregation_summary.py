"""Verify HarnessFix attribution becomes a component-level aggregation prior."""

import json

from qwen_gpt55_experiments.harnessfix.upstream_bridge import attribution_summary


def _payload(role, causing=True):
    key = "failure_causing_wrong_step" if causing else "failure_inevitable_step"
    return {"provided_information": {key: {"step": 1, "role": role}}}


def test_empty_results_returns_empty_summary(tmp_path):
    assert attribution_summary(tmp_path, []) == ""


def test_role_distribution_is_aggregated(tmp_path):
    for iid, role, causing in [
        ("django__django-1", "implementation", True),
        ("django__django-2", "implementation", True),
        ("django__django-3", "final_patch_review", False),
    ]:
        (tmp_path / f"{iid}.json").write_text(json.dumps(_payload(role, causing)))
    results = [
        {"instance_id": "django__django-1"},
        {"instance_id": "django__django-2"},
        {"instance_id": "django__django-3"},
    ]
    summary = attribution_summary(tmp_path, results)
    assert "causing step roles" in summary
    assert "implementation=2" in summary
    assert "inevitable step roles" in summary
    assert "final_patch_review=1" in summary
    assert "file_editing" in summary


def test_missing_payloads_are_ignored(tmp_path):
    results = [{"instance_id": "missing"}]
    assert attribution_summary(tmp_path, results) == ""


def test_flow_context_is_aggregated(tmp_path):
    payload = _payload("implementation", True)
    payload["provided_information"]["failure_causing_wrong_step_context"] = {
        "center_step": 1,
        "window_steps": [1],
        "control_flow": {
            "nodes": [
                {
                    "node_id": "control-step-1",
                    "is_center": True,
                    "phase": "failure_triage",
                    "harness_mechanism": "The harness appends the failed test output.",
                    "relation_to_failure": "The model accepts the failed test.",
                }
            ],
            "edges": [{"source": "control-step-1", "target": "control-step-1", "type": "test_to_triage"}],
        },
        "data_flow": {
            "nodes": [
                {
                    "node_id": "data-step-1",
                    "is_center": True,
                    "information_handling": "The failed assertion is misinterpreted.",
                }
            ],
            "edges": [{"source": "data-step-1", "target": "data-step-1", "type": "misinterpreted"}],
        },
    }
    (tmp_path / "django__django-1.json").write_text(json.dumps(payload))
    summary = attribution_summary(tmp_path, [{"instance_id": "django__django-1", "flow_evidence_refs": [
        "django__django-1/failure_causing_wrong_step/control_flow-edge-0",
        "django__django-1/failure_causing_wrong_step/data_flow-edge-0",
    ]}])
    assert "failure_triage" in summary
    assert "test_to_triage" in summary
    assert "misinterpreted" in summary
    assert "The harness appends the failed test output." in summary
    assert "The failed assertion is misinterpreted." in summary
