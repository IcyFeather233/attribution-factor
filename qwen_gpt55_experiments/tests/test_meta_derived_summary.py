"""Verify derived summary files from provided_analysis are authorized."""

import json

from qwen_gpt55_experiments.meta_harness.input_audit import (
    audit_proposer_inputs,
    derived_output_files,
    reads_derived_file,
)


def test_derived_output_files_are_detected():
    messages = [
        {
            "role": "assistant",
            "extra": {"actions": [
                {"command": "python - <<PY > /tmp/pa_summary.txt\nimport json,glob\nfor p in glob.glob('provided_analysis/*.json'): pass\nPY"}
            ]},
        }
    ]
    assert derived_output_files(messages) == {"/tmp/pa_summary.txt"}


def test_reads_derived_file_matches_path():
    derived = {"/tmp/pa_summary.txt"}
    assert reads_derived_file("sed -n 1,20p /tmp/pa_summary.txt", derived)
    assert not reads_derived_file("cat other.txt", derived)


def test_derived_summary_is_authorized(tmp_path):
    search = tmp_path / "search"
    (search / "traces").mkdir(parents=True)
    (search / "logs").mkdir()
    (tmp_path / "provided_analysis").mkdir()
    (search / "traces/task.traj.json").write_text("{}")
    evidence = "gold_patch modifies both files while the candidate only changes one."
    data = {"provided_information": {
        "failure_causing_wrong_step": {
            "step": 1, "role": "implementation",
            "what_went_wrong": "wrong file", "evidence": evidence,
            "message_index": 2,
        },
        "failure_inevitable_step": {
            "step": 3, "role": "final_patch_review",
            "what_went_wrong": "no verify", "evidence": "submit",
            "message_index": 4,
        },
    }}
    (tmp_path / "provided_analysis/task.json").write_text(json.dumps(data))
    messages = [
        {"role": "assistant", "extra": {"actions": [
            {"command": "python - <<PY > /tmp/pa_summary.txt\nimport json,glob\nfor p in glob.glob(provided_analysis/*.json):\n d=json.load(open(p))\n print(d[provided_information][failure_causing_wrong_step][evidence])\nPY", "tool_call_id": "gen"}
        ]}},
        {"role": "tool", "tool_call_id": "gen", "content": evidence},
        {"role": "assistant", "extra": {"actions": [
            {"command": "sed -n 1,20p /tmp/pa_summary.txt", "tool_call_id": "read"}
        ]}},
        {"role": "tool", "tool_call_id": "read", "content": evidence, "extra": {"raw_output": evidence}},
    ]
    (search / "logs/analysis.traj.json").write_text(json.dumps({"messages": messages}))
    result = audit_proposer_inputs(search)
    assert result["passed"]
