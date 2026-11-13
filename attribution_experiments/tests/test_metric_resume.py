import json

import pytest

from attribution_experiments.gepa.metric_resume import MetricReplay


def fixture_run(tmp_path):
    source, target = tmp_path / "source", tmp_path / "target"
    for root in (source, target):
        (root / "provided_analysis").mkdir(parents=True)
        (root / "provided_analysis" / "task.json").write_text("{}")
    case = source / "metric_calls" / "0000_task"
    (case / "evaluation").mkdir(parents=True)
    result = dict(instance_id="task", model="qwen", step_limit=250, temperature=1.0,
                  api_calls=10, exit_status="Submitted", evaluation={"resolved": False})
    (case / "result.json").write_text(json.dumps(result))
    (case / "evaluation" / "report.json").write_text(
        json.dumps({"instance_id": "task", "resolved": False}))
    (case / "trajectory.json").write_text(json.dumps({"messages": [{"role": "assistant"}]}))
    (case / "candidate_scaffold.yaml").write_text("candidate")
    (case / "patch.diff").write_text("patch")
    return source, target, case


def test_replay_matches_candidate_and_task_and_consumes_once(tmp_path):
    source, target, case = fixture_run(tmp_path)
    replay = MetricReplay(source, target, model="qwen", step_limit=250)
    assert replay.available == 1
    assert replay.restore("different", "task", target / "wrong") is None
    assert replay.restore("candidate", "other", target / "wrong") is None
    result = replay.restore("candidate", "task", target / "restored")
    assert result["evaluation"]["resolved"] is False
    assert result["resumed_from_metric_artifact"] == str(case)
    assert (target / "restored" / "patch.diff").read_text() == "patch"
    assert replay.restore("candidate", "task", target / "duplicate") is None
    assert replay.reused == 1


@pytest.mark.parametrize("damage", ["model", "limit", "report", "trajectory", "provider"])
def test_replay_rejects_incompatible_or_incomplete_results(tmp_path, damage):
    source, target, case = fixture_run(tmp_path)
    model, limit = "qwen", 250
    if damage == "model":
        model = "other"
    elif damage == "limit":
        limit = 50
    elif damage == "report":
        (case / "evaluation" / "report.json").write_text("{}")
    elif damage == "trajectory":
        (case / "trajectory.json").write_text("{")
    else:
        result = json.loads((case / "result.json").read_text())
        result["exit_status"] = "RateLimitError"
        (case / "result.json").write_text(json.dumps(result))
    assert MetricReplay(source, target, model=model, step_limit=limit).available == 0


def test_replay_rejects_changed_attribution(tmp_path):
    source, target, _ = fixture_run(tmp_path)
    (target / "provided_analysis" / "task.json").write_text('{"changed": true}')
    with pytest.raises(ValueError, match="identical"):
        MetricReplay(source, target, model="qwen", step_limit=250)
