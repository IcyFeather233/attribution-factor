"""Verify ReCreate attribution is a dedicated field, not evaluation output."""

import json
from pathlib import Path

import pytest
from jinja2 import Template, StrictUndefined

from recreate_agent.adapters.base import UnifiedResult, UnifiedInstance


EVIDENCE = {"failure_causing_wrong_step": {"step": 3, "what_went_wrong": "wrong file"}}


def _adapter(monkeypatch, payloads):
    from qwen_gpt55_experiments.recreate import run_native

    adapter = run_native.UdockerSWEBenchAdapter(step_limit=3, analysis_payloads=payloads)

    env = _FakeEnv()
    adapter.environments["container"] = env

    def fake_evaluate_patch(*, instance, patch, output_dir):
        return UnifiedResult(
            instance_id=instance.instance_id,
            success=False,
            score=0.0,
            details={"tests_passed": 0, "tests_failed": 1},
            formatted_output="clean evaluation output only",
        )

    monkeypatch.setattr(run_native, "evaluate_patch", fake_evaluate_patch)
    return adapter


class _FakeEnv:
    def execute(self, command, timeout=None):
        return {"returncode": 0, "output": "diff --git a/x b/x"}


def _instance():
    return UnifiedInstance(instance_id="django__django-1", problem_statement="task", domain_data={})


def test_injection_keeps_formatted_output_clean(monkeypatch, tmp_path):
    adapter = _adapter(monkeypatch, {"django__django-1": {"provided_information": EVIDENCE}})
    result = adapter.evaluate("container", _instance(), tmp_path)
    assert result.formatted_output == "clean evaluation output only"
    assert result.details["provided_failure_analysis"] == EVIDENCE


def test_baseline_does_not_inject(monkeypatch, tmp_path):
    adapter = _adapter(monkeypatch, {})
    assert adapter.inject_analysis_payloads is False
    result = adapter.evaluate("container", _instance(), tmp_path)
    assert "provided_failure_analysis" not in result.details


def test_result_payload_is_serialized_from_details():
    from qwen_gpt55_experiments.common.analysis_payloads import model_visible_payload
    details = {"tests_passed": 0, "provided_failure_analysis": EVIDENCE}
    serialized = json.dumps(details.get("provided_failure_analysis"), ensure_ascii=False, indent=2)
    assert "wrong file" in serialized
    assert serialized == json.dumps(model_visible_payload({"provided_information": EVIDENCE}), indent=2, ensure_ascii=False)


def test_meta_instance_template_renders_analysis_section(tmp_path):
    template_path = Path("baselines/recreate/src/recreate_agent/prompts/meta_instance.jinja2")
    template = Template(template_path.read_text(), undefined=StrictUndefined)

    failure = {
        "instance_id": "django__django-1",
        "problem_summary": "task",
        "exit_status": "Submitted",
        "error_type": "",
        "n_steps": 10,
        "total_cost": 0.0,
        "key_issues": ["missed branch"],
        "problematic_decisions": [],
        "eval_result": {"tests_passed": 0, "tests_failed": 1},
        "score": 0.0,
        "formatted_result": {
            "primary_metric_name": "Tests",
            "primary_metric_value": "0P/1F",
            "secondary_metrics": [],
            "failure_list_name": "Still Failing Tests",
            "failure_items": [],
            "extra_files": [],
        },
        "extra_files": [],
    }

    base_vars = {
        "recent_failures": [dict(failure)],
        "recent_successes": [],
        "ablation_eval_results": True,
        "ablation_trajectory": True,
    }

    without_analysis = template.render(**base_vars)
    assert "Provided Failure Analysis" not in without_analysis

    failure_with_analysis = dict(failure, provided_failure_analysis=json.dumps(EVIDENCE, indent=2))
    with_analysis = template.render(**dict(base_vars, recent_failures=[failure_with_analysis]))
    assert "Provided Failure Analysis (Trusted Evidence)" in with_analysis
    assert "wrong file" in with_analysis
    assert "clean evaluation output only" not in with_analysis
