import copy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from jinja2 import Environment, StrictUndefined

from attribution_experiments.common.flow_evidence import packet, records, render, handoff, select_references
from attribution_experiments.harnessfix.upstream_bridge import install_flow_diagnosis, flow_aggregation_evidence
from attribution_experiments.meta_harness.flow_views import make_views, citation_audit, environment_class


def payload(iid="task-1"):
    window = {"center_step": 2, "window_steps": [1, 2], "radius": 3}
    for kind, prefix in (("control_flow", "control"), ("data_flow", "data")):
        window[kind] = {"nodes": [
            {"node_id": f"{prefix}-step-1", "step": 1, "is_center": False, "observed_outcome": "negative test"},
            {"node_id": f"{prefix}-step-2", "step": 2, "is_center": True, "information_handling": "ignored result"},
        ], "edges": [{"source": f"{prefix}-step-1", "target": f"{prefix}-step-2", "type": "ignored",
                      "explanation": "The observation failed to constrain the next choice."}]}
    return {"task_id": iid, "provided_information": {
        "failure_causing_wrong_step": {"step": 2, "role": "implementation", "what_went_wrong": "ignored check", "evidence": "test output", "message_index": 4},
        "failure_causing_wrong_step_context": window,
    }}


def test_packet_keeps_all_original_graph_records_and_no_extra_advice():
    p = payload()
    before = copy.deepcopy(p)
    e = packet(p)
    for kind in ("control_flow", "data_flow"):
        actual = copy.deepcopy(e["windows"]["failure_causing_wrong_step"][kind])
        for item in actual["nodes"] + actual["edges"]:
            item.pop("evidence_ref")
        assert actual == p["provided_information"]["failure_causing_wrong_step_context"][kind]
    assert p == before
    text = render(p, current_context="current model call=80")
    assert "NOT indices" in text and "model call=80" in text
    assert "recoverability" not in text and "why_inevitable_here" not in text
    assert render({"provided_information": {}}) == ""


def test_reference_handoff_preserves_edges_and_endpoints_not_invented_citations():
    p = payload()
    ref = "task-1/failure_causing_wrong_step/data_flow-edge-0"
    h = handoff(p, "task-1", "support: " + ref)
    assert len(h["records"]) == 3
    assert h["records"][ref]["explanation"].startswith("The observation")
    assert handoff(p, "task-1", "no citations")["citation_status"] == "no_explicit_graph_citations"
    with pytest.raises(ValueError):
        select_references(packet(p), ["invented"])


def test_meta_requires_delivered_pages_and_both_edge_kinds(tmp_path):
    manifest = make_views(tmp_path, {"task-1": payload(), "task-2": payload("task-2")})
    refs = [ref for ref in manifest["references"] if "-edge-" in ref]
    proposal = {"flow_evidence_refs": refs}
    assert not citation_audit(proposal, manifest, set())["passed"]
    assert citation_audit(proposal, manifest, set(manifest["pages"]))["passed"]
    assert not citation_audit({"flow_evidence_refs": ["fabricated"]}, manifest, set(manifest["pages"]))["passed"]
    assert not citation_audit({"flow_evidence_refs": None}, manifest, set())["passed"]
    assert all(len(p["content"]) <= 6000 for p in manifest["pages"].values())

    class FakeEnv:
        def __init__(self):
            self.config = SimpleNamespace(cwd=tmp_path)
            self.output = ""
        def execute(self, action):
            if action['command'] == 'echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT':
                from minisweagent.exceptions import Submitted
                raise Submitted({'role': 'exit'})
            return {"returncode": 0, "output": self.output}
    env = environment_class(FakeEnv, manifest)()
    (tmp_path / "proposal.json").write_text(json.dumps(proposal))
    submit = {"command": "echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT"}
    assert env.execute(submit)["returncode"] == 2
    for path, page in manifest["pages"].items():
        env.output = page["content"][:30]
        env.execute({"command": "head " + path})
        assert path not in env.read_pages
        env.output = page["content"]
        env.execute({"command": "cat " + path})
        assert path in env.read_pages
    from minisweagent.exceptions import Submitted
    with pytest.raises(Submitted):
        env.execute(submit)


def test_harnessfix_native_diagnosis_receives_graph_and_aggregation_preserves_refs(tmp_path):
    (tmp_path / "task-1.json").write_text(json.dumps(payload()))
    (tmp_path / "baseline.json").write_text(json.dumps({"provided_information": {}}))
    refs = [ref for ref in records(packet(payload())) if "-edge-" in ref]
    class FakeAgent:
        def __init__(self):
            self.config = SimpleNamespace(instance_template="native", output_path=tmp_path / "trace.json")
        def run(self, **kwargs):
            self.kwargs = kwargs
            return {"submission": json.dumps({"flow_evidence_refs": refs})}
    module = SimpleNamespace(DefaultAgent=FakeAgent, parse_submission=json.loads)
    install_flow_diagnosis(module, tmp_path)
    agent = FakeAgent()
    agent.run(instance_id="task-1")
    assert "negative test" in agent.kwargs["historical_flow_evidence"]
    assert "flow_evidence_refs" in agent.config.instance_template
    plain = FakeAgent()
    plain.run(instance_id="baseline")
    assert plain.config.instance_template == "native"
    assert "historical_flow_evidence" not in plain.kwargs
    summary = flow_aggregation_evidence(tmp_path, [{"instance_id": "task-1", "flow_evidence_refs": refs}])
    assert "The observation failed" in summary and "negative test" in summary


def test_recreate_success_and_synthesis_templates_keep_historical_evidence():
    env = Environment(undefined=StrictUndefined)
    path = Path("baselines/recreate/src/recreate_agent/prompts")
    evidence = render(payload())
    success = dict(instance_id="task-1", problem_summary="task", n_steps=3, total_cost=0,
                   good_decisions=[], provided_failure_analysis=evidence)
    prompt = env.from_string((path / "meta_instance.jinja2").read_text()).render(
        recent_failures=[], recent_successes=[success])
    assert "CURRENT rollout succeeded" in prompt and "negative test" in prompt
    refs = [ref for ref in records(packet(payload())) if "-edge-" in ref]
    mod = dict(instance_id="task-1", success=True, score=1, summary="change", diff="diff",
               historical_flow_evidence={"text": evidence, "handoff": handoff(payload(), "task-1", " ".join(refs))})
    prompt = env.from_string((path / "meta_batch_synthesis.jinja2").read_text()).render(
        batch_idx=0, num_instances=1, num_modifications=1, modifications=[mod])
    assert "data_flow-edge-0" in prompt and "negative test" in prompt
    assert "historical_flow.md" in prompt
