"""Small path/config bridge into the unmodified HarnessFix CLI modules."""

from __future__ import annotations

import argparse
import copy
import json
import os
import sys
import time
from collections import Counter
from pathlib import Path
import fnmatch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from attribution_experiments.common.flow_evidence import (
    PROTOCOL as FLOW_PROTOCOL, has_flow, packet as flow_packet,
    render as render_flow, select_references, normalize_reference_spelling,
)


def install_flow_diagnosis(run_analysis, payload_dir: Path) -> None:
    original_run = run_analysis.DefaultAgent.run

    def run_with_flow(self, *args, **kwargs):
        task_id = kwargs.get("instance_id")
        payload = load_payload(payload_dir, task_id) if task_id else {}
        if payload and not has_flow(payload) and os.environ.get('HARNESSFIX_NATIVE_PROFILE') == '1':
            kwargs['historical_anchor_evidence'] = json.dumps(payload, ensure_ascii=False, indent=2)
            self.config.instance_template += (
                '\n\n## Supplied localization from the ORIGINAL failed trace\n'
                '{{historical_anchor_evidence}}\n'
                'Use these exact step/message indices, diagnosis and evidence alongside native HTIR. '
                'They describe the original trace, not a new candidate rollout. Verify source mechanisms; '
                'do not treat issue-specific facts as a general harness fix.'
            )
        if has_flow(payload):
            kwargs["historical_flow_evidence"] = render_flow(payload, task_id)
            self.config.instance_template += (
                "\n\n{{historical_flow_evidence}}\n\n"
                "Use this alongside the native HTIR and raw trace to diagnose a harness defect. "
                "Add flow_evidence_refs to your diagnosis JSON: a list of exact evidence_ref "
                "values supporting your diagnosis, including a control relation and a data "
                "relation when relevant. Explain the causal connection in failure_reason "
                "and agent_design_issue. Do not claim a relation proves an issue when it does not."
            )
        result = original_run(self, *args, **kwargs)
        if has_flow(payload):
            output = Path(self.config.output_path)
            receipt = {"protocol": FLOW_PROTOCOL, "task_id": task_id,
                       "stage": "native per-trace diagnosis", "delivered_chars": len(kwargs['historical_flow_evidence'])}
            try:
                parsed = run_analysis.parse_submission(result.get('submission', '')) or {}
                refs = parsed.get('flow_evidence_refs', [])
                refs, corrections = normalize_reference_spelling(flow_packet(payload, task_id), refs)
                receipt['reference_spelling_corrections'] = corrections
                receipt.update(flow_evidence_refs=refs, citation_status='cited' if refs else 'missing')
            except (TypeError, ValueError):
                receipt['citation_status'] = 'invalid'
            output.with_suffix('.flow_delivery.json').write_text(json.dumps(receipt, indent=2))
        return result

    run_analysis.DefaultAgent.run = run_with_flow


def flow_aggregation_evidence(payload_dir: Path, results: list[dict]) -> str:
    rows = []
    for result in results:
        iid = result.get("instance_id")
        if not iid:
            continue
        payload = load_payload(payload_dir, iid)
        if not has_flow(payload):
            continue
        refs = result.get("flow_evidence_refs", [])
        evidence = flow_packet(payload, iid)
        refs, corrections = normalize_reference_spelling(evidence, refs)
        selected = select_references(evidence, refs)
        rows.append({"task_id": iid, "component": result.get("affected_component"),
                     "defect_class": result.get("defect_class"), "flow_evidence_refs": refs,
                     "reference_spelling_corrections": corrections,
                     "citation_status": "cited" if refs else "no_explicit_graph_citations",
                     "historical_anchors": {key: window["anchor"] for key, window in evidence["windows"].items()},
                     "referenced_records_and_endpoints": selected})
    if not rows:
        return ""
    return (
        "## Diagnosis-to-aggregation historical flow evidence\n"
        "These are exact frozen graph records cited by per-case diagnosis, not current-rollout steps. "
        "Cluster native component/defect diagnoses using these control and data dependencies. "
        "Carry the supporting references into the repair-plan rationale; missing citations are "
        "explicitly marked and are not evidence of graph use.\n"
        + "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n\n"
    )


ROOT = Path(__file__).resolve().parents[2]
UPSTREAM = ROOT / "baselines" / "harnessfix"


def parse_args() -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("command", choices=["analyze", "aggregate"])
    parser.add_argument("--analysis-config", type=Path)
    parser.add_argument("--artifact-root", type=Path)
    parser.add_argument("--memory-root", type=Path)
    parser.add_argument("--payload-dir", type=Path, required=True)
    args, remainder = parser.parse_known_args()
    return args, remainder


def load_payload(payload_dir: Path, instance_id: str) -> dict:
    path = payload_dir / f"{instance_id}.json"
    if not path.is_file():
        raise FileNotFoundError(f"Missing HarnessFix ablation payload: {path}")
    return json.loads(path.read_text())["provided_information"]


def attribution_summary(payload_dir: Path, results: list[dict]) -> str:
    """Aggregate per-task attribution and local flow patterns into a harness prior."""
    if os.environ.get('HARNESSFIX_NATIVE_PROFILE') == '1':
        anchors = []
        for result in results:
            iid = result.get('instance_id')
            if not iid or not (payload_dir / (iid + '.json')).exists():
                continue
            payload = load_payload(payload_dir, iid)
            if payload and not has_flow(payload):
                anchors.append({'instance_id': iid, 'historical_localization': payload})
        if anchors:
            return '## Exact historical localization for analyzed failures\n' + '\n'.join(json.dumps(row, ensure_ascii=False) for row in anchors) + '\n\n'
    if any(has_flow(load_payload(payload_dir, r["instance_id"]))
           for r in results if r.get("instance_id") and (payload_dir / (r["instance_id"] + ".json")).exists()):
        return flow_aggregation_evidence(payload_dir, results)
    causing = Counter()
    inevitable = Counter()
    flow_stats = {
        "causing": {"phases": Counter(), "control_edges": Counter(), "data_edges": Counter()},
        "inevitable": {"phases": Counter(), "control_edges": Counter(), "data_edges": Counter()},
    }
    mechanism_examples = {"causing": [], "inevitable": []}
    handling_examples = {"causing": [], "inevitable": []}

    def add_example(target: list[str], value: object, limit: int = 3) -> None:
        text = " ".join(str(value or "").split())
        if text and text not in target and len(target) < limit:
            target.append(text[:320])

    for result in results:
        instance_id = result.get("instance_id")
        if not instance_id:
            continue
        try:
            payload = load_payload(payload_dir, instance_id)
        except FileNotFoundError:
            continue
        if not payload:
            continue
        c = payload.get("failure_causing_wrong_step")
        i = payload.get("failure_inevitable_step")
        if isinstance(c, dict) and c.get("role"):
            causing[str(c["role"])] += 1
        if isinstance(i, dict) and i.get("role"):
            inevitable[str(i["role"])] += 1

        for label, anchor_key in (
            ("causing", "failure_causing_wrong_step"),
            ("inevitable", "failure_inevitable_step"),
        ):
            context = payload.get(f"{anchor_key}_context")
            if not isinstance(context, dict):
                continue
            control = context.get("control_flow", {})
            data = context.get("data_flow", {})
            for node in control.get("nodes", []):
                if not isinstance(node, dict) or not node.get("is_center"):
                    continue
                if node.get("phase"):
                    flow_stats[label]["phases"][str(node["phase"])] += 1
                add_example(mechanism_examples[label], node.get("harness_mechanism"))
                add_example(mechanism_examples[label], node.get("relation_to_failure"))
            for edge in control.get("edges", []):
                if isinstance(edge, dict) and edge.get("type"):
                    flow_stats[label]["control_edges"][str(edge["type"])] += 1
            for node in data.get("nodes", []):
                if isinstance(node, dict) and node.get("is_center"):
                    add_example(handling_examples[label], node.get("information_handling"))
            for edge in data.get("edges", []):
                if isinstance(edge, dict) and edge.get("type"):
                    flow_stats[label]["data_edges"][str(edge["type"])] += 1

    has_flow_stats = any(
        stats["phases"] or stats["control_edges"] or stats["data_edges"]
        for stats in flow_stats.values()
    )
    if not causing and not inevitable and not has_flow_stats:
        return ""

    lines = []
    if causing:
        parts = ", ".join(f"{role}={count}" for role, count in causing.most_common())
        lines.append(f"- causing step roles: {parts}")
    if inevitable:
        parts = ", ".join(f"{role}={count}" for role, count in inevitable.most_common())
        lines.append(f"- inevitable step roles: {parts}")
    for label in ("causing", "inevitable"):
        stats = flow_stats[label]
        for title, key in (
            ("center phases", "phases"),
            ("local control transitions", "control_edges"),
            ("local data relations", "data_edges"),
        ):
            if stats[key]:
                parts = ", ".join(f"{name}={count}" for name, count in stats[key].most_common())
                lines.append(f"- {label} {title}: {parts}")
        if mechanism_examples[label]:
            lines.append(f"- {label} representative harness/failure mechanisms:")
            lines.extend(f"  - {item}" for item in mechanism_examples[label])
        if handling_examples[label]:
            lines.append(f"- {label} representative information handling:")
            lines.extend(f"  - {item}" for item in handling_examples[label])

    return (
        "## Provided Failure-Localization Summary\n"
        "Aggregated across the analyzed tasks. Implementation roles usually map "
        "to file_editing or testing; final_patch_review and verification roles map "
        "to testing or verification; design roles map to system_prompt. Use role "
        "distributions and local control/data-flow patterns as evidence when clustering "
        "harness-layer defects. Do not treat a task-specific source-code bug as a "
        "harness defect unless the same scaffold mechanism recurs across tasks.\n"
        + "\n".join(lines)
        + "\n\n"
    )


def validate_plan_consistency(spec: dict) -> None:
    """Reject contradictory instructions before a modify-agent spends its budget."""
    from baselines.harnessfix.failure_analysis.plan_diff_audit import _path_allowed, _fix_is_required
    budget = spec.get('edit_budget', {})
    forbidden = budget.get('forbidden_paths', [])
    conflicts = []
    for fix in spec.get('fixes', []):
        if not _fix_is_required(fix):
            continue
        for target in fix.get('target_files', []):
            for blocked in forbidden:
                if _path_allowed(target, [blocked]):
                    conflicts.append({'fix': fix.get('id'), 'target': target, 'forbidden': blocked})
    if conflicts:
        raise ValueError('Required fixes target forbidden paths: ' + json.dumps(conflicts))


SWE_TARGET_METRICS = frozenset({
    'resolved_rate', 'error_rate', 'empty_patch_rate', 'unresolved_rate',
    'avg_api_calls', 'loop_rate', 'verification_effort_ratio', 'syntax_error_rate',
})


def validate_swe_targets(spec: dict) -> None:
    unsupported = sorted({metric for fix in spec.get('fixes', [])
                          for metric in fix.get('target_metrics', []) if metric not in SWE_TARGET_METRICS})
    if unsupported:
        raise ValueError('Unsupported SWE target metrics: ' + ', '.join(unsupported)
                         + '. Available measured targets: ' + ', '.join(sorted(SWE_TARGET_METRICS)))


def normalize_spec_schema(spec: dict) -> dict:
    """Fill protocol-only fields without inventing a missing repair proposal."""
    if not isinstance(spec, dict):
        return spec

    metadata = spec.setdefault("plan_metadata", {})
    if isinstance(metadata, dict):
        metadata.setdefault("round_label", "HarnessFix Round 1 Analysis")
        metadata.setdefault("summary", "Scoped harness repair plan")
        metadata.setdefault("global_strategy", "Implement the declared fixes in priority order.")

    fixes = spec.get("fixes")
    declared_files: list[str] = []
    if isinstance(fixes, list):
        for index, fix in enumerate(fixes, start=1):
            if not isinstance(fix, dict):
                continue
            target_files = fix.setdefault("target_files", [])
            if isinstance(target_files, list):
                declared_files.extend(str(path) for path in target_files if path)
            title = str(fix.get("title") or fix.get("id") or f"Fix {index}")
            fix.setdefault("id", f"fix_{index}")
            fix.setdefault("title", title)
            fix.setdefault("priority", "P1")
            if fix.get('optional') is True:
                fix.setdefault('required', False)
            fix.setdefault("target_symbols", [])
            fix.setdefault("problem_statement", f"Address the failure pattern described by {title}.")
            fix.setdefault("required_behavior_delta", f"Implement the behavior change described in the plan for {title}.")
            fix.setdefault("implementation_steps", ["Apply the scoped change described in the plan to the declared target files."])
            fix.setdefault("tests", ["Run targeted regression tests for the declared behavior change."])
            fix.setdefault("risk_level", "medium")
            fix.setdefault("dependencies", [])
            fix.setdefault("regression_risks", ["Behavior outside the declared repair scope could regress; run targeted checks."])
            fix.setdefault("must_not_change", ["Behavior outside the declared repair scope."])

    budget = spec.setdefault("edit_budget", {})
    if isinstance(budget, dict):
        allowed = budget.setdefault("allowed_paths", [])
        if not isinstance(allowed, list):
            allowed = []
            budget["allowed_paths"] = allowed
        for path in declared_files:
            if path not in allowed:
                allowed.append(path)
        unique_files = list(dict.fromkeys(declared_files))
        file_count = max(1, len(unique_files))
        budget.setdefault("recommended_budget", "low" if file_count == 1 else "medium" if file_count <= 3 else "high")
        budget.setdefault("max_files_to_modify", file_count)
        budget.setdefault("forbidden_paths", [])
        budget.setdefault("rationale", "Budget derived deterministically from declared target files.")
    return spec


def main() -> None:
    args, remainder = parse_args()
    sys.path.insert(0, str(UPSTREAM))
    sys.path.insert(0, str(UPSTREAM / "agent_framework" / "src"))

    if args.command == "analyze":
        if not args.analysis_config:
            raise ValueError("analyze requires --analysis-config")
        from failure_analysis import run_analysis

        install_flow_diagnosis(run_analysis, args.payload_dir)

        original_environment_execute = run_analysis.TruncatingLocalEnvironment.execute

        def bounded_environment_execute(self, action, cwd="", *, timeout=None):
            self._ablation_action_count = getattr(self, "_ablation_action_count", 0) + 1
            command = str(action.get("command", "")) if isinstance(action, dict) else str(action)
            may_finish = "/tmp/result_" in command or "COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT" in command
            if self._ablation_action_count > 5 and not may_finish:
                return {
                    "returncode": 2,
                    "exception_info": "",
                    "output": (
                        "STRICT CONTROLLER: evidence-reading budget exhausted. Do not inspect "
                        "anything else. Your next command must write the required diagnosis to "
                        "/tmp/result_INSTANCE_ID.json, then submit it on the following command."
                    ),
                    "extra": {"ablation_evidence_budget_blocked": True},
                }
            return original_environment_execute(self, action, cwd=cwd, timeout=timeout)

        if os.environ.get('HARNESSFIX_NATIVE_PROFILE') != '1':
            run_analysis.TruncatingLocalEnvironment.execute = bounded_environment_execute
        run_analysis.CONFIG_PATH_SWE = args.analysis_config.resolve()
        if args.artifact_root:
            run_analysis.RESULTS_DIR = args.artifact_root.resolve()
        sys.argv = [str(UPSTREAM / "failure_analysis" / "run_analysis.py"), *remainder]
        run_analysis.main()
        return

    from failure_analysis import aggregate_results
    native_template = aggregate_results.AGGREGATE_AGENT_INSTANCE_TEMPLATE
    native_step_limit = aggregate_results.AGG_AGENT_STEP_LIMIT

    aggregate_results.MODEL_KWARGS["max_tokens"] = 8192
    aggregate_results.AGG_AGENT_STEP_LIMIT = 60
    aggregate_results.AGGREGATE_AGENT_INSTANCE_TEMPLATE = (
        "STRICT EXECUTION PROTOCOL:\n"
        "- You have at most eight evidence/source-inspection commands.\n"
        "- Inspect only the listed context and recommended harness source files; do not "
        "search external datasets, gold patches, caches, or unrelated experiment runs.\n"
        "- By command nine, write a complete scoped plan and fenced JSON spec to "
        "/tmp/aggregate_plan.md. If evidence is uncertain, state that uncertainty and "
        "choose the narrowest defensible repair.\n"
        "- Immediately submit the file on the following command.\n\n"
        + aggregate_results.AGGREGATE_AGENT_INSTANCE_TEMPLATE
    )
    aggregate_plan_path = f"/tmp/aggregate_plan_{os.getpid()}.md"
    if os.environ.get('HARNESSFIX_NATIVE_PROFILE') == '1':
        scratch = args.payload_dir.parent / 'aggregate_scratch'
        scratch.mkdir(exist_ok=True)
        aggregate_plan_path = str(scratch / f'aggregate_plan_{os.getpid()}.md')
    aggregate_results.AGGREGATE_AGENT_INSTANCE_TEMPLATE = (
        aggregate_results.AGGREGATE_AGENT_INSTANCE_TEMPLATE.replace(
            "/tmp/aggregate_plan.md", aggregate_plan_path
        )
    )

    original_aggregate_environment_execute = aggregate_results.LocalEnvironment.execute

    def bounded_aggregate_environment_execute(self, action, cwd="", *, timeout=None):
        self._ablation_aggregate_action_count = (
            getattr(self, "_ablation_aggregate_action_count", 0) + 1
        )
        command = str(action.get("command", "")) if isinstance(action, dict) else str(action)
        may_finish = (
            "/tmp/aggregate_plan" in command
            or "COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT" in command
        )
        if self._ablation_aggregate_action_count > 8 and not may_finish:
            return {
                "returncode": 2,
                "exception_info": "",
                "output": (
                    "STRICT CONTROLLER: aggregate evidence-reading budget exhausted. "
                    "Do not inspect anything else. Your next command must write the "
                    "complete Markdown plan and fenced JSON spec to "
                    f"{aggregate_plan_path}, then submit it on the following command."
                ),
                "extra": {"ablation_aggregate_evidence_budget_blocked": True},
            }
        return original_aggregate_environment_execute(
            self, action, cwd=cwd, timeout=timeout
        )

    if os.environ.get('HARNESSFIX_NATIVE_PROFILE') == '1':
        aggregate_results.AGGREGATE_AGENT_INSTANCE_TEMPLATE = native_template.replace('/tmp/aggregate_plan.md', aggregate_plan_path)
        aggregate_results.AGGREGATE_AGENT_INSTANCE_TEMPLATE += (
            '\nMeasured SWE target metrics: ' + ', '.join(sorted(SWE_TARGET_METRICS))
            + '. Use only these names in target_metrics. Do not invent accuracy/exact_match or unmeasured metrics. '
            'Container creation and cleanup are fixed transport responsibilities; preserve their class methods. '
            'Required target files must not match any forbidden path, including /... directory patterns.\n'
        )
        aggregate_results.AGG_AGENT_STEP_LIMIT = native_step_limit
    else:
        aggregate_results.LocalEnvironment.execute = bounded_aggregate_environment_execute

    original_validate_spec = aggregate_results.validate_spec

    def validate_spec_with_defaults(spec):
        normalize_spec_schema(spec)
        original_validate_spec(spec)
        if os.environ.get('HARNESSFIX_NATIVE_PROFILE') == '1':
            validate_plan_consistency(spec)
            validate_swe_targets(spec)

    aggregate_results.validate_spec = validate_spec_with_defaults

    original_aggregate_agent = aggregate_results.call_aggregate_agent

    def call_aggregate_agent_with_retry(**kwargs):
        errors = []
        for attempt in range(2):
            try:
                return original_aggregate_agent(**kwargs)
            except (ValueError, RuntimeError) as exc:
                errors.append(f"attempt {attempt + 1}: {exc}")
                saved_plan = Path(aggregate_plan_path)
                if saved_plan.is_file():
                    saved_text = saved_plan.read_text()
                    if len(saved_text) >= 200 and "```json" in saved_text:
                        return saved_text
                if attempt == 0:
                    time.sleep(1)
        raise RuntimeError(
            "HarnessFix aggregate protocol retries exhausted: " + " | ".join(errors)
        )

    aggregate_results.call_aggregate_agent = call_aggregate_agent_with_retry

    original_generate_spec = aggregate_results.generate_spec_from_plan

    def generate_spec_with_retry(*call_args, **call_kwargs):
        errors = []
        for attempt in range(3):
            try:
                return original_generate_spec(*call_args, **call_kwargs)
            except (ValueError, json.JSONDecodeError) as exc:
                errors.append(f"attempt {attempt + 1}: {exc}")
                if os.environ.get('HARNESSFIX_NATIVE_PROFILE') == '1' and 'plan_text' in call_kwargs:
                    call_kwargs['plan_text'] += '\n\nSchema validation failed: ' + str(exc) + '\nReturn a consistent scoped spec; do not require changes under forbidden paths.'
                elif os.environ.get('HARNESSFIX_NATIVE_PROFILE') == '1' and len(call_args) > 1:
                    values = list(call_args)
                    values[1] += '\n\nSchema validation failed: ' + str(exc) + '\nReturn a consistent scoped spec; do not require changes under forbidden paths.'
                    call_args = tuple(values)
                if attempt < 2:
                    time.sleep(1)
        raise RuntimeError(
            "HarnessFix spec protocol retries exhausted: " + " | ".join(errors)
        )

    aggregate_results.generate_spec_from_plan = generate_spec_with_retry

    original_formatter = aggregate_results.format_results_for_llm

    def format_results_with_payload(results):
        formatted = original_formatter(results)
        summary = attribution_summary(args.payload_dir, results)
        if not summary:
            return formatted
        return summary + formatted

    aggregate_results.format_results_for_llm = format_results_with_payload
    if args.memory_root:
        memory_root = args.memory_root.resolve()
        aggregate_results.default_memory_root = lambda _repo_root: memory_root
    sys.argv = [str(UPSTREAM / "failure_analysis" / "aggregate_results.py"), *remainder]
    aggregate_results.main()


if __name__ == "__main__":
    main()
