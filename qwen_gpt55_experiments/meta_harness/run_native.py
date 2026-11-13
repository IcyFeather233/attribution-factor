"""Run one native Meta-Harness iteration for mini-swe-agent.

The upstream repository does not ship a SWE-bench adapter. This domain adapter
keeps its analyze -> implement -> validate -> benchmark -> frontier loop and
replaces only the documented proposer wrapper with GPT-5.5 through llm-sidecar.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import inspect
import json
import os
import shutil
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import yaml
from jinja2 import Environment, TemplateSyntaxError
from minisweagent.agents.default import DefaultAgent
from minisweagent.environments.local import LocalEnvironment

from qwen_gpt55_experiments.common.analysis_payloads import (
    DEFAULT_PAYLOAD_ROOT,
    GROUPS,
    GROUP_BASELINE,
    trace_path,
    load_payloads,
    model_visible_payload,
    snapshot_payloads,
)

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from qwen_gpt55_experiments.common.dataset import (
    load_failed_instances,
    select_repository_balanced_instances,
)
from qwen_gpt55_experiments.common.model_config import (
    OPTIMIZER_MODEL,
    TASK_MODEL,
    TASK_STEP_LIMIT,
    optimizer_api_key,
    task_api_key,
)
from qwen_gpt55_experiments.common.runner import build_optimizer_model, load_original_config, redact_secret, run_task


RUN_ROOT = ROOT / "qwen_gpt55_experiments" / "meta_harness"
UPSTREAM = ROOT / "baselines" / "meta-harness"
UDOCKER_REPO = ROOT / "native_baseline_runs" / "udocker"
UDOCKER_EXECUTABLE = ROOT / "native_baseline_runs" / ".venv" / "bin" / "udocker"
AGENT_TEXT_FIELDS = ("system_template", "instance_template")
MODEL_TEXT_FIELDS = ("format_error_template", "observation_template", "multimodal_regex")
PROTECTED_SURFACE = (
    "fixed Qwen3.5-35B-A3B task-model identity and credentials",
    "250-step and evaluator resource boundary",
    "SWE-bench task data and container images",
    "official evaluator and patch extraction",
)


ANALYST_SYSTEM = """You are the analysis proposer in one native Meta-Harness iteration.
You optimize the complete code and prompt harness around a fixed language model.
Work only inside the current experiment directory and use bash tools to inspect files.
Every bash call already starts in the experiment directory. Commands run in
independent shells: use relative paths and never cd to /workspace.

Follow the Meta-Harness method:
- Read agents/baseline_agent.py, agents/baseline_scaffold.yaml, parent_source/,
  the current frontier, past evolution summary, raw execution trajectories, and
  supporting task JSON under provided_analysis/.
- Treat provided_analysis/ as additional failure-localization evidence supplied
  to this analysis phase; use it together with the raw trajectories.
- Deep-read failures and any successes that exist.
- Search for one general, recurring harness-level failure mechanism.
- Consider arbitrary Python-level interventions: control flow, context/history,
  query policy, observation processing, verification, tool orchestration, static
  memory/assets, or prompt/config changes.
- State one falsifiable hypothesis and predict score and efficiency impact.
- Never hardcode task names, repository names, issue-specific facts, patches, or
  expected answers. A mechanism must generalize to unfamiliar SWE tasks.
- One candidate tests exactly one mechanism.
- Analyze the saved trajectories only. Do not start/reuse task containers,
  reproduce failures, install packages, or run benchmarks in this phase.
- Do not benchmark or implement the candidate in this phase.
- Write proposal.json with fields name, hypothesis, changes,
  expected_efficiency, and evidence_summary.
- Validate proposal.json with python, then finish by running:
  echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT
"""


IMPLEMENTER_SYSTEM = """You are the implementation proposer in one native Meta-Harness iteration.
Work only inside the current experiment directory and use bash tools. The
analysis role is complete: do not reanalyze trajectories or invent a new idea.
Every bash call already starts in the experiment directory. Commands run in
independent shells: use relative paths and never cd to /workspace.

Read proposal.json, agents/baseline_agent.py, agents/baseline_scaffold.yaml,
agents/baseline_assets/, and parent_source/. Implement exactly the one mechanism
selected in proposal.json:
- Copy agents/baseline_agent.py to agents/candidate_agent.py.
- Copy agents/baseline_scaffold.yaml to agents/candidate_scaffold.yaml.
- The candidate must define class AgentHarness and subclass the installed
  minisweagent.agents.default.DefaultAgent, directly or indirectly.
- You may rewrite or override ANY AgentHarness method: run, step, query,
  execute_actions, context handling, observation processing, verification, or
  tool orchestration. Arbitrary Python code inside candidate_agent.py is the
  native search space.
- You may modify any mutable text field in candidate_scaffold.yaml.
- You may create agents/candidate_assets/ with reusable tools or static memories.
  During evaluation that directory is mounted at /workspace/meta_harness in an
  isolated copy for each task. Mention that path in the scaffold if the task
  agent should use those assets.
- Do not modify baseline files, parent_source, evaluator code, task data, model
  identity, credentials, or the 250-step resource boundary.
- Do not encode task names, repository names, issue facts, expected patches, or
  hidden answers.
- Keep one mechanism only. Do not bundle unrelated prompt and code ideas.
- Write pending_eval.json containing iteration 1 and exactly one candidate with
  name, agent_path, scaffold_path, assets_path, hypothesis, changes, and
  expected_efficiency.
- Validate Python import, YAML, and JSON locally. Do not run SWE-bench evaluation;
  the trusted outer loop does that.
- Finish by running: echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT
"""


DOMAIN_SPEC = """# Meta-Harness SWE-bench domain adapter

Candidate shape: a complete mini-swe-agent harness workspace:
- agents/candidate_agent.py: arbitrary Python AgentHarness implementation
- agents/candidate_scaffold.yaml: all mutable agent/model text templates
- agents/candidate_assets/: optional reusable tools and static memories

The fixed Qwen3.5-35B-A3B task-model identity, credentials, 250-step resource boundary,
container/task data, patch extraction, and official SWE-bench evaluator are not
searchable. Candidate assets are copied per task and mounted at
/workspace/meta_harness.

Search evidence: the 148 confirmed Qwen3.5 failed raw mini-swe-agent trajectories,
baseline harness source/scaffold, previous candidate summaries, trusted
evaluator scores, and the experiment-provided ablation payloads under
provided_analysis/. Successful comparison-model traces, gold patches, and all
non-selected analysis fields are prohibited.

One iteration follows the upstream Terminal-Bench reference workflow:
analysis proposer -> implementation proposer -> import/schema/anti-leak
validation -> one candidate benchmark -> frontier update. The upstream native
search space is arbitrary Python harness code; this adapter preserves that
property for mini-swe-agent.
"""


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False))


def prepare_search_workspace(
    run_dir: Path,
    instances: list[dict],
    smoke: bool,
    analysis_payloads: dict[str, dict],
    inject_analysis: bool,
) -> Path:
    search = run_dir / "search"
    if search.exists():
        raise RuntimeError('Search workspace already exists; use an explicit recovery or a fresh run directory')
    (search / "agents" / "baseline_assets").mkdir(parents=True)
    (search / "traces").mkdir()
    (search / "parent_source").mkdir()
    if inject_analysis:
        (search / "provided_analysis").mkdir()

    original = load_original_config()
    scaffold = {
        "agent": {key: original["agent"][key] for key in AGENT_TEXT_FIELDS},
        "model": {key: original["model"][key] for key in MODEL_TEXT_FIELDS},
    }
    (search / "agents" / "baseline_scaffold.yaml").write_text(
        yaml.safe_dump(scaffold, sort_keys=False, width=10000)
    )

    default_source = (
        ROOT / "qwen_failure_comparison" / "mini-swe-agent-v2.4.6" / "src" / "minisweagent" / "agents" / "default.py"
    ).read_text()
    default_source = default_source.replace(
        "from minisweagent import Environment, Model, __version__",
        "from minisweagent import Environment, Model, __version__\n"
        "from minisweagent.agents.default import DefaultAgent as MiniSWEAgent",
    )
    default_source = default_source.replace(
        "class DefaultAgent:",
        "class AgentHarness(MiniSWEAgent):",
        1,
    )
    (search / "agents" / "baseline_agent.py").write_text(default_source)
    (search / "agents" / "baseline_assets" / "README.md").write_text(
        "# Optional harness assets\n\n"
        "Candidate tools and static memories placed here are copied per task and "
        "mounted at /workspace/meta_harness.\n"
    )

    selected = instances[:1] if smoke else instances
    for instance in selected:
        source = trace_path(instance["instance_id"])
        shutil.copy(source, search / "traces" / source.name)
        if inject_analysis:
            write_json(
                search / "provided_analysis" / f"{instance['instance_id']}.json",
                model_visible_payload(analysis_payloads[instance["instance_id"]]),
            )

    from .corpus_index import write_corpus_index
    write_corpus_index(search, selected, inject_analysis)

    source_files = [
        ROOT / "qwen_failure_comparison" / "mini-swe-agent-v2.4.6" / "src" / "minisweagent" / "agents" / "default.py",
        ROOT / "qwen_failure_comparison" / "mini-swe-agent-v2.4.6" / "src" / "minisweagent" / "models" / "litellm_model.py",
        ROOT / "qwen_gpt55_experiments" / "common" / "runner.py",
        ROOT / "qwen_gpt55_experiments" / "common" / "udocker_environment.py",
    ]
    for source in source_files:
        shutil.copy(source, search / "parent_source" / source.name)

    baseline = {
        "agent": "baseline",
        "avg_pass_rate": 0.0,
        "total": len(selected),
        "resolved": 0,
        "source": "given original failed traces",
    }
    write_json(search / "frontier_val.json", {"_best": baseline})
    (search / "evolution_summary.jsonl").write_text(json.dumps(baseline) + "\n")
    domain_spec = DOMAIN_SPEC
    if not inject_analysis:
        domain_spec = domain_spec.replace(
            " and the experiment-provided ablation payloads under\nprovided_analysis/", ""
        )
    (search / "domain_spec.md").write_text(domain_spec)
    shutil.copy(
        UPSTREAM
        / "reference_examples"
        / "terminal_bench_2"
        / ".claude"
        / "skills"
        / "meta-harness-terminal-bench-2"
        / "SKILL.md",
        search / "upstream_meta_harness_skill.md",
    )
    return search


def run_proposer(
    *,
    search: Path,
    api_key: str,
    system_template: str,
    task: str,
    trajectory_path: Path,
    step_limit: int,
) -> dict:
    from .proposer_logs import publish_evidence_log
    runtime_log = search.parent / 'proposer_runtime_logs' / trajectory_path.name
    runtime_log.parent.mkdir(parents=True, exist_ok=True)
    environment = LocalEnvironment
    flow_manifest_path = search / "flow_views_manifest.json"
    if trajectory_path.name == "analysis.traj.json" and flow_manifest_path.exists():
        from .flow_views import environment_class
        environment = environment_class(LocalEnvironment, json.loads(flow_manifest_path.read_text()))
    env = environment(
        cwd=str(search),
        timeout=120,
        env={"PAGER": "cat", "MANPAGER": "cat"},
    )
    agent = DefaultAgent(
        model=build_optimizer_model(api_key=api_key, temperature=1.0),
        env=env,
        system_template=system_template,
        instance_template="{{task}}",
        step_limit=step_limit,
        cost_limit=0,
        output_path=runtime_log,
    )
    try:
        result = agent.run(task)
    finally:
        agent.save(runtime_log)
        redact_secret(runtime_log, api_key)
        publish_evidence_log(runtime_log, trajectory_path)
    return {
        "exit_status": result.get("exit_status", ""),
        "api_calls": agent.n_calls,
        "submission": result.get("submission", ""),
    }


def load_candidate_agent(path: Path) -> type[DefaultAgent]:
    module_name = f"meta_harness_candidate_{hashlib.sha256(path.read_bytes()).hexdigest()[:12]}"
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load candidate module from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    agent_class = getattr(module, "AgentHarness", None)
    if not inspect.isclass(agent_class) or not issubclass(agent_class, DefaultAgent):
        raise TypeError("candidate AgentHarness must subclass DefaultAgent")
    return agent_class


def validate_candidate(
    search: Path,
) -> tuple[dict, dict, type[DefaultAgent], Path | None]:
    proposal_path = search / "proposal.json"
    pending_path = search / "pending_eval.json"
    candidate_agent_path = search / "agents" / "candidate_agent.py"
    candidate_scaffold_path = search / "agents" / "candidate_scaffold.yaml"
    candidate_assets_path = search / "agents" / "candidate_assets"
    required_paths = {
        "proposal.json": proposal_path,
        "pending_eval.json": pending_path,
        "candidate_agent.py": candidate_agent_path,
        "candidate_scaffold.yaml": candidate_scaffold_path,
    }
    missing = [name for name, candidate_path in required_paths.items() if not candidate_path.exists()]
    if missing:
        raise RuntimeError(f"implementation proposer did not create: {', '.join(missing)}")

    proposal = json.loads(proposal_path.read_text())
    pending = json.loads(pending_path.read_text())
    scaffold = yaml.safe_load(candidate_scaffold_path.read_text())
    required_proposal = {"name", "hypothesis", "changes", "expected_efficiency"}
    if not required_proposal.issubset(proposal):
        raise ValueError(f"proposal missing fields: {required_proposal - set(proposal)}")
    candidates = pending.get("candidates")
    if candidates is None and isinstance(pending.get("candidate"), dict):
        candidates = [pending.pop("candidate")]
        pending["candidates"] = candidates
        write_json(pending_path, pending)
    if not isinstance(candidates, list) or len(candidates) != 1:
        raise ValueError("pending_eval.json must contain exactly one candidate")

    if not isinstance(scaffold, dict) or set(scaffold) != {"agent", "model"}:
        raise ValueError("candidate scaffold must contain exactly agent and model sections")
    expected = {"agent": set(AGENT_TEXT_FIELDS), "model": set(MODEL_TEXT_FIELDS)}
    for section, fields in expected.items():
        values = scaffold.get(section)
        if not isinstance(values, dict) or set(values) != fields:
            raise ValueError(f"candidate {section} fields must be exactly {sorted(fields)}")
        for key, value in values.items():
            if not isinstance(value, str):
                raise ValueError(f"candidate {section}.{key} must be a string")
            if key != "multimodal_regex" and not value.strip():
                raise ValueError(f"candidate {section}.{key} must not be empty")
            try:
                Environment().parse(value)
            except TemplateSyntaxError as exc:
                raise ValueError(f"invalid Jinja in {section}.{key}: {exc}") from exc
    if "{{task" not in scaffold["agent"]["instance_template"].replace(" ", ""):
        raise ValueError("candidate instance_template must render task")

    agent_class = load_candidate_agent(candidate_agent_path)
    assets = candidate_assets_path if candidate_assets_path.is_dir() else None
    return proposal, scaffold, agent_class, assets


def audit_candidate_artifact(search: Path, instances: list[dict]) -> dict:
    paths = [
        search / "agents" / "candidate_agent.py",
        search / "agents" / "candidate_scaffold.yaml",
    ]
    assets = search / "agents" / "candidate_assets"
    if assets.is_dir():
        paths.extend(path for path in assets.rglob("*") if path.is_file())

    forbidden_tokens = {
        "qwen3.5-35b-a3b_success_comparisons",
        "SWE-bench_Verified",
        "evaluate_patch",
        "SIDECAR_API_KEY",
        "gold_patch",
        "failure_analysis.json",
        *[instance["instance_id"] for instance in instances],
    }
    hits = []
    for artifact_path in paths:
        content = artifact_path.read_text(errors="replace")
        for token in forbidden_tokens:
            if token in content:
                hits.append({"path": str(artifact_path.relative_to(search)), "token": token})
    result = {
        "passed": not hits,
        "checked_paths": [str(path.relative_to(search)) for path in paths],
        "forbidden_reference_hits": hits,
    }
    write_json(search / "candidate_audit.json", result)
    return result


def audit_proposer_inputs(search: Path) -> dict:
    from .input_audit import audit_proposer_inputs as audit
    return audit(search)


def benchmark_candidate(
    *,
    instances: list[dict],
    scaffold: dict,
    agent_class: type[DefaultAgent],
    assets_dir: Path | None,
    output_dir: Path,
    api_key: str,
    step_limit: int,
    workers: int,
) -> dict:
    def one(instance: dict) -> dict:
        return run_task(
            instance=instance,
            output_dir=output_dir / instance["instance_id"],
            udocker_repo=UDOCKER_REPO,
            udocker_executable=UDOCKER_EXECUTABLE,
            api_key=api_key,
            agent_overrides=scaffold["agent"],
            model_overrides=scaffold["model"],
            agent_class=agent_class,
            harness_assets_dir=assets_dir,
            step_limit=step_limit,
        )

    by_id = {}
    failed_tasks = []
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(one, item): item["instance_id"] for item in instances}
        for future in as_completed(futures):
            task_id = futures[future]
            try:
                result = future.result()
                from qwen_gpt55_experiments.common.result_validity import retryable_result
                if retryable_result(result) or not isinstance(result.get('evaluation', {}).get('resolved'), bool):
                    raise RuntimeError('Task has no gradeable evaluation')
                by_id[task_id] = result
            except Exception as exc:
                failed_tasks.append(task_id)
                # Preserve the actual exception: a missing grade alone cannot
                # distinguish an API/transport failure from a candidate error.
                failure_dir = output_dir / task_id
                failure_dir.mkdir(parents=True, exist_ok=True)
                failure_path = failure_dir / 'benchmark_error.json'
                write_json(failure_path, {'instance_id': task_id, 'error_type': type(exc).__name__,
                                         'message': str(exc)})
                redact_secret(failure_path, api_key)
    if failed_tasks:
        raise RuntimeError(f'Benchmark incomplete; preserve successful artifacts and resume these tasks: {failed_tasks}')

    results = []
    for instance in instances:
        run = by_id[instance["instance_id"]]
        evaluation = run.get("evaluation", {})
        results.append(
            {
                "instance_id": instance["instance_id"],
                "resolved": bool(evaluation.get("resolved")),
                "status": evaluation.get(
                    "error", "resolved" if evaluation.get("resolved") else run.get("error", "unresolved")
                ),
            }
        )
    return {
        "resolved": sum(item["resolved"] for item in results),
        "total": len(results),
        "avg_pass_rate": sum(item["resolved"] for item in results) / max(1, len(results)),
        "results": results,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-name", default="native_round1")
    parser.add_argument("--ablation-group", choices=GROUPS, required=True)
    parser.add_argument("--payload-root", type=Path, default=DEFAULT_PAYLOAD_ROOT)
    parser.add_argument("--benchmark-workers", type=int, default=1)
    parser.add_argument("--adaptation-tasks", type=int, default=148)
    parser.add_argument("--adaptation-seed", type=int, default=20260902)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--resume-analysis-proposal", action="store_true",
                        help="Reuse a recovered and audited proposal; continue with native implementation")
    args = parser.parse_args()
    if args.benchmark_workers < 1:
        parser.error("--benchmark-workers must be at least 1")
    if args.adaptation_tasks < 1:
        parser.error("--adaptation-tasks must be at least 1")
    optimizer_key = optimizer_api_key()
    task_key = task_api_key()

    run_dir = RUN_ROOT / args.ablation_group / args.run_name
    from qwen_gpt55_experiments.common.execution_protocol import record_execution_backend
    analysis_payloads = load_payloads(args.ablation_group, args.payload_root.resolve())
    instances = load_failed_instances()
    if args.adaptation_tasks > len(instances):
        parser.error(f"--adaptation-tasks cannot exceed {len(instances)}")
    adaptation_instances = (
        instances[:1]
        if args.smoke
        else select_repository_balanced_instances(
            instances,
            limit=args.adaptation_tasks,
            seed=args.adaptation_seed,
        )
    )
    selected = instances[:1] if args.smoke else instances
    from qwen_gpt55_experiments.common.optimization_budget import pin_budget
    pin_budget(run_dir, {'protocol': 'meta_full_corpus_v3', 'group': args.ablation_group,
                        'analysis_task_ids': [x['instance_id'] for x in adaptation_instances],
                        'analyst_steps': 40, 'implementer_steps': 30},
               has_progress=(run_dir / 'search').exists())
    record_execution_backend(run_dir)
    snapshot_payloads(run_dir, args.ablation_group, analysis_payloads)
    inject_analysis = args.ablation_group != GROUP_BASELINE
    search = run_dir / 'search' if args.resume_analysis_proposal else prepare_search_workspace(
        run_dir,
        adaptation_instances,
        args.smoke,
        analysis_payloads,
        inject_analysis,
    )
    (search / "logs").mkdir(exist_ok=args.resume_analysis_proposal)
    analyst_system = ANALYST_SYSTEM
    analysis_task = (
        "Run analysis phase for iteration 1. Read domain_spec.md, "
        "agents/baseline_agent.py, agents/baseline_scaffold.yaml, parent_source/, "
        "frontier_val.json, evolution_summary.jsonl, and raw traces under traces/. "
        "Use traces/_index.jsonl to search the entire supplied corpus for recurring patterns, "
        "then deep-read supporting raw trajectories. Corpus size is availability, not proof "
        "that every trace has been read; report the actual task IDs supporting your diagnosis."
    )
    if inject_analysis:
        analysis_task += (
            " Search the supplied localization corpus under provided_analysis/ and read "
            "the complete matching payloads for the traces supporting your diagnosis."
        )
        from .flow_views import make_views
        from qwen_gpt55_experiments.common.flow_evidence import write_receipt
        selected_payloads = {item['instance_id']: analysis_payloads[item['instance_id']] for item in adaptation_instances}
        views = {} if args.resume_analysis_proposal else make_views(search, selected_payloads)
        write_receipt(run_dir, selected_payloads, 'native analysis proposer paged graph evidence')
        if views:
            analysis_task += (
                "\n\n" + views['index'] + "\n\n"
                "Read complete pages using cat one page per tool call; each is below the output limit. "
                "Deep-read both control and data dependencies for at least two tasks supporting "
                "your recurring mechanism. Cite exact evidence_ref values in proposal.json under "
                "flow_evidence_refs, including control and data edges across those tasks. "
                "Explain their causal relevance in evidence_summary. Dictionary keys, filenames, "
                "head-only previews and type frequencies are not substitutes for reading the graph."
            )
    analysis_task += (
        " Write proposal.json by call 30; reserve remaining calls for JSON validation "
        "and submission."
    )
    if not inject_analysis:
        analyst_system = analyst_system.replace(
            ", and\n  supporting task JSON under provided_analysis/.", "."
        ).replace(
            "- Treat provided_analysis/ as additional failure-localization evidence supplied\n"
            "  to this analysis phase; use it together with the raw trajectories.\n",
            "",
        )

    if args.resume_analysis_proposal:
        from .recover_analysis_proposal import recover_proposal
        analysis = recover_proposal(search)
    else:
        analysis = run_proposer(
            search=search, api_key=optimizer_key, system_template=analyst_system,
            task=analysis_task, trajectory_path=search / "logs" / "analysis.traj.json", step_limit=40,
        )
    proposal_path = search / "proposal.json"
    if not proposal_path.exists():
        raise RuntimeError("analysis proposer did not write proposal.json")
    handoff = json.loads(proposal_path.read_text())
    provided_analysis = search / "provided_analysis"
    if provided_analysis.exists():
        shutil.rmtree(provided_analysis)
    implementation = run_proposer(
        search=search,
        api_key=optimizer_key,
        system_template=IMPLEMENTER_SYSTEM,
        task=(
            "Run implementation phase for iteration 1. The analysis role selected "
            f"candidate name {handoff.get('name')!r}, hypothesis "
            f"{handoff.get('hypothesis')!r}, and changes {handoff.get('changes')!r}. "
            "Read proposal.json, agents/baseline_agent.py, agents/baseline_scaffold.yaml, "
            "and parent_source/. Implement exactly that change, create candidate_agent.py, "
            "candidate_scaffold.yaml, optional candidate_assets/, and pending_eval.json; "
            "validate all candidate artifacts, then submit. "
            "Do not read trajectories; analysis is already complete."
        ),
        trajectory_path=search / "logs" / "implementation.traj.json",
        step_limit=30,
    )
    proposal, scaffold, agent_class, assets_dir = validate_candidate(search)
    audit = audit_proposer_inputs(search)
    if not audit["passed"]:
        raise RuntimeError("proposer referenced forbidden analysis inputs")
    candidate_audit = audit_candidate_artifact(search, instances)
    if not candidate_audit["passed"]:
        raise RuntimeError("candidate contains forbidden task/evaluator references")

    benchmark = benchmark_candidate(
        instances=selected,
        scaffold=scaffold,
        agent_class=agent_class,
        assets_dir=assets_dir,
        output_dir=run_dir / "candidate_benchmark",
        api_key=task_key,
        step_limit=5 if args.smoke else TASK_STEP_LIMIT,
        workers=args.benchmark_workers,
    )
    candidate = {
        "iteration": 1,
        "agent": proposal["name"],
        "avg_pass_rate": benchmark["avg_pass_rate"],
        "resolved": benchmark["resolved"],
        "total": benchmark["total"],
        "hypothesis": proposal["hypothesis"],
        "changes": proposal["changes"],
        "expected_efficiency": proposal["expected_efficiency"],
        "outcome": (
            f"{benchmark['resolved']}/{benchmark['total']} "
            f"({benchmark['avg_pass_rate']:.1%}, baseline +{benchmark['avg_pass_rate']:.1%})"
        ),
    }
    with (search / "evolution_summary.jsonl").open("a") as handle:
        handle.write(json.dumps(candidate, ensure_ascii=False) + "\n")
    frontier = json.loads((search / "frontier_val.json").read_text())
    if benchmark["avg_pass_rate"] > frontier["_best"]["avg_pass_rate"]:
        frontier["_best"] = {
            "agent": proposal["name"],
            "avg_pass_rate": benchmark["avg_pass_rate"],
            "agent_path": "agents/candidate_agent.py",
            "scaffold_path": "agents/candidate_scaffold.yaml",
            "assets_path": "agents/candidate_assets" if assets_dir else None,
        }
    write_json(search / "frontier_val.json", frontier)

    summary = {
        "baseline": "Meta-Harness",
        "task_model": TASK_MODEL,
        "task_step_limit": TASK_STEP_LIMIT,
        "optimizer_model": OPTIMIZER_MODEL,
        "ablation_group": args.ablation_group,
        "analysis_payloads_injected_at": (
            "analysis proposer workspace only" if inject_analysis else "none"
        ),
        "upstream_loop": "analyze -> implement -> validate -> benchmark -> frontier",
        "adaptation_protocol": "meta_full_corpus_v3",
        "analysis_coverage_policy": "all supplied records searchable; native proposer selects and deep-reads evidence within 40 steps",
        "adaptation_tasks": len(adaptation_instances),
        "adaptation_task_ids": [x["instance_id"] for x in adaptation_instances],
        "adaptation_seed": args.adaptation_seed,
        "analysis_workers": 1,
        "implementation_workers": 1,
        "benchmark_workers": args.benchmark_workers,
        "proposer_wrapper": "llm-sidecar GPT-5.5 replacement allowed by upstream README",
        "analysis_json_injected": inject_analysis,
        "analysis_proposer": analysis,
        "implementation_proposer": implementation,
        "proposal": proposal,
        "input_audit": audit,
        "candidate_audit": candidate_audit,
        "candidate_surface": {
            "python_agent": "arbitrary AgentHarness subclass",
            "scaffold": {
                "agent": list(AGENT_TEXT_FIELDS),
                "model": list(MODEL_TEXT_FIELDS),
            },
            "assets_mounted_at": "/workspace/meta_harness",
            "protected": list(PROTECTED_SURFACE),
        },
        "candidate_benchmark": benchmark,
        "frontier": frontier,
    }
    write_json(run_dir / "summary.json", summary)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
