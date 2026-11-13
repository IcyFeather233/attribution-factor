"""Run native SCOPE online guideline evolution around mini-swe-agent."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import shutil
import traceback
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from jinja2 import StrictUndefined, Template
from minisweagent.agents.default import DefaultAgent
from scope import SCOPEOptimizer
from scope.models import create_openai_model

from qwen_gpt55_experiments.common.analysis_payloads import (
    DEFAULT_PAYLOAD_ROOT,
    GROUPS,
    load_payloads,
    render_payload,
    snapshot_payloads,
)

from qwen_gpt55_experiments.common.dataset import (
    image_for_instance,
    load_failed_instances,
    select_repository_balanced_instances,
)
from qwen_gpt55_experiments.common.evaluator import evaluate_patch
from qwen_gpt55_experiments.common.model_config import (
    OPTIMIZER_BASE_URL,
    OPTIMIZER_MODEL,
    TASK_MODEL,
    TASK_STEP_LIMIT,
    optimizer_api_key,
    task_api_key,
)
from qwen_gpt55_experiments.common.runner import build_model, load_original_config, redact_secret, run_task
from qwen_gpt55_experiments.common.result_validity import retryable_result
from qwen_gpt55_experiments.common.execution_environment import ExecutionEnvironment as UdockerEnvironment


WORKSPACE = Path(__file__).resolve().parents[2]
RUN_ROOT = WORKSPACE / "qwen_gpt55_experiments" / "scope"
UDOCKER_REPO = WORKSPACE / "native_baseline_runs" / "udocker"
UDOCKER_EXECUTABLE = WORKSPACE / "native_baseline_runs" / ".venv" / "bin" / "udocker"
AGENT_NAME = "mini-swe-agent"
SCOPE_RULES_VARIABLE = "scope_strategic_rules"
SCOPE_RULES_TEMPLATE = "\n\n{{ scope_strategic_rules }}"


def build_scope_prompt_template(base_prompt: str) -> str:
    return base_prompt + SCOPE_RULES_TEMPLATE


def render_scope_prompt(template: str, strategic_rules: str) -> str:
    """Render once; braces inside strategic_rules remain literal text."""
    return Template(template, undefined=StrictUndefined).render(
        **{SCOPE_RULES_VARIABLE: strategic_rules}
    )


class ScopeAgent(DefaultAgent):
    def __init__(
        self,
        *args,
        optimizer: SCOPEOptimizer,
        task_id: str,
        analysis_payload: dict,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.optimizer = optimizer
        self.task_id = task_id
        self.analysis_payload = analysis_payload
        self.problem_statement = ""
        self.guideline_events: list[dict] = []

    def run(self, task: str = "", **kwargs) -> dict:
        self.problem_statement = task
        return super().run(task, **kwargs)

    def execute_actions(self, message: dict) -> list[dict]:
        actions = message.get("extra", {}).get("actions", [])
        outputs = [self.env.execute(action) for action in actions]
        observations = self.model.format_observation_messages(
            message,
            outputs,
            self.get_template_vars(),
        )
        self.add_messages(*observations)

        exception_text = "\n".join(output.get("exception_info", "") for output in outputs if output.get("exception_info"))
        error = RuntimeError(exception_text) if exception_text else None
        from qwen_gpt55_experiments.common.flow_evidence import has_flow, render as render_flow
        evidence = (
            render_flow(self.analysis_payload, self.task_id,
                        current_context=f'task={self.task_id}; current model call={self.n_calls}. '
                        'The separate model_output, tool_calls and observations below belong to this current rollout. '
                        'Relate them to historical graph dependencies by behavior/evidence, not equal step numbers. '
                        'Generate a guideline only when the current evidence supports a transferable mechanism.')
            if has_flow(self.analysis_payload) else render_payload(self.analysis_payload)
        )
        result = asyncio.run(
            self.optimizer.on_step_complete(
                agent_name=AGENT_NAME,
                agent_role="Software engineering agent with bash tool access",
                task=(
                    f"{self.problem_statement}\n\n{rendered}"
                    if (rendered := evidence)
                    else self.problem_statement
                ),
                model_output=message.get("content") or message.get("reasoning_content") or "",
                tool_calls=json.dumps(actions, default=str),
                observations=json.dumps(outputs, default=str),
                error=error,
                current_system_prompt=self.messages[0]["content"],
                task_id=self.task_id,
            )
        )
        if result:
            guideline, guideline_type = result
            self.messages[0]["content"] += f"\n\n## Learned Guideline:\n{guideline}"
            self.guideline_events.append(
                {
                    "step": self.n_calls,
                    "guideline": guideline,
                    "type": guideline_type,
                }
            )
        return observations


def make_optimizer(
    *,
    api_key: str,
    memory_dir: Path,
    quality_analysis_frequency: int = 1,
) -> SCOPEOptimizer:
    model = create_openai_model(
        model=OPTIMIZER_MODEL,
        api_key=api_key,
        base_url=OPTIMIZER_BASE_URL,
        temperature=1.0,
        max_tokens=4096,
    )
    return SCOPEOptimizer(
        synthesizer_model=model,
        exp_path=str(memory_dir),
        quality_analysis_frequency=quality_analysis_frequency,
    )


def run_adaptation_task(
    *,
    instance: dict,
    output_dir: Path,
    memory_dir: Path,
    optimizer_key: str,
    task_key: str,
    step_limit: int,
    quality_analysis_frequency: int,
    analysis_payload: dict,
) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    original = load_original_config()
    optimizer = make_optimizer(
        api_key=optimizer_key,
        memory_dir=memory_dir,
        quality_analysis_frequency=quality_analysis_frequency,
    )
    strategic_rules = optimizer.get_strategic_rules_for_agent(AGENT_NAME)
    agent_config = original["agent"]
    agent_config["system_template"] = build_scope_prompt_template(
        agent_config["system_template"]
    )
    render_scope_prompt(agent_config["system_template"], strategic_rules)
    agent_config["step_limit"] = step_limit
    agent_config["output_path"] = output_dir / "trajectory.json"

    env_config = original["environment"]
    env = UdockerEnvironment(
        image=image_for_instance(instance),
        repo=UDOCKER_REPO,
        executable=UDOCKER_EXECUTABLE,
        cwd=env_config["cwd"],
        env=env_config["env"],
        timeout=env_config["timeout"],
        interpreter=env_config["interpreter"],
        pull_timeout=3600,
        container_name=f"scope-{uuid.uuid4().hex[:12]}",
    )
    agent = ScopeAgent(
        model=build_model(api_key=task_key, temperature=1.0),
        env=env,
        optimizer=optimizer,
        task_id=instance["instance_id"],
        analysis_payload=analysis_payload,
        **agent_config,
    )
    agent.extra_template_vars[SCOPE_RULES_VARIABLE] = strategic_rules

    patch = ""
    from qwen_gpt55_experiments.common.execution_environment import backend_name
    result = {"instance_id": instance["instance_id"], 'execution_backend': backend_name()}
    try:
        info = agent.run(instance["problem_statement"])
        result["exit_status"] = info.get("exit_status", "")
        patch = info.get("submission", "") or ""
    except Exception as exc:
        result["exit_status"] = type(exc).__name__
        result["exception"] = str(exc)
        result["traceback"] = traceback.format_exc()
    finally:
        if agent.n_calls > 0 and not patch.strip():
            diff = env.execute({"command": "git -c core.fileMode=false diff"}, timeout=120)
            if diff["returncode"] == 0:
                patch = diff["output"]
        agent.save(
            output_dir / "trajectory.json",
            {
                "instance_id": instance["instance_id"],
                "info": {"submission": patch, "scope_guidelines": agent.guideline_events},
            },
        )
        redact_secret(output_dir / "trajectory.json", task_key)
        redact_secret(output_dir / "trajectory.json", optimizer_key)
        env.cleanup()

    (output_dir / "patch.diff").write_text(patch)
    result["api_calls"] = agent.n_calls
    result["guideline_count"] = len(agent.guideline_events)
    result["guidelines"] = agent.guideline_events
    result["patch_chars"] = len(patch)
    (output_dir / "result.json").write_text(json.dumps(result, indent=2))
    if result["api_calls"] == 0 or retryable_result(result):
        raise RuntimeError(
            "SCOPE adaptation produced a non-gradeable result for {}: {}: {}".format(
                instance["instance_id"],
                result.get("exit_status"),
                result.get("exception", ""),
            )
        )
    return result


def completed_adaptation_prefix(output_dir: Path, instances: list[dict]) -> list[dict]:
    completed = []
    for instance in instances:
        result_path = output_dir / instance["instance_id"] / "result.json"
        if not result_path.is_file():
            break
        try:
            result = json.loads(result_path.read_text())
        except (OSError, json.JSONDecodeError):
            break
        if (
            result.get("instance_id") != instance["instance_id"]
            or int(result.get("api_calls", 0)) <= 0
            or bool(result.get("exception"))
            or "overdue-payment" in json.dumps(result)
            or "token-limit" in json.dumps(result)
        ):
            break
        result["resumed_from_completed_artifact"] = True
        completed.append(result)
    return completed


def completed_final_results(output_dir: Path, instances: list[dict]) -> dict[str, dict]:
    completed = {}
    for instance in instances:
        task_dir = output_dir / instance["instance_id"]
        result_path = task_dir / "result.json"
        report_path = task_dir / "evaluation" / "report.json"
        if not result_path.is_file() or not report_path.is_file():
            continue
        try:
            result = json.loads(result_path.read_text())
            report = json.loads(report_path.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        if (
            result.get("instance_id") != instance["instance_id"]
            or not isinstance(result.get("evaluation"), dict)
            or "resolved" not in result["evaluation"]
            or "resolved" not in report
            or not isinstance(result["evaluation"].get("resolved"), bool)
            or report.get("resolved") != result["evaluation"]["resolved"]
            or retryable_result(result)
            or "overdue-payment" in json.dumps(result)
            or "token-limit" in json.dumps(result)
        ):
            continue
        result["resumed_from_completed_artifact"] = True
        completed[instance["instance_id"]] = result
    return completed


def load_strategic_rules(memory_dir: Path) -> dict:
    path = memory_dir / "strategic_memory" / "global_rules.json"
    if not path.is_file():
        return {}
    return json.loads(path.read_text())


def rule_texts(rules: dict) -> set[str]:
    return {
        str(rule.get("rule", "")).strip().lower()
        for domains in rules.values()
        for domain_rules in domains.values()
        for rule in domain_rules
        if str(rule.get("rule", "")).strip()
    }


def merge_batch_memories(
    *,
    memory_dir: Path,
    branches: list[tuple[str, Path]],
    optimizer_key: str,
) -> list[dict]:
    """Merge isolated task memories through SCOPE's native strategic store."""
    optimizer = make_optimizer(api_key=optimizer_key, memory_dir=memory_dir)
    known = rule_texts(load_strategic_rules(memory_dir))
    audit = []
    for instance_id, branch_memory in branches:
        branch_rules = load_strategic_rules(branch_memory)
        for agent_name in sorted(branch_rules):
            for domain in sorted(branch_rules[agent_name]):
                for rule in branch_rules[agent_name][domain]:
                    text = str(rule.get("rule", "")).strip()
                    normalized = text.lower()
                    if not text or normalized in known:
                        continue
                    promoted = asyncio.run(
                        optimizer.strategic_store.add_strategic_rule(
                            agent_name=agent_name,
                            rule_text=text,
                            rationale=str(rule.get("rationale", "")),
                            confidence=float(rule.get("confidence", 0)),
                            domain=domain,
                            source_task_id=rule.get("source_task_id") or instance_id,
                        )
                    )
                    known.add(normalized)
                    audit.append({
                        "instance_id": instance_id,
                        "agent_name": agent_name,
                        "domain": domain,
                        "rule": text,
                        "promoted": bool(promoted),
                    })
    return audit


def committed_batch_task_ids(run_dir: Path) -> set[str] | None:
    batch_root = run_dir / "adaptation_batches"
    if not batch_root.exists():
        return None
    committed = set()
    for path in sorted(batch_root.glob("batch_*/commit.json")):
        try:
            payload = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        committed.update(payload.get("instance_ids", []))
    return committed

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-name", default="native_round1")
    parser.add_argument("--ablation-group", choices=GROUPS, required=True)
    parser.add_argument("--payload-root", type=Path, default=DEFAULT_PAYLOAD_ROOT)
    parser.add_argument("--final-workers", type=int, default=1)
    parser.add_argument("--adaptation-tasks", type=int, default=32)
    parser.add_argument("--adaptation-seed", type=int, default=20260902)
    parser.add_argument("--adaptation-batch-size", type=int, default=1)
    parser.add_argument("--adaptation-workers", type=int, default=1)
    parser.add_argument("--quality-analysis-frequency", type=int, default=1)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.final_workers < 1:
        parser.error("--final-workers must be at least 1")
    for name in (
        "adaptation_tasks",
        "adaptation_batch_size",
        "adaptation_workers",
        "quality_analysis_frequency",
    ):
        if getattr(args, name) < 1:
            parser.error(f"--{name.replace('_', '-')} must be at least 1")
    optimizer_key = optimizer_api_key()
    task_key = task_api_key()

    run_dir = RUN_ROOT / args.ablation_group / args.run_name
    memory_dir = run_dir / "scope_memory"
    from qwen_gpt55_experiments.common.execution_protocol import record_execution_backend
    analysis_payloads = load_payloads(args.ablation_group, args.payload_root.resolve())
    from qwen_gpt55_experiments.common.flow_evidence import write_receipt
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
    final_instances = instances[:1] if args.smoke else instances
    step_limit = 3 if args.smoke else TASK_STEP_LIMIT
    from qwen_gpt55_experiments.common.optimization_budget import pin_budget, committed_resume_boundary
    if not args.resume and ((run_dir / 'adaptation').exists() or memory_dir.exists()):
        raise RuntimeError('Existing SCOPE progress requires an explicit compatible resume')
    pin_budget(run_dir, {'protocol': 'scope_task_coverage_v3',
                        'group': args.ablation_group,
                        'task_ids': [x['instance_id'] for x in adaptation_instances],
                        'batch_size': args.adaptation_batch_size,
                        'quality_analysis_frequency': args.quality_analysis_frequency,
                        'step_limit': step_limit},
               has_progress=(run_dir / 'adaptation').exists() or memory_dir.exists())
    record_execution_backend(run_dir)
    snapshot_payloads(run_dir, args.ablation_group, analysis_payloads)
    write_receipt(run_dir, analysis_payloads, 'native on_step_complete; historical graph and current observation comparison')

    adaptation_results = (
        completed_adaptation_prefix(run_dir / "adaptation", adaptation_instances)
        if args.resume
        else []
    )
    committed_ids = committed_batch_task_ids(run_dir)
    if args.resume and committed_ids is not None:
        adaptation_results = [
            result for result in adaptation_results
            if result["instance_id"] in committed_ids
        ]
    resumed_adaptation_tasks = len(adaptation_results)
    batch_size = 1 if args.smoke else args.adaptation_batch_size
    adaptation_workers = 1 if args.smoke else args.adaptation_workers
    if batch_size > 1:
        resumed_adaptation_tasks = committed_resume_boundary(resumed_adaptation_tasks, len(adaptation_instances), batch_size)
        adaptation_results = adaptation_results[:resumed_adaptation_tasks]

    for batch_start in range(resumed_adaptation_tasks, len(adaptation_instances), batch_size):
        batch = adaptation_instances[batch_start : batch_start + batch_size]
        batch_index = batch_start // batch_size
        batch_dir = run_dir / "adaptation_batches" / f"batch_{batch_index:04d}"
        commit_path = batch_dir / "commit.json"
        if batch_dir.exists() and not commit_path.is_file() and not args.resume:
            shutil.rmtree(batch_dir)
        batch_dir.mkdir(parents=True, exist_ok=True)

        branches = []
        branch_by_id = {}
        restored = {}
        for instance in batch:
            instance_id = instance["instance_id"]
            branch_memory = batch_dir / instance_id / "scope_memory"
            cached = completed_adaptation_prefix(run_dir / 'adaptation', [instance]) if args.resume else []
            if cached and branch_memory.is_dir():
                restored[instance_id] = cached[0]
            elif branch_memory.exists():
                # Only incomplete branch state is reset; completed branches keep learned memory.
                import time
                branch_memory.rename(branch_memory.with_name('scope_memory.interrupted.' + str(time.time_ns())))
            if instance_id in restored:
                pass
            elif memory_dir.exists():
                shutil.copytree(memory_dir, branch_memory)
            else:
                branch_memory.mkdir(parents=True)
            branches.append((instance_id, branch_memory))
            branch_by_id[instance_id] = branch_memory

        by_id = dict(restored)
        with ThreadPoolExecutor(
            max_workers=min(adaptation_workers, len(batch))
        ) as executor:
            futures = {
                executor.submit(
                    run_adaptation_task,
                    instance=instance,
                    output_dir=run_dir / "adaptation" / instance["instance_id"],
                    memory_dir=branch_by_id[instance["instance_id"]],
                    optimizer_key=optimizer_key,
                    task_key=task_key,
                    step_limit=step_limit,
                    quality_analysis_frequency=args.quality_analysis_frequency,
                    analysis_payload=analysis_payloads[instance["instance_id"]],
                ): instance["instance_id"]
                for instance in batch if instance['instance_id'] not in restored
            }
            for future in as_completed(futures):
                by_id[futures[future]] = future.result()

        merge_audit = merge_batch_memories(
            memory_dir=memory_dir,
            branches=branches,
            optimizer_key=optimizer_key,
        )
        adaptation_results.extend(
            by_id[instance["instance_id"]] for instance in batch
        )
        temporary = commit_path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps({
            "batch_index": batch_index,
            "instance_ids": [instance["instance_id"] for instance in batch],
            "workers": min(adaptation_workers, len(batch)),
            "memory_merge": merge_audit,
        }, indent=2))
        temporary.replace(commit_path)

    frozen_optimizer = make_optimizer(api_key=optimizer_key, memory_dir=memory_dir)
    strategic_rules = frozen_optimizer.get_strategic_rules_for_agent(AGENT_NAME)
    (run_dir / "frozen_strategic_rules.txt").write_text(strategic_rules)
    base_prompt = load_original_config()["agent"]["system_template"]
    frozen_prompt_template = build_scope_prompt_template(base_prompt)
    frozen_prompt = render_scope_prompt(frozen_prompt_template, strategic_rules)
    (run_dir / "frozen_system_prompt.txt").write_text(frozen_prompt)

    def final_rollout(instance: dict) -> dict:
        result = run_task(
            instance=instance,
            output_dir=run_dir / "final" / instance["instance_id"],
            udocker_repo=UDOCKER_REPO,
            udocker_executable=UDOCKER_EXECUTABLE,
            api_key=task_key,
            system_template=frozen_prompt_template,
            template_vars={SCOPE_RULES_VARIABLE: strategic_rules},
            step_limit=step_limit,
        )
        if result.get("api_calls", 0) == 0 or retryable_result(result):
            raise RuntimeError(
                "SCOPE final produced a non-gradeable result for {}: {}: {}".format(
                    instance["instance_id"],
                    result.get("exit_status"),
                    result.get("exception", ""),
                )
            )
        return result

    final_results_by_id = (
        completed_final_results(run_dir / "final", final_instances)
        if args.resume
        else {}
    )
    resumed_final_tasks = len(final_results_by_id)
    pending_final_instances = [
        item for item in final_instances if item["instance_id"] not in final_results_by_id
    ]
    workers = args.final_workers
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(final_rollout, instance): instance["instance_id"]
            for instance in pending_final_instances
        }
        for future in as_completed(futures):
            task_id = futures[future]
            final_results_by_id[task_id] = future.result()

    final_results = []
    for instance in final_instances:
        task_id = instance["instance_id"]
        run = final_results_by_id[task_id]
        final_results.append(
            {
                "instance_id": task_id,
                "resolved": bool(run.get("evaluation", {}).get("resolved")),
                "status": run.get("evaluation", {}).get("error", "resolved" if run.get("evaluation", {}).get("resolved") else "unresolved"),
            }
        )

    summary = {
        "ablation_group": args.ablation_group,
        "task_model": TASK_MODEL,
        "task_step_limit": TASK_STEP_LIMIT,
        "optimizer_model": OPTIMIZER_MODEL,
        "analysis_json_injected": args.ablation_group != "baseline",
        "analysis_payloads_injected_at": (
            "none" if args.ablation_group == "baseline"
            else "SCOPE on_step_complete task context only"
        ),
        "adaptation_tasks": len(adaptation_results),
        "adaptation_task_ids": [
            instance["instance_id"] for instance in adaptation_instances
        ],
        "adaptation_seed": args.adaptation_seed,
        "adaptation_protocol": "scope_task_coverage_v3",
        "adaptation_passes": 1,
        "unique_adaptation_tasks_completed": len({x['instance_id'] for x in adaptation_results}),
        "adaptation_batch_size": batch_size,
        "adaptation_workers": adaptation_workers,
        "quality_analysis_frequency": args.quality_analysis_frequency,
        "resumed_adaptation_tasks": resumed_adaptation_tasks,
        "final_workers": workers,
        "resumed_final_tasks": resumed_final_tasks,
        "adaptation_guidelines": sum(x["guideline_count"] for x in adaptation_results),
        "strategic_rule_chars": len(strategic_rules),
        "resolved": sum(item["resolved"] for item in final_results),
        "total": len(final_results),
        "results": final_results,
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
