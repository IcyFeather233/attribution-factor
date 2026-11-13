"""Resume an interrupted native Meta-Harness candidate benchmark."""

from __future__ import annotations

import argparse
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from qwen_gpt55_experiments.common.analysis_payloads import GROUPS, GROUP_BASELINE
from qwen_gpt55_experiments.common.model_config import (
    OPTIMIZER_MODEL,
    TASK_MODEL,
    TASK_STEP_LIMIT,
    task_api_key,
)

from qwen_gpt55_experiments.common.dataset import (
    load_failed_instances,
    select_repository_balanced_instances,
)
from qwen_gpt55_experiments.common.runner import run_task
from qwen_gpt55_experiments.common.result_validity import retryable_result, retryable_text
from qwen_gpt55_experiments.meta_harness.run_native import (
    AGENT_TEXT_FIELDS,
    MODEL_TEXT_FIELDS,
    PROTECTED_SURFACE,
    RUN_ROOT,
    UDOCKER_EXECUTABLE,
    UDOCKER_REPO,
    audit_candidate_artifact,
    audit_proposer_inputs,
    validate_candidate,
    write_json,
)


def proposer_result(path: Path) -> dict:
    trajectory = json.loads(path.read_text())
    info = trajectory["info"]
    return {
        "exit_status": info.get("exit_status", ""),
        "api_calls": info.get("model_stats", {}).get("api_calls", 0),
        "submission": info.get("submission", ""),
        "resumed_from_trajectory": True,
    }


def completed_result(output_dir: Path, instance_id: str) -> dict | None:
    task_dir = output_dir / instance_id
    result_path = task_dir / "result.json"
    if result_path.exists():
        try:
            raw_result = json.loads(result_path.read_text())
        except (json.JSONDecodeError, OSError):
            raw_result = None
        if isinstance(raw_result, dict) and retryable_result(raw_result):
            return None
    skip = task_dir / 'skip.json'
    if skip.exists():
        result = json.loads(skip.read_text())
        if result.get('instance_id') != instance_id or not result.get('skipped'):
            raise ValueError('Invalid explicit task skip')
        if retryable_text(result):
            return None
        return result
    report_path = task_dir / "evaluation" / "report.json"
    if not result_path.exists() or not report_path.exists():
        return None
    try:
        result = json.loads(result_path.read_text())
        report = json.loads(report_path.read_text())
    except (json.JSONDecodeError, OSError):
        return None
    evaluation = result.get("evaluation")
    if (
        result.get("instance_id") != instance_id
        or not isinstance(evaluation, dict)
        or not isinstance(evaluation.get('resolved'), bool)
        or not isinstance(report.get('resolved'), bool)
        or evaluation['resolved'] != report['resolved']
        or result.get('api_calls', 0) <= 0
        or result.get('exception')
    ):
        return None
    result["resumed_from_completed_artifact"] = True
    return result


def benchmark_remaining(
    *,
    instances: list[dict],
    scaffold: dict,
    agent_class,
    assets_dir: Path | None,
    output_dir: Path,
    api_key: str,
    workers: int,
    offline_only: bool = False,
) -> tuple[dict, int]:
    cached = {
        instance["instance_id"]: result
        for instance in instances
        if (result := completed_result(output_dir, instance["instance_id"])) is not None
    }
    pending = [item for item in instances if item["instance_id"] not in cached]
    if offline_only and pending:
        raise RuntimeError(f'Offline finalization refuses to execute {len(pending)} missing tasks')

    def one(instance: dict) -> dict:
        from .timed_task import run_isolated
        return run_isolated(instance, output_dir / instance['instance_id'])

    by_id = dict(cached)
    failed_tasks = []
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(one, item): item["instance_id"] for item in pending}
        for future in as_completed(futures):
            task_id = futures[future]
            try:
                by_id[task_id] = future.result()
            except Exception as exc:
                failed_tasks.append(task_id)
    if failed_tasks:
        raise RuntimeError(f'{len(failed_tasks)} benchmark tasks failed to execute; completed artifacts retained for retry: {failed_tasks}')

    results = []
    for instance in instances:
        run = by_id[instance["instance_id"]]
        evaluation = run.get("evaluation", {})
        results.append(
            {
                "instance_id": instance["instance_id"],
                "resolved": (False if run.get('count_as_failure') else None) if run.get('skipped') else bool(evaluation.get("resolved")),
                "status": run.get('exit_status') if run.get('skipped') else evaluation.get(
                    "error",
                    "resolved" if evaluation.get("resolved") else run.get("error", "unresolved"),
                ),
            }
        )
    benchmark = {
        "resolved": sum(item["resolved"] is True for item in results),
        "total": len(results),
        "exceptions_counted_as_failure": sum(bool(run.get('count_as_failure')) for run in by_id.values()),
        "skipped": sum(item['resolved'] is None for item in results),
        "score_is_incomplete": any(item['resolved'] is None for item in results),
        "avg_pass_rate": sum(item["resolved"] is True for item in results) / max(1, len(results)),
        "results": results,
    }
    return benchmark, len(cached)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-name", default="native_round1")
    parser.add_argument("--ablation-group", choices=GROUPS, required=True)
    parser.add_argument("--adaptation-tasks", type=int, default=32)
    parser.add_argument("--adaptation-seed", type=int, default=20260902)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument('--offline-only', action='store_true', help='Use existing results only; never execute missing tasks')
    args = parser.parse_args()
    if args.adaptation_tasks < 1:
        parser.error("--adaptation-tasks must be at least 1")
    if args.workers < 1:
        parser.error("--workers must be at least 1")
    api_key = task_api_key()

    run_dir = RUN_ROOT / args.ablation_group / args.run_name
    from qwen_gpt55_experiments.common.execution_protocol import record_execution_backend
    record_execution_backend(run_dir)
    search = run_dir / "search"
    instances = load_failed_instances()
    if args.adaptation_tasks > len(instances):
        parser.error(f"--adaptation-tasks cannot exceed {len(instances)}")
    adaptation_instances = select_repository_balanced_instances(
        instances, limit=args.adaptation_tasks, seed=args.adaptation_seed
    )
    trace_ids = {
        path.name.removesuffix(".traj.json")
        for path in (search / "traces").glob("*.traj.json")
    }
    expected_trace_ids = {item["instance_id"] for item in adaptation_instances}
    if trace_ids != expected_trace_ids:
        raise RuntimeError(
            "resume trace set does not match the requested adaptation sample"
        )
    proposal, scaffold, agent_class, assets_dir = validate_candidate(search)
    input_audit = audit_proposer_inputs(search)
    candidate_audit = audit_candidate_artifact(search, instances)
    if not input_audit["passed"] or not candidate_audit["passed"]:
        raise RuntimeError("cannot resume a candidate that fails an existing audit")

    benchmark, reused_tasks = benchmark_remaining(
        instances=instances,
        scaffold=scaffold,
        agent_class=agent_class,
        assets_dir=assets_dir,
        output_dir=run_dir / "candidate_benchmark",
        api_key=api_key,
        workers=args.workers,
        offline_only=args.offline_only,
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
    history_path = search / "evolution_summary.jsonl"
    history = [json.loads(line) for line in history_path.read_text().splitlines() if line.strip()]
    history = [item for item in history if item.get("iteration") != 1]
    history.append(candidate)
    history_path.write_text("".join(json.dumps(item, ensure_ascii=False) + "\n" for item in history))
    frontier = {
        "_best": {
            "agent": "baseline",
            "avg_pass_rate": 0.0,
            "total": len(instances),
            "resolved": 0,
            "source": "given original failed traces",
        }
    }
    if benchmark["avg_pass_rate"] > 0.0:
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
            "none" if args.ablation_group == GROUP_BASELINE
            else "analysis proposer workspace only"
        ),
        "upstream_loop": "analyze -> implement -> validate -> benchmark -> frontier",
        "adaptation_protocol": (json.loads((run_dir / 'optimization_budget.json').read_text())['protocol']
                                if (run_dir / 'optimization_budget.json').exists() else 'shared_repository_balanced_32_v1'),
        "analysis_coverage_policy": "supplied corpus searchable; native proposer selectively deep-reads evidence",
        "adaptation_tasks": len(adaptation_instances),
        "adaptation_task_ids": [item["instance_id"] for item in adaptation_instances],
        "adaptation_seed": args.adaptation_seed,
        "benchmark_workers": args.workers,
        "proposer_wrapper": "llm-sidecar GPT-5.5 replacement allowed by upstream README",
        "analysis_json_injected": args.ablation_group != GROUP_BASELINE,
        "analysis_proposer": proposer_result(search / "logs" / "analysis.traj.json"),
        "implementation_proposer": proposer_result(search / "logs" / "implementation.traj.json"),
        "proposal": proposal,
        "input_audit": input_audit,
        "candidate_audit": candidate_audit,
        "candidate_surface": {
            "python_agent": "arbitrary AgentHarness subclass",
            "scaffold": {"agent": list(AGENT_TEXT_FIELDS), "model": list(MODEL_TEXT_FIELDS)},
            "assets_mounted_at": "/workspace/meta_harness",
            "protected": list(PROTECTED_SURFACE),
        },
        "resume": {"reused_completed_tasks": reused_tasks},
        "candidate_benchmark": benchmark,
        "frontier": frontier,
    }
    write_json(run_dir / "summary.json", summary)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
