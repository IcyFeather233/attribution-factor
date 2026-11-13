"""Resume an interrupted GEPA final benchmark without rerunning evolution."""

from __future__ import annotations

import argparse
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from attribution_experiments.common.analysis_payloads import GROUPS
from attribution_experiments.common.dataset import load_failed_instances
from attribution_experiments.common.model_config import TASK_STEP_LIMIT, task_api_key
from attribution_experiments.gepa.run_native import NativeEvaluator, RUN_ROOT
from attribution_experiments.gepa.protocol import require_completed_evolution


def completed_results(output_dir: Path, instance_ids: set[str]) -> dict[str, dict]:
    completed = {}
    for result_path in (output_dir / "metric_calls").glob("*/result.json"):
        try:
            result = json.loads(result_path.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        instance_id = result.get("instance_id")
        evaluation = result.get("evaluation")
        if (
            instance_id not in instance_ids
            or not isinstance(evaluation, dict)
            or "resolved" not in evaluation
            or "overdue-payment" in json.dumps(result)
        ):
            continue
        completed[instance_id] = {
            "instance_id": instance_id,
            "resolved": bool(evaluation["resolved"]),
            "status": evaluation.get("error")
            or ("resolved" if evaluation["resolved"] else "unresolved"),
            "resumed_from_completed_artifact": True,
        }
    return completed


def advance_metric_counter(evaluator: NativeEvaluator) -> None:
    call_ids = []
    for path in (evaluator.output_dir / "metric_calls").glob("*"):
        try:
            call_ids.append(int(path.name.split("_", 1)[0]))
        except ValueError:
            continue
    evaluator._counter = max(call_ids, default=-1) + 1


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-name", default="native_round1")
    parser.add_argument("--ablation-group", choices=GROUPS, required=True)
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be at least 1")

    run_dir = RUN_ROOT / args.ablation_group / args.run_name
    best_path = run_dir / "best_candidate_overlay.yaml"
    evolution_path = run_dir / "evolution_summary.json"
    if not best_path.is_file() or not evolution_path.is_file():
        raise FileNotFoundError("Cannot resume GEPA final without completed evolution artifacts")
    require_completed_evolution(json.loads(evolution_path.read_text()))

    instances = load_failed_instances()
    output_dir = run_dir / "final"
    evaluator = NativeEvaluator(
        output_dir=output_dir,
        api_key=task_api_key(),
        step_limit=TASK_STEP_LIMIT,
    )
    advance_metric_counter(evaluator)
    best_artifact = best_path.read_text()
    by_id = completed_results(output_dir, {item["instance_id"] for item in instances})
    pending = [item for item in instances if item["instance_id"] not in by_id]

    def rollout(instance: dict) -> dict:
        score, info = evaluator(best_artifact, instance)
        return {
            "instance_id": instance["instance_id"],
            "resolved": bool(score),
            "status": info["Feedback"]["Status"],
        }

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(rollout, instance): instance["instance_id"]
            for instance in pending
        }
        for future in as_completed(futures):
            by_id[futures[future]] = future.result()

    results = [by_id[item["instance_id"]] for item in instances]
    final_summary = {
        "resolved": sum(item["resolved"] for item in results),
        "total": len(results),
        "workers": args.workers,
        "resumed_completed_tasks": len(instances) - len(pending),
        "results": results,
    }
    (run_dir / "final_summary.json").write_text(
        json.dumps(final_summary, indent=2) + "\n"
    )
    evolution_summary = json.loads(evolution_path.read_text())
    print(json.dumps({"evolution": evolution_summary, "final": final_summary}, indent=2))


if __name__ == "__main__":
    main()
