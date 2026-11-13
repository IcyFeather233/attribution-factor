"""Build and load frozen attribution payloads for 148 confirmed task model failures."""

from __future__ import annotations

import copy
import hashlib
import json
import os
import random
from pathlib import Path
from typing import Any


WORKSPACE = Path(__file__).resolve().parents[2]
ANALYSIS_ROOT = Path(os.environ.get("ATTRIBUTION_ANALYSIS_ROOT", WORKSPACE / "agent_runtime" / "trace_analysis"))
GRAPH_ANALYSIS_ROOT = (
    Path(os.environ.get("ATTRIBUTION_GRAPH_ROOT", WORKSPACE / "agent_runtime" / "graph_context_analysis" / "v1" / "results"))
)
FAILED_IDS = Path(os.environ.get("ATTRIBUTION_TASK_IDS", WORKSPACE / "attribution_experiments" / "task_ids.txt"))
DEFAULT_PAYLOAD_ROOT = Path(os.environ.get("ATTRIBUTION_PAYLOAD_ROOT", WORKSPACE / "attribution_experiments" / "payloads"))
DEFAULT_RANDOM_SEED = 20260829
GRAPH_RADIUS = 3

GROUP_BASELINE = "baseline"
GROUP_1_1_CAUSING = "group1_1_causing"
GROUP_1_2_INEVITABLE = "group1_2_inevitable"
GROUP_1_3_BOTH = "group1_3_both"
GROUP_2_1_CAUSING_SHIFTED = "group2_1_causing_shifted"
GROUP_2_2_INEVITABLE_SHIFTED = "group2_2_inevitable_shifted"
GROUP_2_3_BOTH_SHIFTED = "group2_3_both_shifted"
GROUP_3_1_CAUSING_FLOW_CONTEXT = "group3_1_causing_flow_context"
GROUP_3_2_INEVITABLE_FLOW_CONTEXT = "group3_2_inevitable_flow_context"
GROUP_3_3_BOTH_FLOW_CONTEXT = "group3_3_both_flow_context"
GROUPS = (
    GROUP_BASELINE,
    GROUP_1_1_CAUSING,
    GROUP_1_2_INEVITABLE,
    GROUP_1_3_BOTH,
    GROUP_2_1_CAUSING_SHIFTED,
    GROUP_2_2_INEVITABLE_SHIFTED,
    GROUP_2_3_BOTH_SHIFTED,
    GROUP_3_1_CAUSING_FLOW_CONTEXT,
    GROUP_3_2_INEVITABLE_FLOW_CONTEXT,
    GROUP_3_3_BOTH_FLOW_CONTEXT,
)


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def task_ids() -> list[str]:
    return [line.strip() for line in FAILED_IDS.read_text().splitlines() if line.strip()]


def analysis_path(task_id: str) -> Path:
    return ANALYSIS_ROOT / f"{task_id}.analysis.json"


def trace_path(task_id: str) -> Path:
    analysis = _read_json(analysis_path(task_id))
    path = Path(analysis["trace"]["path"])
    return path if path.is_absolute() else WORKSPACE / path


def graph_analysis_path(task_id: str) -> Path:
    return GRAPH_ANALYSIS_ROOT / f"{task_id}.graph.json"


def _semantic_graph_context(
    graph: dict[str, Any],
    task_id: str,
    anchor_key: str,
    anchor: dict[str, Any],
) -> dict[str, Any]:
    if graph.get("schema_version") != "1.0":
        raise ValueError(f"Expected graph-analysis schema 1.0 for {task_id}")
    if graph.get("task_id") != task_id:
        raise ValueError(f"Graph-analysis task mismatch for {task_id}")
    if not graph.get("canonical_scaffold_version"):
        raise ValueError(f"Missing canonical scaffold version for {task_id}")
    window = copy.deepcopy(graph.get("windows", {}).get(anchor_key))
    if not isinstance(window, dict):
        raise ValueError(f"Missing graph window {anchor_key} for {task_id}")
    center = int(anchor["step"])
    if window.get("center_step") != center:
        raise ValueError(f"Graph center mismatch for {task_id}/{anchor_key}")
    steps = window.get("window_steps")
    if (
        not isinstance(steps, list)
        or not steps
        or center not in steps
        or steps != list(range(steps[0], steps[-1] + 1))
        or any(abs(step - center) > GRAPH_RADIUS for step in steps)
    ):
        raise ValueError(f"Invalid graph radius window for {task_id}/{anchor_key}")
    return {
        "radius": GRAPH_RADIUS,
        "canonical_scaffold_version": graph["canonical_scaffold_version"],
        **window,
    }


def _compact(text: Any, limit: int) -> str:
    value = str(text or "").strip()
    if len(value) <= limit:
        return value
    head = limit // 2
    tail = limit - head
    return f"{value[:head]}\n...[{len(value) - limit} chars omitted]...\n{value[-tail:]}"


def _assistant_steps(trace: dict[str, Any]) -> list[dict[str, Any]]:
    messages = trace.get("messages", [])
    assistant_indexes = [
        index for index, message in enumerate(messages) if message.get("role") == "assistant"
    ]
    records = []
    for offset, message_index in enumerate(assistant_indexes):
        next_index = (
            assistant_indexes[offset + 1]
            if offset + 1 < len(assistant_indexes)
            else len(messages)
        )
        previous_index = assistant_indexes[offset - 1] if offset else 1
        message = messages[message_index]
        actions = message.get("extra", {}).get("actions", []) or []
        preceding = [
            item
            for item in messages[previous_index + 1 : message_index]
            if item.get("role") in {"tool", "user"}
        ]
        following = [
            item
            for item in messages[message_index + 1 : next_index]
            if item.get("role") in {"tool", "user"}
        ]
        reasoning = message.get("reasoning_content") or message.get("content") or ""
        commands = [_compact(action.get("command", ""), 500) for action in actions[:4]]
        records.append(
            {
                "step": offset + 1,
                "message_index": message_index,
                "reasoning": _compact(reasoning, 700),
                "commands": commands,
                "input_observations": [
                    _compact(item.get("content", ""), 500) for item in preceding[-3:]
                ],
                "output_observations": [
                    _compact(item.get("content", ""), 500) for item in following[:4]
                ],
            }
        )
    return records


def _step_lookup(steps: list[dict[str, Any]]) -> dict[int, dict[str, Any]]:
    return {item["step"]: item for item in steps}


def _shift_delta(
    task_id: str, label: str, center: int, max_step: int, seed: int
) -> int:
    candidates = [
        delta for delta in (-3, -2, -1, 1, 2, 3)
        if 1 <= center + delta <= max_step
    ]
    if not candidates:
        raise ValueError(
            f"No valid shift for {task_id}/{label}: center={center}, max={max_step}"
        )
    digest = hashlib.sha256(f"{seed}:{task_id}:{label}".encode()).digest()
    rng = random.Random(int.from_bytes(digest[:8], "big"))
    return rng.choice(candidates)


def _shift_anchor(
    task_id: str,
    label: str,
    anchor: dict[str, Any],
    records: list[dict[str, Any]],
    seed: int,
) -> tuple[dict[str, Any], int]:
    lookup = _step_lookup(records)
    delta = _shift_delta(task_id, label, int(anchor["step"]), len(records), seed)
    shifted = copy.deepcopy(anchor)
    shifted_step = int(anchor["step"]) + delta
    shifted["step"] = shifted_step
    shifted["message_index"] = lookup[shifted_step]["message_index"]
    return shifted, delta


def _window_steps(centers: dict[str, dict[str, Any] | None], max_step: int) -> list[int]:
    selected: set[int] = set()
    for anchor in centers.values():
        if anchor is None:
            continue
        center = int(anchor["step"])
        selected.update(range(max(1, center - GRAPH_RADIUS), min(max_step, center + GRAPH_RADIUS) + 1))
    return sorted(selected)


def _center_labels(step: int, centers: dict[str, dict[str, Any] | None]) -> list[str]:
    return [
        label
        for label, anchor in centers.items()
        if anchor is not None and int(anchor["step"]) == step
    ]


def _control_flow_context(
    centers: dict[str, dict[str, Any] | None],
    records: list[dict[str, Any]],
) -> dict[str, Any]:
    lookup = _step_lookup(records)
    selected = _window_steps(centers, len(records))
    nodes = []
    for step in selected:
        item = lookup[step]
        nodes.append(
            {
                "node_id": f"control-step-{step}",
                "step": step,
                "message_index": item["message_index"],
                "is_center_for": _center_labels(step, centers),
                "decision": item["reasoning"],
                "actions": item["commands"],
            }
        )
    selected_set = set(selected)
    edges = [
        {"source": f"control-step-{step}", "target": f"control-step-{step + 1}", "type": "next"}
        for step in selected
        if step + 1 in selected_set
    ]
    return {"radius": GRAPH_RADIUS, "nodes": nodes, "edges": edges}


def _data_flow_context(
    centers: dict[str, dict[str, Any] | None],
    records: list[dict[str, Any]],
) -> dict[str, Any]:
    lookup = _step_lookup(records)
    selected = _window_steps(centers, len(records))
    nodes = []
    for step in selected:
        item = lookup[step]
        nodes.append(
            {
                "node_id": f"data-step-{step}",
                "step": step,
                "message_index": item["message_index"],
                "is_center_for": _center_labels(step, centers),
                "inputs": item["input_observations"],
                "transforms": item["commands"],
                "outputs": item["output_observations"],
            }
        )
    selected_set = set(selected)
    edges = [
        {
            "source": f"data-step-{step}",
            "target": f"data-step-{step + 1}",
            "type": "observation_to_next_decision",
        }
        for step in selected
        if step + 1 in selected_set
    ]
    return {"radius": GRAPH_RADIUS, "nodes": nodes, "edges": edges}


def build_task_payloads(task_id: str, seed: int = DEFAULT_RANDOM_SEED) -> dict[str, dict[str, Any]]:
    analysis = _read_json(analysis_path(task_id))
    if analysis.get("schema_version") != "2.0":
        raise ValueError(f"Expected trace-analysis schema 2.0 for {task_id}")
    if analysis.get("classification") != "confirmed_agent_failure":
        raise ValueError(f"Task is not a confirmed agent failure: {task_id}")
    attribution = analysis["step_attribution"]
    causing = copy.deepcopy(attribution.get("failure_causing_wrong_step"))
    inevitable = copy.deepcopy(attribution.get("failure_inevitable_step"))
    if causing is None or inevitable is None:
        raise ValueError(f"Confirmed task lacks a required attribution anchor: {task_id}")
    # Ablation payloads expose only the step localization, not the recoverability
    # or irreversibility conclusions derived from offline evidence.
    causing.pop("recoverability", None)
    inevitable.pop("why_inevitable_here", None)
    graph_path = graph_analysis_path(task_id)
    if not graph_path.is_file():
        raise FileNotFoundError(f"Missing validated graph analysis: {graph_path}")
    graph = _read_json(graph_path)
    causing_context = _semantic_graph_context(
        graph, task_id, "failure_causing_wrong_step", causing
    )
    inevitable_context = _semantic_graph_context(
        graph, task_id, "failure_inevitable_step", inevitable
    )
    graph_sha256 = hashlib.sha256(graph_path.read_bytes()).hexdigest()
    records = _assistant_steps(_read_json(trace_path(task_id)))
    shifted_causing, causing_delta = _shift_anchor(
        task_id, "failure_causing_wrong_step", causing, records, seed
    )
    shifted_inevitable, inevitable_delta = _shift_anchor(
        task_id, "failure_inevitable_step", inevitable, records, seed
    )

    common = {"schema_version": "2.0", "task_id": task_id}
    return {
        GROUP_BASELINE: {**common, "provided_information": {}},
        GROUP_1_1_CAUSING: {
            **common,
            "provided_information": {"failure_causing_wrong_step": causing},
        },
        GROUP_1_2_INEVITABLE: {
            **common,
            "provided_information": {"failure_inevitable_step": inevitable},
        },
        GROUP_1_3_BOTH: {
            **common,
            "provided_information": {
                "failure_causing_wrong_step": causing,
                "failure_inevitable_step": inevitable,
            },
        },
        GROUP_2_1_CAUSING_SHIFTED: {
            **common,
            "provided_information": {"failure_causing_wrong_step": shifted_causing},
            "generation_metadata": {
                "random_seed": seed, "causing_delta": causing_delta
            },
        },
        GROUP_2_2_INEVITABLE_SHIFTED: {
            **common,
            "provided_information": {"failure_inevitable_step": shifted_inevitable},
            "generation_metadata": {
                "random_seed": seed, "inevitable_delta": inevitable_delta
            },
        },
        GROUP_2_3_BOTH_SHIFTED: {
            **common,
            "provided_information": {
                "failure_causing_wrong_step": shifted_causing,
                "failure_inevitable_step": shifted_inevitable,
            },
            "generation_metadata": {
                "random_seed": seed,
                "causing_delta": causing_delta,
                "inevitable_delta": inevitable_delta,
                "independent_field_shifts": True,
            },
        },
        GROUP_3_1_CAUSING_FLOW_CONTEXT: {
            **common,
            "provided_information": {
                "failure_causing_wrong_step": causing,
                "failure_causing_wrong_step_context": causing_context,
            },
            "generation_metadata": {
                "graph_analysis_schema": "1.0",
                "graph_analysis_sha256": graph_sha256,
            },
        },
        GROUP_3_2_INEVITABLE_FLOW_CONTEXT: {
            **common,
            "provided_information": {
                "failure_inevitable_step": inevitable,
                "failure_inevitable_step_context": inevitable_context,
            },
            "generation_metadata": {
                "graph_analysis_schema": "1.0",
                "graph_analysis_sha256": graph_sha256,
            },
        },
        GROUP_3_3_BOTH_FLOW_CONTEXT: {
            **common,
            "provided_information": {
                "failure_causing_wrong_step": causing,
                "failure_causing_wrong_step_context": causing_context,
                "failure_inevitable_step": inevitable,
                "failure_inevitable_step_context": inevitable_context,
            },
            "generation_metadata": {
                "graph_analysis_schema": "1.0",
                "graph_analysis_sha256": graph_sha256,
                "independent_anchor_windows": True,
            },
        },
    }


def generate_payloads(
    output_root: Path = DEFAULT_PAYLOAD_ROOT,
    seed: int = DEFAULT_RANDOM_SEED,
) -> dict[str, Any]:
    ids = task_ids()
    manifest: dict[str, Any] = {
        "schema_version": "2.0",
        "source_analysis_schema": "2.0",
        "random_seed": seed,
        "groups": list(GROUPS),
        "tasks": {},
    }
    for group in GROUPS:
        (output_root / group).mkdir(parents=True, exist_ok=True)
    for task_id in ids:
        payloads = build_task_payloads(task_id, seed)
        task_manifest = {}
        for group, payload in payloads.items():
            path = output_root / group / f"{task_id}.json"
            path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
            task_manifest[group] = {
                "path": str(path.relative_to(output_root)),
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
            if payload.get("generation_metadata"):
                task_manifest[group]["generation_metadata"] = payload["generation_metadata"]
        manifest["tasks"][task_id] = task_manifest
    manifest_path = output_root / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
    return manifest


def load_payloads(group: str, payload_root: Path = DEFAULT_PAYLOAD_ROOT) -> dict[str, dict[str, Any]]:
    if group not in GROUPS:
        raise ValueError(f"Unknown ablation group {group!r}; expected one of {GROUPS}")
    payloads = {}
    for task_id in task_ids():
        path = payload_root / group / f"{task_id}.json"
        if not path.is_file():
            raise FileNotFoundError(f"Missing frozen ablation payload: {path}")
        payloads[task_id] = _read_json(path)
    return payloads


def model_visible_payload(payload: dict[str, Any]) -> dict[str, Any]:
    """Return only the requested information, excluding experiment metadata."""
    return copy.deepcopy(payload["provided_information"])


def render_payload(payload: dict[str, Any]) -> str:
    visible = model_visible_payload(payload)
    if not visible:
        return ""
    return (
        "## Provided Failure-Localization Information\n"
        "Treat this as additional analysis evidence for the baseline's repair/evolution phase.\n"
        "```json\n"
        + json.dumps(visible, indent=2, ensure_ascii=False)
        + "\n```"
    )


def snapshot_payloads(
    run_dir: Path,
    group: str,
    payloads: dict[str, dict[str, Any]],
) -> Path:
    """Freeze the exact per-task payloads consumed by one baseline run."""
    target = run_dir / "provided_analysis"
    target.mkdir(parents=True, exist_ok=True)
    hashes = {}
    for task_id, payload in payloads.items():
        path = target / f"{task_id}.json"
        path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
        hashes[task_id] = hashlib.sha256(path.read_bytes()).hexdigest()
    manifest = {
        "schema_version": "2.0",
        "experiment_group": group,
        "model_visible_information_empty": group == GROUP_BASELINE,
        "task_count": len(payloads),
        "payload_sha256": hashes,
    }
    (target / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n"
    )
    return target
