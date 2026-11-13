"""Lossless, source-labelled graph evidence for native optimizer phases only."""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

PROTOCOL = "native_flow_evidence_v2"
ANCHORS = ("failure_causing_wrong_step", "failure_inevitable_step")
ORIGIN = (
    "Historical evidence from the ORIGINAL failed trace. "
    "Historical step/message indices are NOT indices in the current candidate rollout. "
    "A current success does not invalidate the historical observation; distinguish the two. "
    "Compare observed control transitions and data production/consumption with the current "
    "harness before proposing a general change. Evidence is not a prescribed repair. "
    "Do not copy task-specific facts, node identifiers or answers into the task-agent scaffold."
)


def visible(payload: dict) -> dict:
    return payload.get("provided_information", payload)


def has_flow(payload: dict) -> bool:
    return any(isinstance(visible(payload).get(key + "_context"), dict) for key in ANCHORS)


def packet(payload: dict, task_id: str | None = None) -> dict:
    """Keep every supplied graph node/edge; add reference IDs, never repair advice."""
    if not has_flow(payload):
        return {}
    task_id = task_id or payload.get("task_id")
    if not task_id:
        raise ValueError("Flow evidence requires a task identity")
    source = visible(payload)
    result = {"protocol": PROTOCOL, "task_id": task_id, "origin": ORIGIN, "windows": {}}
    for anchor in ANCHORS:
        context = source.get(anchor + "_context")
        if not isinstance(context, dict):
            continue
        window = {"anchor": copy.deepcopy(source[anchor]),
                  "radius": context.get("radius", 3), "center_step": context["center_step"],
                  "window_steps": copy.deepcopy(context["window_steps"])}
        for kind in ("control_flow", "data_flow"):
            graph = copy.deepcopy(context[kind])
            for node in graph.get("nodes", []):
                node["evidence_ref"] = f"{task_id}/{anchor}/{node['node_id']}"
            for index, edge in enumerate(graph.get("edges", [])):
                edge["evidence_ref"] = f"{task_id}/{anchor}/{kind}-edge-{index}"
            window[kind] = graph
        result["windows"][anchor] = window
    return result


def records(evidence: dict) -> dict[str, dict]:
    result = {}
    for window in evidence.get("windows", {}).values():
        for kind in ("control_flow", "data_flow"):
            for collection in ("nodes", "edges"):
                for item in window[kind].get(collection, []):
                    result[item["evidence_ref"]] = item
    return result


def render(payload: dict, task_id: str | None = None, *, current_context: str = "") -> str:
    evidence = packet(payload, task_id)
    if not evidence:
        return ""
    lines = ["## Historical control/data-flow evidence", ORIGIN]
    if current_context:
        lines.append("Current rollout context (separate from historical evidence): " + current_context)
    for key, window in evidence["windows"].items():
        lines.append(f"### {key} | task={evidence['task_id']} | original step={window['center_step']}")
        lines.append("Anchor: " + json.dumps(window["anchor"], ensure_ascii=False))
        for kind in ("control_flow", "data_flow"):
            lines.append(kind + " (all supplied nodes and edges):")
            for item in window[kind].get("nodes", []) + window[kind].get("edges", []):
                lines.append(json.dumps(item, ensure_ascii=False, separators=(",", ":")))
    return "\n".join(lines)


def select_references(evidence: dict, refs: list[str]) -> dict:
    """Resolve citations plus edge endpoints; reject fabricated references."""
    index = records(evidence)
    if not isinstance(refs, list) or any(not isinstance(r, str) or r not in index for r in refs):
        raise ValueError("Unknown flow evidence reference")
    selected = {r: index[r] for r in refs}
    for ref, item in list(selected.items()):
        for field in ("source", "target"):
            if field in item:
                endpoint = ref.rsplit("/", 1)[0] + "/" + item[field]
                if endpoint in index:
                    selected[endpoint] = index[endpoint]
    return selected


def normalize_reference_spelling(evidence: dict, refs: list[str]) -> tuple[list[str], list[dict]]:
    """Repair only a known edge-token spelling; never infer a different source/node."""
    index = records(evidence)
    if not isinstance(refs, list):
        raise ValueError('Flow references must be a list')
    normalized, corrections = [], []
    for ref in refs:
        if not isinstance(ref, str):
            raise ValueError('Flow references must be strings')
        canonical = ref
        if ref not in index:
            prefix, separator, token = ref.rpartition('/')
            for original, replacement in [('control-flow-edge-', 'control_flow-edge-'), ('data-flow-edge-', 'data_flow-edge-')]:
                if separator and token.startswith(original):
                    canonical = prefix + '/' + replacement + token[len(original):]
                    break
        if canonical not in index:
            raise ValueError('Unknown flow evidence reference: ' + ref)
        normalized.append(canonical)
        if canonical != ref:
            corrections.append({'original': ref, 'canonical': canonical, 'reason': 'edge token separator spelling only'})
    return normalized, corrections


def handoff(payload: dict, task_id: str, text: str) -> dict:
    evidence = packet(payload, task_id)
    if not evidence:
        return {}
    refs = [ref for ref in records(evidence) if ref in text]
    # Missing citations stay explicit; do not invent attribution after the fact.
    return {"protocol": PROTOCOL, "origin": ORIGIN, "task_id": task_id,
            "citation_status": "cited" if refs else "no_explicit_graph_citations",
            "references": refs, "records": select_references(evidence, refs)}


def write_receipt(run_dir: Path, payloads: dict[str, dict], stage: str) -> None:
    selected = {iid: p for iid, p in payloads.items() if has_flow(p)}
    if not selected:
        return
    data = {"protocol": PROTOCOL, "stage": stage, "task_count": len(selected),
            "payload_sha256": {iid: hashlib.sha256(json.dumps(visible(p), sort_keys=True, ensure_ascii=False).encode()).hexdigest()
                               for iid, p in selected.items()}}
    (run_dir / "flow_delivery_protocol.json").write_text(json.dumps(data, indent=2) + "\n")
