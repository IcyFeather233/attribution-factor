"""Paged frozen evidence with observable reads in the native analysis workspace."""
import json
from pathlib import Path

from qwen_gpt55_experiments.common.flow_evidence import ORIGIN, PROTOCOL, packet, records


def make_views(search: Path, payloads: dict[str, dict]) -> dict:
    views = search / "provided_analysis/flow_views"
    pages, references, index = {}, {}, ["# Historical flow evidence index", ORIGIN]
    for iid, payload in sorted(payloads.items()):
        evidence = packet(payload, iid)
        if not evidence:
            continue
        views.mkdir(parents=True, exist_ok=True)
        index.append(f"\nTask {iid}")
        for anchor, window in evidence["windows"].items():
            index.append(f"  {anchor}: historical step {window['center_step']}; role={window['anchor'].get('role')}")
            for kind in ("control_flow", "data_flow"):
                header = f"Task={iid}; anchor={anchor}; original center={window['center_step']}; kind={kind}\n"
                lines, refs = [], []
                def flush():
                    if not lines:
                        return
                    path = f"provided_analysis/flow_views/page-{len(pages):04d}.md"
                    content = header + "\n".join(lines) + "\n"
                    (search / path).write_text(content)
                    pages[path] = {"content": content, "refs": list(refs)}
                    for ref in refs:
                        references[ref] = {"task_id": iid, "anchor": anchor, "kind": kind, "page": path}
                    index.append(f"    {kind}: {path}")
                    lines.clear()
                    refs.clear()
                for item in window[kind].get("nodes", []) + window[kind].get("edges", []):
                    text = json.dumps(item, ensure_ascii=False, separators=(",", ":"))
                    if len(header) + len(text) > 6000:
                        raise ValueError("One flow record exceeds an untruncated observation page")
                    if sum(len(x) + 1 for x in lines) + len(text) + len(header) > 6000:
                        flush()
                    lines.append(text)
                    refs.append(item["evidence_ref"])
                flush()
    if not pages:
        return {}
    data = {"protocol": PROTOCOL, "index": "\n".join(index), "pages": pages, "references": references}
    (search / "flow_views_manifest.json").write_text(json.dumps(data, ensure_ascii=False, indent=2))
    return data


def citation_audit(proposal: dict, manifest: dict, read_pages: set[str]) -> dict:
    refs = proposal.get("flow_evidence_refs", [])
    known = manifest["references"]
    valid = isinstance(refs, list) and all(isinstance(ref, str) and ref in known for ref in refs)
    selected = [known[ref] for ref in refs] if valid else []
    unseen = sorted({entry["page"] for entry in selected} - read_pages)
    kinds = {entry["kind"] for entry in selected}
    tasks = {entry["task_id"] for entry in selected}
    edges = {known[ref]["kind"] for ref in (refs if valid else []) if "-edge-" in ref}
    return {"passed": bool(valid and not unseen and len(tasks) >= 2 and kinds == {"control_flow", "data_flow"}
                           and edges == {"control_flow", "data_flow"}),
            "valid_references": valid, "unread_cited_pages": unseen, "read_pages": sorted(read_pages),
            "cited_task_count": len(tasks), "cited_kinds": sorted(kinds), "cited_edge_kinds": sorted(edges),
            "flow_evidence_refs": refs}


def environment_class(base, manifest: dict):
    class FlowAnalysisEnvironment(base):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.read_pages = set()

        def execute(self, action, *args, **kwargs):
            from minisweagent.exceptions import Submitted
            try:
                result = super().execute(action, *args, **kwargs)
            except Submitted:
                path = Path(self.config.cwd) / "proposal.json"
                try:
                    audit = citation_audit(json.loads(path.read_text()), manifest, self.read_pages)
                except (OSError, ValueError):
                    audit = {"passed": False, "reason": "proposal.json missing or invalid"}
                (path.parent / "flow_read_audit.json").write_text(json.dumps(audit, indent=2))
                if not audit["passed"]:
                    return {"returncode": 2, "exception_info": "", "output":
                            "Flow evidence not yet verified. Read the complete cited pages using cat. "
                            "Add flow_evidence_refs to proposal.json citing exact control and data EDGE "
                            "references across at least two tasks supporting your recurring mechanism. "
                            "Do not invent references or change the hypothesis solely to satisfy the audit. "
                            + json.dumps(audit)}
                raise
            if result.get("returncode") == 0:
                output = str(result.get("output", ""))
                for path, page in manifest["pages"].items():
                    # Large outputs will be elided by the model observation template.
                    if len(output) <= 10000 and page["content"] in output:
                        self.read_pages.add(path)
            return result
    return FlowAnalysisEnvironment
