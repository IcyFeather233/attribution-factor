"""Versioned position hints resolved against the ORIGINAL trace, not a diagnosis.

Old payloads and running jobs remain untouched. The only supplied anchor fields
are step/message_index; excerpts are mechanically extracted from that position.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import sys
from pathlib import Path
import shlex

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    __package__ = 'qwen_gpt55_experiments.common'

from . import analysis_payloads as old

PROTOCOL = 'position_step_evidence_v1'
PAYLOAD_ROOT = old.WORKSPACE / 'qwen_gpt55_experiments/payloads_position_v1'
ANCHORS = ('failure_causing_wrong_step', 'failure_inevitable_step')
EXCERPT_CHARS = 8000
INSTRUCTIONS = (
    'These are position hints in the ORIGINAL historical failed trajectory, not diagnoses. '
    'step is one-based over assistant messages; message_index is zero-based over all messages. '
    'historical_step_evidence contains the selected assistant message and its following '
    'observations up to, but excluding, the next assistant message. It is resolved from '
    'the supplied indices, including perturbed indices. Do not map these numbers onto '
    'a current candidate rollout. Read the actual excerpt and perform your native diagnosis; '
    'a hinted position need not prove a defect. Do not invent missing evidence or copy '
    'task-specific fixes into the shared harness. Truncation is explicit; the reader command '
    'can retrieve the full selected step. No annotated role, error explanation or repair advice '
    'is supplied in the anchor. Flow conditions explicitly add the existing graph evidence.'
)


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def clip(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    marker = '\n...[mechanical excerpt truncated; use reader for full step]...\n'
    left = (limit - len(marker)) // 2
    return text[:left] + marker + text[-(limit - len(marker) - left):]


def resolve_step(trace: dict, anchor: dict, *, excerpt_chars=EXCERPT_CHARS) -> dict:
    if set(anchor) != {'step', 'message_index'}:
        raise ValueError('A position anchor must contain exactly step and message_index')
    if any(type(v) is not int for v in anchor.values()) or excerpt_chars < 200:
        raise ValueError('Integer indices and an excerpt budget >=200 are required')
    messages = trace['messages']
    indices = [i for i, m in enumerate(messages) if m.get('role') == 'assistant']
    step, index = anchor['step'], anchor['message_index']
    if not 1 <= step <= len(indices) or indices[step - 1] != index:
        raise ValueError('Step/message_index mismatch in original trajectory')
    end = indices[step] if step < len(indices) else len(messages)
    selected = []
    for offset in range(index, end):
        message = messages[offset]
        # Provider config, credentials and unrelated metadata are never copied.
        clean = {k: copy.deepcopy(message[k]) for k in
                 ('role', 'content', 'reasoning_content', 'tool_calls', 'tool_call_id') if k in message}
        extra = message.get('extra') or {}
        actions = extra.get('actions')
        if actions is not None:
            clean['actions'] = copy.deepcopy(actions)
        selected.append({'message_index': offset, 'message': clean})
    serialized = json.dumps(selected, ensure_ascii=False, indent=2)
    return {**anchor, 'message_indices': list(range(index, end)),
            'raw_excerpt': clip(serialized, excerpt_chars), 'truncated': len(serialized) > excerpt_chars,
            'full_chars': len(serialized), 'excerpt_chars_limit': excerpt_chars,
            'selected_messages_sha256': digest(serialized.encode())}


def convert_payload(payload: dict, trace: dict, trace_sha256: str, *, excerpt_chars=EXCERPT_CHARS) -> dict:
    source = payload['provided_information']
    info = {}
    resolved = {}
    iid = payload['task_id']
    if not iid or any(c not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_.-' for c in iid):
        raise ValueError('Invalid task ID')
    for key in ANCHORS:
        if key not in source:
            continue
        anchor = {field: source[key][field] for field in ('step', 'message_index')}
        info[key] = anchor
        resolved[key] = resolve_step(trace, anchor, excerpt_chars=excerpt_chars)
        resolved[key]['reader_command'] = (
            shlex.quote(sys.executable) + ' -B '
            + shlex.quote(str(Path(__file__).resolve())) + ' '
            f'--task-id {iid} --step {anchor["step"]} --message-index {anchor["message_index"]} '
            f'--trace-sha256 {trace_sha256} --full')
        context = source.get(key + '_context')
        if context is not None:
            if context['center_step'] != anchor['step']:
                raise ValueError('Flow center disagrees with position')
            info[key + '_context'] = copy.deepcopy(context)
    if info:
        info['position_evidence_protocol'] = PROTOCOL
        info['historical_trace'] = {'task_id': iid, 'sha256': trace_sha256,
                                    'runtime': 'original trace (see dataset provenance)'}
        info['interpretation'] = INSTRUCTIONS
        info['historical_step_evidence'] = resolved
    return {'schema_version': '3.0', 'evidence_protocol': PROTOCOL,
            'task_id': iid, 'provided_information': info}


def verify_payload(payload: dict, *, source_trace: dict | None = None) -> None:
    if payload.get('schema_version') != '3.0' or payload.get('evidence_protocol') != PROTOCOL:
        raise ValueError('Expected position-only protocol; legacy inputs cannot be resumed here')
    info = payload['provided_information']
    if not info:
        return
    keys = {k for k in ANCHORS if k in info}
    allowed = keys | {k + '_context' for k in keys} | {
        'position_evidence_protocol', 'historical_trace', 'interpretation', 'historical_step_evidence'}
    if not keys or set(info) - allowed or set(info['historical_step_evidence']) != keys:
        raise ValueError('Unexpected diagnostic field or missing resolved step')
    if info['position_evidence_protocol'] != PROTOCOL or info['historical_trace']['task_id'] != payload['task_id']:
        raise ValueError('Historical source identity mismatch')
    for key in keys:
        anchor, evidence = info[key], info['historical_step_evidence'][key]
        if set(anchor) != {'step', 'message_index'} or any(evidence[k] != anchor[k] for k in anchor):
            raise ValueError('Resolved evidence does not match supplied indices')
        if source_trace is not None:
            expected = resolve_step(source_trace, anchor, excerpt_chars=evidence['excerpt_chars_limit'])
            if any(evidence.get(k) != value for k, value in expected.items()):
                raise ValueError('Evidence is not a mechanical extraction of the indicated step')


def read_payloads(group: str, root: Path = PAYLOAD_ROOT) -> dict:
    manifest = json.loads((root / 'manifest.json').read_text())
    if manifest['evidence_protocol'] != PROTOCOL or group not in manifest['groups']:
        raise ValueError('Position protocol manifest mismatch')
    result = {}
    for iid in old.task_ids():
        path = root / group / (iid + '.json')
        raw = path.read_bytes()
        if digest(raw) != manifest['tasks'][iid][group]['sha256']:
            raise ValueError('Frozen payload changed: ' + str(path))
        payload = json.loads(raw)
        trace_bytes = old.trace_path(iid).read_bytes()
        info = payload['provided_information']
        if info and digest(trace_bytes) != info['historical_trace']['sha256']:
            raise ValueError('Original trace changed: ' + iid)
        verify_payload(payload, source_trace=json.loads(trace_bytes))
        result[iid] = payload
    return result


def generate(root: Path = PAYLOAD_ROOT) -> dict:
    if root.exists():
        raise FileExistsError('Refuse to replace frozen payloads: ' + str(root))
    root.mkdir(parents=True)
    manifest = {'evidence_protocol': PROTOCOL, 'schema_version': '3.0',
                'groups': list(old.GROUPS), 'excerpt_chars': EXCERPT_CHARS,
                'tasks': {}, 'source_sha256': {}}
    for group in old.GROUPS:
        (root / group).mkdir()
    for iid in old.task_ids():
        trace_path = old.trace_path(iid)
        trace_bytes = trace_path.read_bytes()
        trace = json.loads(trace_bytes)
        manifest['source_sha256'][str(trace_path)] = digest(trace_bytes)
        manifest['tasks'][iid] = {}
        for group in old.GROUPS:
            source = old.DEFAULT_PAYLOAD_ROOT / group / (iid + '.json')
            source_bytes = source.read_bytes()
            manifest['source_sha256'][str(source)] = digest(source_bytes)
            payload = convert_payload(json.loads(source_bytes), trace, digest(trace_bytes))
            verify_payload(payload, source_trace=trace)
            path = root / group / (iid + '.json')
            raw = (json.dumps(payload, ensure_ascii=False, indent=2) + '\n').encode()
            path.write_bytes(raw)
            manifest['tasks'][iid][group] = {'path': str(path.relative_to(root)), 'sha256': digest(raw)}
    (root / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--generate', action='store_true')
    parser.add_argument('--output', type=Path, default=PAYLOAD_ROOT)
    parser.add_argument('--task-id')
    parser.add_argument('--step', type=int)
    parser.add_argument('--message-index', type=int)
    parser.add_argument('--trace-sha256')
    parser.add_argument('--full', action='store_true')
    args = parser.parse_args()
    if args.generate:
        if not args.output.resolve().is_relative_to(old.WORKSPACE / 'qwen_gpt55_experiments'):
            parser.error('Output must be inside the persistent experiment package')
        manifest = generate(args.output.resolve())
        print(json.dumps({'root': str(args.output), 'tasks': len(manifest['tasks']), 'groups': len(manifest['groups'])}))
        return
    if args.task_id not in old.task_ids() or args.step is None or args.message_index is None:
        parser.error('Supply a registered task ID, step and message-index')
    raw = old.trace_path(args.task_id).read_bytes()
    if args.trace_sha256 and digest(raw) != args.trace_sha256:
        raise ValueError('Historical trace checksum mismatch')
    result = resolve_step(json.loads(raw), {'step': args.step, 'message_index': args.message_index},
                          excerpt_chars=len(raw) * 10 if args.full else EXCERPT_CHARS)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
