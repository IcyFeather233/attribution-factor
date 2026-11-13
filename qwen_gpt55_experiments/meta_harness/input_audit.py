"""Distinguish quoted authorized evidence from forbidden proposer inputs."""
import copy
import json
import shlex
import re
from pathlib import Path

from qwen_gpt55_experiments.common.analysis_payloads import model_visible_payload

FORBIDDEN = ('trace_analysis', 'official_traces', 'gold_patch', 'failure_analysis.json')


def frozen_analysis_view(search):
    lines = []
    for trace in sorted((search / 'traces').glob('*.traj.json')):
        source = search.parent / 'provided_analysis' / (trace.name.removesuffix('.traj.json') + '.json')
        if not source.exists():
            return None
        data = model_visible_payload(json.loads(source.read_text()))
        if not all(key in data for key in ('failure_causing_wrong_step', 'failure_inevitable_step')):
            return None
        lines.append('\n### ' + source.name)
        for key in ('failure_causing_wrong_step', 'failure_inevitable_step'):
            value = data[key]
            lines.append(f"{key}: step={value.get('step')} role={value.get('role')} msg={value.get('message_index')}")
            for field in ('what_went_wrong', 'evidence', 'recoverability', 'why_inevitable_here'):
                if field in value:
                    lines.append(field + ': ' + str(value[field]))
    return '\n'.join(lines) + '\n'


def verified_view_read(search, command, expected):
    try:
        args = shlex.split(command)
    except ValueError:
        return False
    if not (len(args) == 4 and args[:2] == ['sed', '-n'] and re.fullmatch(r'\d+,\d+p', args[2]) and args[3] == 'pa_full.txt'):
        return False
    path = search / 'pa_full.txt'
    return expected is not None and path.is_file() and not path.is_symlink() and path.read_text() == expected


def derived_output_files(messages):
    """Find files the proposer generated from provided_analysis."""
    derived = set()
    for message in messages:
        if message.get("role") != "assistant":
            continue
        for action in message.get("extra", {}).get("actions", []):
            command = action.get("command", "")
            if "provided_analysis/" not in command or ">" not in command:
                continue
            first_line = command.splitlines()[0] if command else ""
            match = re.search(r">\s*(\S+)", first_line)
            if match:
                derived.add(match.group(1))
    return derived


def reads_derived_file(command, derived):
    return bool(derived) and any(path in command for path in derived)


def evidence_strings(value):
    if isinstance(value, str):
        if len(value) > 32 and any(token in value for token in FORBIDDEN):
            yield value
            yield json.dumps(value, ensure_ascii=False)[1:-1]
            yield json.dumps(value, ensure_ascii=True)[1:-1]
    elif isinstance(value, dict):
        for child in value.values():
            yield from evidence_strings(child)
    elif isinstance(value, list):
        for child in value:
            yield from evidence_strings(child)


def audit_proposer_inputs(search: Path) -> dict:
    # Use the host's frozen run inputs, never a proposer-edited workspace file.
    approved = set()
    for trace in (search / 'traces').glob('*.traj.json'):
        task_id = trace.name.removesuffix('.traj.json')
        source = search.parent / 'provided_analysis' / (task_id + '.json')
        if source.is_file():
            approved.update(evidence_strings(model_visible_payload(json.loads(source.read_text()))))
    approved = sorted(approved, key=len, reverse=True)
    expected_view = frozen_analysis_view(search)
    hits, quotations = [], []
    for trajectory in [search / 'logs/analysis.traj.json', search / 'logs/implementation.traj.json']:
        if not trajectory.exists():
            continue
        messages = json.loads(trajectory.read_text()).get('messages', [])
        derived = derived_output_files(messages)
        commands = {}
        for index, message in enumerate(messages):
            for action in message.get('extra', {}).get('actions', []):
                if message.get('role') == 'assistant' and action.get('tool_call_id'):
                    commands[action['tool_call_id']] = action.get('command', '')
            original = json.dumps(message, ensure_ascii=False)
            checked = copy.deepcopy(message)
            command = commands.get(message.get('tool_call_id'), '')
            eligible = (trajectory.name == 'analysis.traj.json' and message.get('role') == 'tool'
                        and ('provided_analysis/' in command
                             or verified_view_read(search, command, expected_view)
                             or reads_derived_file(command, derived))
                        and not any(t in command for t in FORBIDDEN))
            if eligible:
                def mask(text):
                    if isinstance(text, str):
                        for phrase in approved:
                            text = text.replace(phrase, '<verified frozen analysis quotation>')
                    return text
                checked['content'] = mask(checked.get('content', ''))
                if isinstance(checked.get('extra'), dict) and 'raw_output' in checked['extra']:
                    checked['extra']['raw_output'] = mask(checked['extra']['raw_output'])
            text = json.dumps(checked, ensure_ascii=False)
            for token in FORBIDDEN:
                location = {'trajectory': trajectory.name, 'message': index, 'token': token}
                if token in text:
                    hits.append(location)
                elif token in original:
                    quotations.append({**location, 'reason': 'exact quotation from frozen selected analysis input',
                                       'tool_call_id': message.get('tool_call_id')})
    result = {'passed': not hits, 'audit_version': 'frozen_evidence_quotes_v3',
              'forbidden_reference_hits': hits, 'authorized_quotation_hits': quotations}
    (search / 'input_audit.json').write_text(json.dumps(result, indent=2, ensure_ascii=False))
    return result
