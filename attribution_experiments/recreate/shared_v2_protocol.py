"""Pure-data contract between the legacy ReCreate optimizer and the v2 task worker."""
from copy import deepcopy
import hashlib
import json
import os
import re
from pathlib import Path

from jinja2 import Environment, StrictUndefined
import yaml

ROOT = Path(__file__).resolve().parents[2]
V2_SOURCE = ROOT / 'agent_runtime/mini-swe-agent-v2.4.6/src'
REFERENCE_TRACE = Path(os.environ.get('ATTRIBUTION_REFERENCE_TRACE', ROOT / 'data/reference.traj.json'))
PROTOCOL = 'recreate_shared_mini_v246_v1'
RUN_ROOT = ROOT / 'attribution_experiments/recreate_shared_v246'
TEXT_FIELDS = ('system_template', 'instance_template', 'format_error_template',
               'action_observation_template', 'multimodal_regex', 'memory_template')


def original_config():
    return deepcopy(json.loads(REFERENCE_TRACE.read_text())['info']['config'])


def seed_scaffold():
    original = original_config()
    return {
        'system_template': original['agent']['system_template'],
        'instance_template': original['agent']['instance_template'],
        'format_error_template': original['model']['format_error_template'],
        # Flat alias required by native ReCreate scaffold tooling.
        'action_observation_template': original['model']['observation_template'],
        'multimodal_regex': original['model'].get('multimodal_regex', ''),
        'memory_template': '', 'step_limit': 250, 'cost_limit': 0,
    }


def validate_scaffold(scaffold):
    if not isinstance(scaffold, dict):
        raise ValueError('Shared-v2 scaffold must be a mapping')
    if set(scaffold) - set(TEXT_FIELDS) - {'step_limit', 'cost_limit'}:
        raise ValueError('Unsupported shared-v2 scaffold fields: ' + str(sorted(set(scaffold) - set(TEXT_FIELDS) - {'step_limit', 'cost_limit'})))
    required = set(TEXT_FIELDS) - {'memory_template', 'multimodal_regex'}
    if required - set(scaffold):
        raise ValueError('Missing shared-v2 text fields: ' + str(sorted(required - set(scaffold))))
    for key in TEXT_FIELDS:
        if key in scaffold and not isinstance(scaffold[key], str):
            raise ValueError('Scaffold text must be a string: ' + key)
    if scaffold.get('step_limit', 250) != 250 or scaffold.get('cost_limit', 0) != 0:
        raise ValueError('Task budgets are fixed at 250 calls and cost_limit=0')
    template_env = Environment(undefined=StrictUndefined)
    # Match the common task runner's available template variables.
    context = dict(task='TASK', problem_statement='TASK', error='ERROR',
                   output={'output': 'OBSERVATION', 'returncode': 0, 'exception_info': ''},
                   system='Linux', machine='x86_64', release='test', version='test', node='test')
    for key in required | {'memory_template'}:
        if key in scaffold:
            template_env.from_string(scaffold[key]).render(**context)
    from jinja2 import meta
    variables = meta.find_undeclared_variables(template_env.parse(scaffold['instance_template']))
    if 'task' not in variables:
        raise ValueError('instance_template must retain the task variable')


def task_overrides(scaffold):
    """Convert native flat YAML into the common v2 agent/model text surfaces."""
    validate_scaffold(scaffold)
    system = scaffold['system_template']
    memory = scaffold.get('memory_template', '').strip()
    if memory and memory not in system:  # load_scaffold may already have composed it.
        system += '\n\n' + memory
    return ({'system_template': system, 'instance_template': scaffold['instance_template']},
            {'format_error_template': scaffold['format_error_template'],
             'observation_template': scaffold['action_observation_template'],
             'multimodal_regex': scaffold.get('multimodal_regex', '')})


def escape_literal_probe_examples(scaffold):
    """Preserve quoted Django template examples inside the outer Jinja prompt.

    Only the explicitly named Django probe CLI's --template argument is literal.
    Undefined variables elsewhere still fail the unchanged strict validator.
    """
    corrected = deepcopy(scaffold)
    pattern = re.compile(r"(--template\s+)(['\"])(.*?)(\2)")
    for field in TEXT_FIELDS:
        text = corrected.get(field)
        if not isinstance(text, str):
            continue
        lines = []
        for line in text.splitlines(keepends=True):
            if 'django_template_probe/main.py' in line:
                def quote(match):
                    body = match.group(3)
                    if ('{{' in body or '{%' in body) and '{% raw %}' not in body:
                        body = '{% raw %}' + body + '{% endraw %}'
                    return match.group(1) + match.group(2) + body + match.group(4)
                line = pattern.sub(quote, line)
            lines.append(line)
        corrected[field] = ''.join(lines)
    return corrected


def validate_scaffold_file(path):
    """Validate first; normalize known literal examples only when required."""
    path = Path(path)
    original_text = path.read_text()
    scaffold = yaml.safe_load(original_text)
    from jinja2 import UndefinedError, TemplateSyntaxError
    try:
        validate_scaffold(scaffold)
        return
    except (UndefinedError, TemplateSyntaxError):
        corrected = escape_literal_probe_examples(scaffold)
        if corrected == scaffold:
            raise
        validate_scaffold(corrected)
    before_hash = hashlib.sha256(original_text.encode()).hexdigest()
    backup = path.with_name(path.name + '.before_literal_escape_' + before_hash[:12])
    if not backup.exists():
        backup.write_text(original_text)
    text = yaml.safe_dump(corrected, sort_keys=False, allow_unicode=True)
    path.write_text(text)
    path.with_name(path.name + '.literal_escape.json').write_text(json.dumps({
        'kind': 'literal Django probe argument quoted for outer Jinja',
        'before_sha256': before_hash, 'after_sha256': hashlib.sha256(text.encode()).hexdigest(),
        'backup': str(backup), 'model_calls': 0, 'scores_changed': False}, indent=2) + '\n')


def protocol_record():
    files = [REFERENCE_TRACE, V2_SOURCE / 'minisweagent/agents/default.py',
             V2_SOURCE / 'minisweagent/models/litellm_model.py',
             Path(__file__), Path(__file__).with_name('shared_v2_worker.py'),
             Path(__file__).with_name('shared_v2_adapter.py'),
             ROOT / 'baselines/recreate/scripts/evolve_utils/evolution.py']
    return {'protocol': PROTOCOL, 'task_runtime': 'mini-swe-agent-2.4.6',
            'optimizer_runtime': 'ReCreate native mini-swe-agent-1.14.4 (isolated process)',
            'initialization': 'fixed shared seed; no LLM initial creation',
            'seed_sha256': hashlib.sha256(json.dumps(seed_scaffold(), sort_keys=True).encode()).hexdigest(),
            'source_sha256': {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in files},
            'initial_assets': 'empty agent_tools; native memory helper scripts with empty memories; no memory instructions in task seed',
            'task_budget': {'step_limit': 250, 'wall_time_seconds': 3600}}


def guard_run(run, *, resume=False):
    """Never overwrite or resume legacy/unmarked runs with a different seed."""
    run = Path(run)
    expected = protocol_record()
    marker = run / 'shared_v2_protocol.json'
    if run.exists() and any(run.iterdir()):
        if not resume or not marker.exists() or json.loads(marker.read_text()) != expected:
            raise RuntimeError('Use a fresh shared-v2 run directory; resume requires an identical protocol/seed')
    elif resume:
        raise RuntimeError('Cannot resume an absent shared-v2 run')
    return expected


def write_fixed_seed(workspace, init_memory):
    workspace = Path(workspace)
    if workspace.exists() and any(workspace.iterdir()):
        raise RuntimeError('Fixed initialization refuses to overwrite a workspace')
    base = workspace / 'global_v000'
    (base / 'agent_tools').mkdir(parents=True)
    init_memory(base)
    scaffold = seed_scaffold()
    validate_scaffold(scaffold)
    (base / 'scaffold.yaml').write_text(yaml.safe_dump(scaffold, sort_keys=False, allow_unicode=True))
    (workspace / 'current').symlink_to(base.name)
    return base


def legacy_analysis_view(canonical):
    """Expose real v2 actions to upstream readers without changing step numbering.

    The original v2 JSON is preserved separately. This view retains tool_calls,
    IDs and extras, projects commands into assistant text, and labels tool output
    as user feedback for the upstream indexer. It never supplies new evidence.
    """
    data = deepcopy(canonical)
    data['analysis_view_protocol'] = 'v2_tool_calls_to_legacy_reader_v1'
    for message in data.get('messages', []):
        role = message.get('role')
        message['content'] = message.get('content') or ''
        if role == 'assistant':
            commands = []
            for call in message.get('tool_calls') or []:
                function = call.get('function', {})
                if function.get('name') != 'bash':
                    continue
                args = function.get('arguments', {})
                try:
                    args = json.loads(args) if isinstance(args, str) else args
                except ValueError:
                    continue
                if isinstance(args, dict) and isinstance(args.get('command'), str):
                    commands.append(args['command'])
            if not commands:
                commands = [a['command'] for a in message.get('extra', {}).get('actions', [])
                            if isinstance(a.get('command'), str)]
            text = message['content'] or message.get('reasoning_content') or ''
            message['content'] = text + ''.join('\n```bash\n' + c + '\n```\n' for c in commands)
        elif role == 'tool':
            message['original_role'] = 'tool'
            message['role'] = 'user'
    return data
