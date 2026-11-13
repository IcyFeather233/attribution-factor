"""Explicit new-protocol entrypoint for all five native method adapters.

No changes are installed in old workers. New jobs must use this entrypoint and
fresh directories; old experiments cannot be resumed with new evidence.
"""
import argparse
import importlib
import json
import os
from pathlib import Path
import sys

from .common import analysis_payloads as legacy
from .common import position_evidence as position
from .common.udocker_runtime import persistent_environment
from .plan_budget_repair import jobs, ROOT, PACKAGE


def install_common_interfaces():
    legacy.load_payloads = position.read_payloads
    original_snapshot = legacy.snapshot_payloads

    def snapshot(run, group, payloads):
        for payload in payloads.values():
            position.verify_payload(payload)
        path = original_snapshot(run, group, payloads)
        receipt = {'protocol': position.PROTOCOL, 'group': group, 'tasks': len(payloads),
                   'stage': 'resolved original step content available at native optimizer input',
                   'claim': 'delivery only; does not prove model attention or causal use',
                   'anchors': {iid: {key: value for key, value in p['provided_information'].items()
                                    if key in position.ANCHORS} for iid, p in payloads.items()}}
        (Path(run) / 'position_evidence_protocol.json').write_text(json.dumps(receipt, indent=2) + '\n')
        return path

    legacy.snapshot_payloads = snapshot
    legacy.render_payload = lambda p: (position.INSTRUCTIONS + '\n' + json.dumps(
        p['provided_information'], ensure_ascii=False, indent=2)) if p['provided_information'] else ''
    # Flow is an explicit additional treatment. Its original graph remains intact,
    # but position-only anchors and their raw messages must survive the flow route.
    from .common import flow_evidence as flow
    old_packet, old_render = flow.packet, flow.render

    def packet(payload, task_id=None):
        result = old_packet(payload, task_id)
        info = flow.visible(payload)
        if result and info.get('position_evidence_protocol') == position.PROTOCOL:
            result['historical_step_evidence'] = info['historical_step_evidence']
            result['position_interpretation'] = position.INSTRUCTIONS
        return result

    def render(payload, task_id=None, *, current_context=''):
        text = old_render(payload, task_id, current_context=current_context)
        info = flow.visible(payload)
        if text and info.get('position_evidence_protocol') == position.PROTOCOL:
            text = position.INSTRUCTIONS + '\n' + json.dumps(info['historical_step_evidence'], ensure_ascii=False) + '\n' + text
        return text

    flow.packet, flow.render = packet, render


def install_method_interfaces(method, payloads=None):
    if method == 'gepa':
        sys.path.insert(0, str(ROOT / 'baselines/gepa/src'))
        module = importlib.import_module('qwen_gpt55_experiments.gepa.run_native')
        original = module.build_reflection_prompt_template

        def prompt():
            text = original()
            start = text.index('A "Provided Failure Analysis" section')
            end = text.index('Propose one general improvement', start)
            return text[:start] + position.INSTRUCTIONS + '\n\n' + text[end:]

        module.build_reflection_prompt_template = prompt
    elif method == 'meta_harness':
        module = importlib.import_module('qwen_gpt55_experiments.meta_harness.run_native')
        module.ANALYST_SYSTEM += '\n' + position.INSTRUCTIONS + (
            '\nRead historical_step_evidence in provided_analysis/<task>.json. '
            'Do not look for a what_went_wrong field: this protocol does not provide one. '
            'In evidence_summary cite the task ID and supplied message_index for supporting examples.')
        module.DOMAIN_SPEC += '\n' + position.INSTRUCTIONS
    elif method == 'harnessfix':
        module = importlib.import_module('qwen_gpt55_experiments.harnessfix.native_profile')
        adapter = importlib.import_module('qwen_gpt55_experiments.harnessfix.run_native')
        adapter.BRIDGE = PACKAGE / 'harnessfix/position_bridge.py'
    elif method == 'recreate':
        module = importlib.import_module('qwen_gpt55_experiments.recreate.run_shared_v2')
        driver = importlib.import_module('qwen_gpt55_experiments.recreate.run_native')
        from .recreate.position_adapter import install
        install(driver, payloads or {})
    else:
        module = importlib.import_module('qwen_gpt55_experiments.scope.run_native')
    return module


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--method', choices=['gepa', 'scope', 'recreate', 'meta_harness', 'harnessfix'], required=True)
    parser.add_argument('--group', choices=legacy.GROUPS, required=True)
    parser.add_argument('--run-name', default='position_step_v1_full148_20261019')
    parser.add_argument('--payload-root', type=Path, default=position.PAYLOAD_ROOT)
    parser.add_argument('--validate-only', action='store_true')
    args, remaining = parser.parse_known_args()
    if not args.run_name.startswith('position_step_v1_') or Path(args.run_name).name != args.run_name:
        parser.error('A fresh position_step_v1_ run name is required')
    if any(x.split('=', 1)[0] in ['--run-dir', '--ablation-group', '--resume-metrics-from'] for x in remaining):
        parser.error('Do not override run identity or reuse legacy metrics')
    root = args.payload_root.resolve()
    payloads = position.read_payloads(args.group, root)
    row = next(j for j in jobs([args.group]) if j['method'] == args.method)
    run = Path(row['run_dir']).parent / args.run_name
    marker = run / 'position_evidence_protocol.json'
    if run.exists():
        if '--resume' not in remaining or not marker.exists():
            raise ValueError('Existing run requires a position protocol marker and explicit --resume')
        previous = json.loads(marker.read_text())
        if previous['protocol'] != position.PROTOCOL or previous['group'] != args.group:
            raise ValueError('Run protocol/group mismatch')
        for iid, payload in payloads.items():
            saved = json.loads((run / 'provided_analysis' / (iid + '.json')).read_text())
            if saved != payload:
                raise ValueError('Cannot resume with changed position evidence')
    native = row['command'][4:]
    flag = '--run-dir' if args.method == 'harnessfix' else '--run-name'
    native[native.index(flag) + 1] = str(run) if args.method == 'harnessfix' else args.run_name
    native += ['--payload-root', str(root), *remaining]
    if args.validate_only:
        print(json.dumps({'method': args.method, 'group': args.group, 'tasks': len(payloads),
                          'protocol': position.PROTOCOL, 'run_dir': str(run), 'native_arguments': native}))
        return
    os.environ.update(persistent_environment())
    os.environ['EXPERIMENT_ALIGNED_TASK_BUDGET'] = '1'
    os.environ.setdefault('EXPERIMENT_CONTAINER_BACKEND', 'remote_docker')
    os.environ.setdefault('EXPERIMENT_REMOTE_HOST', 'server241')
    from .run_pipeline import child_environment, load_sidecar_key
    from .common.model_config import task_api_key
    os.environ.update(child_environment(load_sidecar_key(), task_api_key()))
    install_common_interfaces()
    module = install_method_interfaces(args.method, payloads)
    sys.argv = [str(Path(module.__file__)), *native]
    module.main()


if __name__ == '__main__':
    main()
