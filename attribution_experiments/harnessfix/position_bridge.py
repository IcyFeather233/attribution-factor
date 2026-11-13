"""New-protocol child bridge; old HarnessFix jobs keep their original bridge."""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from attribution_experiments.common import position_evidence as position


def install_diagnosis(module, payload_dir):
    original = module.DefaultAgent.run

    def run(self, *args, **kwargs):
        iid = kwargs.get('instance_id')
        if iid:
            payload = json.loads((payload_dir / (iid + '.json')).read_text())
            position.verify_payload(payload)
            if payload['provided_information']:
                kwargs['position_evidence'] = json.dumps(payload['provided_information'], ensure_ascii=False)
                self.config.instance_template += '\n' + position.INSTRUCTIONS + '\n{{position_evidence}}\n'
        return original(self, *args, **kwargs)

    module.DefaultAgent.run = run


def aggregate_packet(results, payload_dir):
    """Deliver native diagnoses BEFORE the first planning response, for all conditions.

Original-message excerpts are available in the frozen payload files and were
already supplied to each per-case diagnosis. No offline annotation text returns.
"""
    rows = []
    for record in results:
        iid = record['instance_id']
        payload = json.loads((payload_dir / (iid + '.json')).read_text())
        position.verify_payload(payload)
        info = payload['provided_information']
        rows.append({'instance_id': iid,
                     'affected_component': record.get('affected_component'),
                     'failure_reason': position.clip(str(record.get('failure_reason', '')), 400),
                     'agent_design_issue': position.clip(str(record.get('agent_design_issue', '')), 600),
                     'position_hints': {k: info[k] for k in position.ANCHORS if k in info},
                     'original_step_file': str(payload_dir / (iid + '.json')) if info else None})
    return ('\n## Native per-case diagnoses delivered before planning\n'
            'The following are outputs of your native diagnosis phase, not supplied ground-truth '
            'error explanations. Excerpts are mechanically capped; read the full analysis/context '
            'files for detail before making claims. Use the position references and original-step '
            'files when grounding fixes. Reading to a shell file alone is not model consumption.\n'
            + json.dumps(rows, ensure_ascii=False) + '\n')


def install_aggregate(module, payload_dir):
    # The old helper keyed contexts only by "improvement_plan", sharing a directory
    # across independent experiments. Keep new contexts private to each run.
    module._aggregate_context_dir = lambda output_path: Path(output_path).parent / 'aggregate_context_position_v1'
    original = module.call_aggregate_agent

    def call(**kwargs):
        packet = aggregate_packet(kwargs.get('results') or [], payload_dir)
        kwargs['mode_system_prompt'] += packet
        output = Path(kwargs['output_path'])
        output.parent.mkdir(parents=True, exist_ok=True)
        (output.parent / 'position_aggregate_delivery.json').write_text(json.dumps({
            'protocol': position.PROTOCOL, 'diagnoses_in_initial_prompt': len(kwargs.get('results') or []),
            'packet_sha256': position.digest(packet.encode()),
            'stage': 'aggregate initial prompt, before any generated plan',
        }, indent=2) + '\n')
        return original(**kwargs)

    module.call_aggregate_agent = call


def main():
    from attribution_experiments.harnessfix import upstream_bridge as bridge
    args, _ = bridge.parse_args()
    sys.path.insert(0, str(bridge.UPSTREAM))
    sys.path.insert(0, str(bridge.UPSTREAM / 'agent_framework/src'))
    bridge.install_flow_diagnosis = install_diagnosis
    if args.command == 'aggregate':
        from failure_analysis import aggregate_results
        install_aggregate(aggregate_results, args.payload_dir)
    bridge.main()


if __name__ == '__main__':
    main()
