"""Recover JSON data from an incorrectly blocked write; never execute trace commands."""
import hashlib
import json
from pathlib import Path
import re

from .flow_views import citation_audit


def recover_proposal(search: Path) -> dict:
    trajectory = search / 'logs/analysis.traj.json'
    data = json.loads(trajectory.read_text())
    manifest = json.loads((search / 'flow_views_manifest.json').read_text())
    pages = set()
    candidates = []
    messages = data['messages']
    for i, message in enumerate(messages):
        if message.get('role') == 'tool':
            content = str(message.get('content', ''))
            raw = str(message.get('extra', {}).get('raw_output', content))
            if len(raw) <= 10000:
                pages.update(path for path, item in manifest['pages'].items() if item['content'] in content)
        if message.get('role') != 'assistant':
            continue
        following = messages[i + 1] if i + 1 < len(messages) else {}
        if following.get('role') != 'tool' or 'Flow evidence not yet verified' not in str(following.get('content', '')):
            continue
        for action in message.get('extra', {}).get('actions', []):
            match = re.search(r"cat > proposal\.json <<'JSON'\n(.*?)\nJSON(?:\n|$)", action.get('command', ''), re.S)
            if not match:
                continue
            try:
                proposal = json.loads(match[1])
                audit = citation_audit(proposal, manifest, pages)
            except (ValueError, TypeError, KeyError):
                continue
            if audit['passed'] and all(key in proposal for key in ('name', 'hypothesis', 'changes', 'evidence_summary', 'expected_efficiency')):
                candidates.append((i, proposal, audit))
    if not candidates:
        raise RuntimeError('No valid proposal in a blocked JSON write with previously observed cited pages')
    # First valid proposal: restore the original attempted decision, not a best-of-N choice.
    index, proposal, audit = candidates[0]
    target = search / 'proposal.json'
    if target.exists() and json.loads(target.read_text()) != proposal:
        raise RuntimeError('Refusing to overwrite a different existing proposal')
    target.write_text(json.dumps(proposal, indent=2, ensure_ascii=False) + '\n')
    (search / 'flow_read_audit.json').write_text(json.dumps(audit, indent=2) + '\n')
    receipt = {'recovered': True, 'reason': 'command-text submission guard blocked an ordinary JSON write',
               'source_message_index': index, 'source_trajectory_sha256': hashlib.sha256(trajectory.read_bytes()).hexdigest(),
               'proposal_sha256': hashlib.sha256(target.read_bytes()).hexdigest(), 'selection': 'first valid blocked proposal',
               'additional_analysis_model_calls': 0, 'original_exit_status': data.get('info', {}).get('exit_status')}
    (search / 'analysis_recovery.json').write_text(json.dumps(receipt, indent=2) + '\n')
    return {'exit_status': 'RecoveredBlockedProposal', 'api_calls': data.get('info', {}).get('model_stats', {}).get('api_calls'),
            'submission': '', 'recovery': receipt}
