"""Recover completed per-case evolution after an orderly container drain."""
import json
from pathlib import Path


def recovered_batch_result(output: Path, base: Path, instance_id: str):
    marker = output / 'batch_result.json'
    status_file = output / 'evolution_status.json'
    if status_file.exists() and json.loads(status_file.read_text()).get('completed') is not True:
        return None
    if marker.exists():
        data = json.loads(marker.read_text())
        if data['instance_id'] != instance_id:
            raise ValueError('ReCreate checkpoint task mismatch')
        if data.get('error'):
            return None
        return data
    evaluation = output / 'eval_result.json'
    trajectory = output / f'{instance_id}.traj.json'
    if not evaluation.exists() or not trajectory.exists():
        return None
    if (output / 'recreate_temp').exists():
        raise RuntimeError('Cannot reuse unfinished ReCreate evolution')
    data = json.loads(evaluation.read_text())
    if type(data.get('success')) is not bool or data.get('error'):
        raise RuntimeError('Cannot reuse invalid ReCreate evaluation')
    if not (output / 'scaffold.yaml').is_file():
        raise RuntimeError('Missing evolved scaffold')
    tools = output / 'agent_tools'
    old_tools = base / 'agent_tools'
    memory, old_memory = output / 'agent_memory/memories.yaml', base / 'agent_memory/memories.yaml'
    info = json.loads(trajectory.read_text()).get('info', {})
    return dict(instance_id=instance_id, success=data['success'], score=data.get('score', float(data['success'])),
                scaffold_changed=(output / 'scaffold_diff.txt').is_file(),
                has_new_tools=len(list(tools.iterdir())) > len(list(old_tools.iterdir())) if tools.exists() and old_tools.exists() else False,
                has_new_memories=memory.exists() and old_memory.exists() and memory.read_bytes() != old_memory.read_bytes(),
                error='', exit_status=info.get('exit_status', ''), duration=0.0)
