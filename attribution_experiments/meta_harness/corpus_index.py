"""A routing index over available evidence, without synthetic failure diagnoses."""
import json


def write_corpus_index(search, instances, inject_analysis):
    rows = []
    for instance in instances:
        iid = instance['instance_id']
        rows.append({'instance_id': iid, 'repo': instance.get('repo', ''),
                     'trajectory': f'traces/{iid}.traj.json',
                     'provided_analysis': f'provided_analysis/{iid}.json' if inject_analysis else None})
    path = search / 'traces/_index.jsonl'
    path.write_text(''.join(json.dumps(row, ensure_ascii=False) + '\n' for row in rows))
    return path
