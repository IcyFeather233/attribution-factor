"""Keep optimizer connection configuration outside its evidence workspace."""
from copy import deepcopy
import json


def evidence_log(trajectory):
    """Keep messages verbatim for input auditing; omit runtime configuration."""
    public = deepcopy(trajectory)
    info = public.get('info', {})
    public['info'] = {key: info[key] for key in ('model_stats', 'mini_version', 'exit_status', 'submission') if key in info}
    return public


def publish_evidence_log(private_path, public_path):
    public_path.parent.mkdir(parents=True, exist_ok=True)
    public_path.write_text(json.dumps(evidence_log(json.loads(private_path.read_text())),
                                     ensure_ascii=False, indent=2) + '\n')
