from copy import deepcopy
import json
from pathlib import Path
import pytest

from qwen_gpt55_experiments.meta_harness.proposer_logs import evidence_log


def test_evidence_log_preserves_messages_but_omits_connection_config():
    original = {'info': {'config': {'model': {'api_key': 'test-secret', 'api_base': 'test-endpoint'}},
                         'model_stats': {'api_calls': 12}, 'exit_status': 'Submitted', 'submission': ''},
                'messages': [{'role': 'tool', 'content': 'Original task evidence, unchanged.'}],
                'trajectory_format': 'mini-swe-agent-1'}
    before = deepcopy(original)
    public = evidence_log(original)
    assert public['messages'] == original['messages']
    assert public['info']['exit_status'] == 'Submitted'
    assert public['info']['model_stats']['api_calls'] == 12
    assert 'config' not in public['info']
    assert original == before


@pytest.mark.parametrize('fail', [False, True])
def test_live_runtime_log_is_outside_evidence_workspace(tmp_path, monkeypatch, fail):
    from qwen_gpt55_experiments.meta_harness import run_native as module
    search = tmp_path / 'search'; search.mkdir()
    public = search / 'logs/analysis.traj.json'
    monkeypatch.setattr(module, 'LocalEnvironment', lambda **kwargs: object())
    monkeypatch.setattr(module, 'build_optimizer_model', lambda **kwargs: object())
    class Agent:
        def __init__(self, **kwargs):
            self.path = kwargs['output_path']; self.n_calls = 1
        def run(self, task):
            assert not self.path.is_relative_to(search)
            assert not public.exists()
            if fail:
                raise RuntimeError('simulated API error')
            return {'exit_status': 'Submitted', 'submission': ''}
        def save(self, path):
            Path(path).write_text(json.dumps({'info': {'config': {'api_base': 'private-endpoint'},
                'model_stats': {'api_calls': 1}, 'exit_status': 'APIError' if fail else 'Submitted'},
                'messages': [{'role': 'user', 'content': 'task evidence'}]}))
    monkeypatch.setattr(module, 'DefaultAgent', Agent)
    kwargs = dict(search=search, api_key='test-only', system_template='system', task='task',
                  trajectory_path=public, step_limit=40)
    if fail:
        with pytest.raises(RuntimeError, match='simulated'):
            module.run_proposer(**kwargs)
    else:
        assert module.run_proposer(**kwargs)['exit_status'] == 'Submitted'
    assert 'config' not in json.loads(public.read_text())['info']
    assert (tmp_path / 'proposer_runtime_logs/analysis.traj.json').exists()
