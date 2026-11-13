import json
from types import SimpleNamespace

import pytest


@pytest.mark.parametrize('status', ['APIError: provider rejected request', 'LimitsExceeded'])
def test_failed_optimizer_does_not_promote_partial_scaffold(monkeypatch, tmp_path, status):
    from evolve_utils.evolution import run_recreate_evolution_isolated
    import recreate_agent.recreate_agent as module

    class FakeOptimizer:
        def __init__(self, workspace_path, **kwargs):
            self.workspace = workspace_path
            self.model = SimpleNamespace(n_calls=4, cost=0)
        def evolve(self, **kwargs):
            (self.workspace / 'working/scaffold.yaml').write_text('partial optimizer change')
            return status, str(tmp_path / 'failed_optimizer.traj.json')
    monkeypatch.setattr(module, 'ReCreateAgent', FakeOptimizer)
    (tmp_path / 'scaffold.yaml').write_text('original scaffold')
    result = SimpleNamespace(success=True, score=1.0, error='', details={}, eval_result={}, formatted_output='success')
    with pytest.raises(RuntimeError, match='optimizer did not complete'):
        run_recreate_evolution_isolated(instance=SimpleNamespace(instance_id='task', problem_statement='task'),
            eval_result=result, agent_exit_status='Submitted', working_dir=tmp_path, base_scaffold_dir=tmp_path,
            traj_path=tmp_path / 'not_created.json', container_id='', recreate_model='unused', recreate_temp=1,
            domain='swe', domain_config={}, runs_recreate_dir=tmp_path / 'runs')
    assert (tmp_path / 'scaffold.yaml').read_text() == 'original scaffold'
    receipt = json.loads((tmp_path / 'evolution_status.json').read_text())
    assert receipt['completed'] is False and receipt['exit_status'] == status
    preserved = list(tmp_path.glob('recreate_failed_*'))
    assert len(preserved) == 1
    assert (preserved[0] / 'working/scaffold.yaml').read_text() == 'partial optimizer change'
