"""Offline contract tests: real v2 execution, no LLM API or GPU/container creation."""
from copy import deepcopy
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest
import yaml
from jinja2 import UndefinedError

from attribution_experiments.recreate.shared_v2_protocol import (
    ROOT, V2_SOURCE, PROTOCOL, original_config, seed_scaffold, task_overrides,
    validate_scaffold, guard_run, write_fixed_seed, legacy_analysis_view, protocol_record,
)
from attribution_experiments.recreate.shared_v2_adapter import TaskTrace, worker_environment, adapter_class
from attribution_experiments.common.udocker_runtime import persistent_environment


def test_seed_is_the_shared_prompt_and_model_text_with_no_added_guidance():
    seed = seed_scaffold()
    agent, model = task_overrides(seed)
    original = original_config()
    assert agent == {k: original['agent'][k] for k in ('system_template', 'instance_template')}
    assert model == {k: original['model'][k] for k in ('format_error_template', 'observation_template', 'multimodal_regex')}
    assert seed['memory_template'] == ''
    assert seed_scaffold() == seed  # Independent calls/groups get the same seed.


def test_fixed_seed_cannot_overwrite_legacy_or_existing_run(tmp_path):
    workspace = tmp_path / 'workspace'
    def memory(base):
        (base / 'agent_memory').mkdir()
        (base / 'agent_memory/memories.yaml').write_text('memories: []\n')
    base = write_fixed_seed(workspace, memory)
    assert yaml.safe_load((base / 'scaffold.yaml').read_text()) == seed_scaffold()
    assert list((base / 'agent_tools').iterdir()) == []
    with pytest.raises(RuntimeError, match='overwrite'):
        write_fixed_seed(workspace, memory)
    old = tmp_path / 'legacy'
    old.mkdir()
    (old / 'summary.json').write_text('{}')
    with pytest.raises(RuntimeError, match='fresh'):
        guard_run(old, resume=True)
    current = tmp_path / 'new'
    current.mkdir()
    (current / 'shared_v2_protocol.json').write_text(json.dumps(protocol_record()))
    assert guard_run(current, resume=True)['protocol'] == PROTOCOL
    with pytest.raises(RuntimeError, match='fresh'):
        guard_run(current)


@pytest.mark.parametrize('change', [
    {'step_limit': 999}, {'cost_limit': 7}, {'model_name': 'different-model'},
    {'instance_template': 'no task variable'}, {'action_observation_template': '{{ unknown }}'},
])
def test_incompatible_candidate_cannot_silently_change_protocol(change):
    with pytest.raises((ValueError, UndefinedError)):
        validate_scaffold({**seed_scaffold(), **change})


def test_memory_and_model_templates_reach_task_config_once():
    seed = seed_scaffold()
    seed.update(memory_template='Check the saved memory.', format_error_template='Invalid: {{ error }}',
                action_observation_template='{{ output.returncode }}: {{ output.output }}')
    agent, model = task_overrides(seed)
    assert agent['system_template'].count('Check the saved memory.') == 1
    seed['system_template'] = agent['system_template']  # native load_scaffold composes it
    assert task_overrides(seed)[0]['system_template'].count('Check the saved memory.') == 1
    assert model['observation_template'] == seed['action_observation_template']


def test_compatibility_view_keeps_actions_observations_and_step_identity(tmp_path):
    canonical = {'info': {'mini_version': '2.4.6'}, 'messages': [
        {'role': 'assistant', 'content': None, 'tool_calls': [{'id': 'call-1', 'type': 'function',
          'function': {'name': 'bash', 'arguments': json.dumps({'command': 'cat /testbed/example.py'})}}]},
        {'role': 'tool', 'tool_call_id': 'call-1', 'content': 'ValueError: bad value'},
    ]}
    before = deepcopy(canonical)
    view = legacy_analysis_view(canonical)
    assert canonical == before
    assert len(view['messages']) == len(canonical['messages'])
    assert 'cat /testbed/example.py' in view['messages'][0]['content']
    assert view['messages'][0]['tool_calls'] == canonical['messages'][0]['tool_calls']
    assert view['messages'][1]['role'] == 'user' and view['messages'][1]['original_role'] == 'tool'
    path = tmp_path / 'trajectory.v2.json'
    path.write_text(json.dumps(canonical))
    proxy = TaskTrace(path)
    proxy.save(tmp_path / 'task.traj.json')
    proxy.save(tmp_path / 'trajectory.json')
    assert json.loads((tmp_path / 'trajectory.json').read_text())['messages'] == canonical['messages']
    assert json.loads((tmp_path / 'task.traj.json').read_text())['messages'] == view['messages']


def test_worker_environment_removes_legacy_import_and_optimizer_credentials(monkeypatch):
    monkeypatch.setenv('PYTHONPATH', '/old/recreate/src')
    monkeypatch.setenv('OPTIMIZER_API_KEY', 'optimizer-test-secret')
    monkeypatch.setenv('OPENAI_API_KEY', 'optimizer-test-secret')
    monkeypatch.setenv('TASK_MODEL_API_KEY', 'task-test-secret')
    env = worker_environment()
    assert env['PYTHONPATH'].split(os.pathsep) == [str(V2_SOURCE), str(ROOT)]
    assert 'OPTIMIZER_API_KEY' not in env and 'OPENAI_API_KEY' not in env
    assert env['TASK_MODEL_API_KEY'] == 'task-test-secret'
    assert env['TMPDIR'].startswith('/volume/med/')


def test_worker_failure_releases_parent_owned_container(monkeypatch, tmp_path):
    from types import SimpleNamespace
    cleaned = []
    env = SimpleNamespace(image='test-image', cleanup=lambda: cleaned.append(True))
    class Parent:
        def create_environment(self, *args):
            return env, 'owned'
    adapter = adapter_class(SimpleNamespace(UdockerSWEBenchAdapter=Parent))()
    adapter.step_limit = 250
    def failed(*args, **kwargs):
        raise subprocess.CalledProcessError(1, ['worker'])
    monkeypatch.setattr('attribution_experiments.recreate.shared_v2_adapter.subprocess.run', failed)
    from attribution_experiments.common.model_config import TASK_MODEL
    with pytest.raises(subprocess.CalledProcessError):
        adapter.run_agent(SimpleNamespace(instance_id='x', problem_statement='test'),
                          seed_scaffold(), TASK_MODEL, tmp_path)
    assert cleaned == [True]


def test_actual_v2_loop_tool_execution_submission_and_parent_ownership(tmp_path):
    code = r'''
import json, sys
from pathlib import Path
from types import SimpleNamespace
from litellm import ModelResponse
from attribution_experiments.recreate.shared_v2_protocol import seed_scaffold, PROTOCOL
from attribution_experiments.recreate.shared_v2_worker import run_task, BorrowedRemoteEnvironment
from minisweagent.environments.local import LocalEnvironment
from minisweagent.models.litellm_model import LitellmModel
root=Path(sys.argv[1])
class ScriptedModel(LitellmModel):
    def __init__(self):
        super().__init__(model_name='openai/test-offline', cost_tracking='ignore_errors')
        self.seen=[]
    def _query(self,messages,**kwargs):
        self.seen.append(messages)
        command='echo v2-command-ok' if len(self.seen)==1 else 'echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT; echo fixed-patch'
        return ModelResponse(model='openai/test-offline', choices=[{'finish_reason':'tool_calls','message':{
            'role':'assistant','content':None,'tool_calls':[{'id':'c'+str(len(self.seen)), 'type':'function',
            'function':{'name':'bash','arguments':json.dumps({'command':command})}}]}}])
model=ScriptedModel()
request={'protocol':PROTOCOL,'instance_id':'offline-case','problem_statement':'Test task',
         'output_dir':str(root),'scaffold':seed_scaffold(),'step_limit':3,'container_name':'parent-owned'}
reply=run_task(request,environment=LocalEnvironment(cwd=str(root)),model=model)
assert reply['exit_status']=='Submitted' and reply['submission'].strip()=='fixed-patch'
data=json.loads((root/'trajectory.v2.json').read_text())
assert data['info']['mini_version']=='2.4.6' and data['info']['model_stats']['api_calls']==2
assert any(m['role']=='tool' and 'v2-command-ok' in m['content'] for m in data['messages'])
assert data['info']['config']['agent']['step_limit']==3
assert data['info']['config']['agent']['wall_time_limit_seconds']==3600
borrowed=BorrowedRemoteEnvironment.__new__(BorrowedRemoteEnvironment)
borrowed.runtime=SimpleNamespace(cleanup=lambda: (_ for _ in ()).throw(AssertionError('parent container cleaned')))
borrowed.cleanup()
del borrowed
print('OFFLINE_V2_OK')
'''
    result = subprocess.run([sys.executable, '-B', '-c', code, str(tmp_path)],
                            cwd=ROOT, env=worker_environment(), text=True, capture_output=True, timeout=90)
    assert result.returncode == 0, result.stdout[-3000:] + result.stderr[-3000:]
    assert 'OFFLINE_V2_OK' in result.stdout


def test_native_evolution_and_synthesis_preserve_dedicated_evidence(tmp_path):
    code = r'''
import json,sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import yaml
from attribution_experiments.recreate import run_native as legacy
from attribution_experiments.recreate.shared_v2_adapter import install,TaskTrace
from attribution_experiments.recreate.shared_v2_protocol import seed_scaffold,validate_scaffold
from recreate_agent.adapters import UnifiedInstance,UnifiedResult
import recreate_agent.recreate_agent as native_optimizer
import minisweagent.agents.default as native_agent
import minisweagent.models as native_models
install(legacy)
root=Path(sys.argv[1]); workspace=root/'workspace'; runs=root/'optimizer_logs'
evidence={'failure_causing_wrong_step':{'step':3,'message_index':6,'role':'implementation','what_went_wrong':'wrong branch','evidence':'trace'}}
class FakeEnv:
    container_name='owned'
    def execute(self,*args,**kwargs): return {'returncode':0,'output':'diff --git a/x b/x'}
    def cleanup(self): pass
class Adapter(legacy.UdockerSWEBenchAdapter):
    def run_agent(self,instance,scaffold,model_name,output_dir,tools_dir,temperature):
        self.environments['owned']=FakeEnv()
        canonical={'info':{'mini_version':'2.4.6','model_stats':{'api_calls':1,'instance_cost':0},
                          'exit_status':'Submitted','submission':'patch'},'messages':[
            {'role':'assistant','content':None,'tool_calls':[{'id':'c','type':'function','function':{
             'name':'bash','arguments':json.dumps({'command':'cat /testbed/x.py'})}}]},
            {'role':'tool','tool_call_id':'c','content':'TypeError: test failure'}]}
        p=output_dir/'trajectory.v2.json';p.write_text(json.dumps(canonical))
        return 'Submitted','patch','owned',SimpleNamespace(agent=TaskTrace(p),env=FakeEnv(),cleanup=lambda:None)
def grade(**kwargs):
    return UnifiedResult(instance_id=kwargs['instance'].instance_id,success=False,error='',
                         details={'tests_passed':0,'tests_failed':1},formatted_output='clean evaluator output')
class Optimizer:
    def __init__(self,workspace_path,**kwargs):
        self.workspace=workspace_path;self.model=SimpleNamespace(n_calls=1,cost=0)
    def evolve(self,**kwargs):
        result=json.loads(next((self.workspace/'results').glob('*/result.json')).read_text())
        assert json.loads(result['provided_failure_analysis'])==evidence
        assert result['eval_result']=={'tests_passed':0,'tests_failed':1,'provided_failure_analysis':evidence}
        assert (next((self.workspace/'results').glob('*/evaluation.txt'))).read_text()=='clean evaluator output'
        trace=json.loads(next((self.workspace/'results').glob('*/*.traj.json')).read_text())
        assert 'cat /testbed/x.py' in trace['messages'][0]['content']
        index=json.loads(next((self.workspace/'results').glob('*/trajectory_index.json')).read_text())
        assert index['meta']['total_steps']==1 and index['errors'][0]['type']=='TypeError'
        p=self.workspace/'working/scaffold.yaml';sc=yaml.safe_load(p.read_text())
        sc['instance_template']+='\nReview edge cases.';p.write_text(yaml.safe_dump(sc))
        return 'Submitted','offline-optimizer-trace'
adapter=Adapter(step_limit=250,analysis_payloads={'case':{'provided_information':evidence}})
setup=legacy.initialize_workspace(workspace=workspace,runs_recreate_dir=runs,adapter=adapter,smoke=False)
out=workspace/'batch_000/case'
with patch.object(legacy,'evaluate_patch',grade),patch.object(native_optimizer,'ReCreateAgent',Optimizer):
    result=legacy.run_single_evolution(instance=UnifiedInstance('case','task'),global_scaffold_dir=workspace/'global_v000',
        output_dir=out,adapter=adapter,agent_model=legacy.TASK_MODEL,recreate_model=legacy.OPTIMIZER_MODEL,
        agent_temp=1.,recreate_temp=1.,domain='swe',domain_config=setup['domain_config'],runs_recreate_dir=runs,
        ablation_settings=setup['ablations'])
assert not result.error,result.error
assert result.scaffold_changed
assert yaml.safe_load((workspace/'global_v000/scaffold.yaml').read_text())==seed_scaffold()
class Synthesizer:
    def __init__(self,model,env,**kwargs): self.model=model;self.env=env;self.messages=[]
    def add_message(self,role,content): self.messages.append({'role':role,'content':content})
    def step(self):
        p=Path(self.env.config.cwd)/'working/scaffold.yaml';sc=yaml.safe_load(p.read_text())
        sc['instance_template']+='\nSynthesized review.';p.write_text(yaml.safe_dump(sc))
        raise native_agent.Submitted('done')
with patch.object(native_agent,'DefaultAgent',Synthesizer),patch.object(native_models,'get_model',lambda *a,**k:SimpleNamespace(n_calls=1,cost=0)):
    legacy.run_batch_synthesis(workspace=workspace,batch_dir=out.parent,batch_results={'case':result},
        current_global_dir=workspace/'global_v000',new_global_dir=workspace/'global_v001',batch_idx=0,
        recreate_model=legacy.OPTIMIZER_MODEL,recreate_temp=1.,domain='swe',domain_config=setup['domain_config'],
        runs_recreate_dir=runs,ablation_settings=setup['ablations'])
sc=yaml.safe_load((workspace/'global_v001/scaffold.yaml').read_text());validate_scaffold(sc)
assert 'Synthesized review.' in sc['instance_template']
class BrokenSynthesizer(Synthesizer):
    def step(self): raise RuntimeError('offline interrupted synthesis')
try:
    with patch.object(native_agent,'DefaultAgent',BrokenSynthesizer),patch.object(native_models,'get_model',lambda *a,**k:SimpleNamespace(n_calls=1,cost=0)):
        legacy.run_batch_synthesis(workspace=workspace,batch_dir=out.parent,batch_results={'case':result},
            current_global_dir=workspace/'global_v001',new_global_dir=workspace/'global_v002',batch_idx=1,
            recreate_model=legacy.OPTIMIZER_MODEL,recreate_temp=1.,domain='swe',domain_config=setup['domain_config'],
            runs_recreate_dir=runs,ablation_settings=setup['ablations'])
    raise AssertionError('Interrupted synthesis was accepted')
except RuntimeError as exc:
    assert 'did not complete' in str(exc)
assert not (workspace/'global_v002').exists()
assert list(out.parent.glob('failed_shared_synthesis_*'))
class InvalidOptimizer(Optimizer):
    def evolve(self,**kwargs):
        answer=super().evolve(**kwargs)
        p=self.workspace/'working/scaffold.yaml';sc=yaml.safe_load(p.read_text());sc['step_limit']=999
        p.write_text(yaml.safe_dump(sc));return answer
bad=workspace/'invalid_case'
with patch.object(legacy,'evaluate_patch',grade),patch.object(native_optimizer,'ReCreateAgent',InvalidOptimizer):
    rejected=legacy.run_single_evolution(instance=UnifiedInstance('case','task'),global_scaffold_dir=workspace/'global_v000',
        output_dir=bad,adapter=adapter,agent_model=legacy.TASK_MODEL,recreate_model=legacy.OPTIMIZER_MODEL,
        agent_temp=1.,recreate_temp=1.,domain='swe',domain_config=setup['domain_config'],runs_recreate_dir=runs,
        ablation_settings=setup['ablations'])
assert 'contract invalid' in rejected.error
from attribution_experiments.recreate.resume_state import recovered_batch_result
assert recovered_batch_result(bad,workspace/'global_v000','case') is None
print('NATIVE_CYCLE_OK')
'''
    env = {**os.environ, **persistent_environment(), 'PYTHONDONTWRITEBYTECODE': '1', 'PYTHONPATH': str(ROOT)}
    result = subprocess.run([sys.executable, '-B', '-c', code, str(tmp_path)], cwd=ROOT, env=env,
                            text=True, capture_output=True, timeout=90)
    assert result.returncode == 0, result.stdout[-4000:] + result.stderr[-4000:]
    assert 'NATIVE_CYCLE_OK' in result.stdout
