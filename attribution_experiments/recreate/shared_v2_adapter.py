"""Legacy-side bridge; no imports of the task-side v2 mini-swe-agent here."""
from copy import deepcopy
from dataclasses import replace
import json
import os
from pathlib import Path
import subprocess
import sys
import uuid
from types import SimpleNamespace

from .shared_v2_protocol import (
    PROTOCOL, ROOT, V2_SOURCE, original_config, seed_scaffold, validate_scaffold,
    legacy_analysis_view, write_fixed_seed,
)
from attribution_experiments.common.udocker_runtime import persistent_environment


class TaskTrace:
    def __init__(self, path):
        self.path = Path(path)

    def save(self, path, **kwargs):
        canonical = json.loads(self.path.read_text())
        # Final evaluation records stay canonical. Per-case diagnostics use a
        # compatible text view for native ReCreate's unmodified trajectory tools.
        data = canonical if Path(path).name == 'trajectory.json' else legacy_analysis_view(canonical)
        data['canonical_trajectory'] = str(self.path)
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps(data, indent=2, ensure_ascii=False) + '\n')


def install_trace_saver(legacy):
    import minisweagent.run.utils.save as save_module
    original = save_module.save_traj
    def save(agent, path, **kwargs):
        if isinstance(agent, TaskTrace):
            return agent.save(path, **kwargs)
        return original(agent, path, **kwargs)
    save_module.save_traj = save
    legacy.save_traj = save


def worker_environment():
    env = {**os.environ, **persistent_environment(), 'PYTHONDONTWRITEBYTECODE': '1'}
    # Remove the legacy import path entirely; both versions use the same package name.
    env['PYTHONPATH'] = str(V2_SOURCE) + os.pathsep + str(ROOT)
    for key in ('OPTIMIZER_API_KEY', 'OPENAI_API_KEY', 'MSWEA_MODEL_API_KEY', 'LLM_API_BASE'):
        env.pop(key, None)
    return env


def adapter_class(legacy):
    class SharedMiniAdapter(legacy.UdockerSWEBenchAdapter):
        def get_prompt_config(self):
            config = super().get_prompt_config()
            seed = seed_scaffold()
            config.scaffold_system = seed['system_template']
            config.scaffold_instance = seed['instance_template']
            config.scaffold_format_error = seed['format_error_template']
            config.submit_command = 'echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT'
            config.submit_instructions = 'Task agent uses the shared v2 submission protocol: marker on first output line.'
            config.workflow_notes += [
                'The TASK agent runs mini-swe-agent v2.4.6 using the bash function tool with JSON command arguments. '
                'Your own optimizer interface still uses one bash code block per turn; do not impose that format on the task agent.',
                'Task scaffold is flat YAML: system_template, instance_template, memory_template, '
                'format_error_template, action_observation_template, multimodal_regex; step_limit=250 and cost_limit=0 are fixed. '
                'action_observation_template is the v2 model observation template (output.output, output.returncode).',
                'Keep {{task}} in instance_template and use the task bash function tool for submission. '
                'Tools and memory remain editable; seed has no learned memory or additional task guidance.',
                'Task diagnostic traces are faithful compatibility views; tool_calls and original_role are retained. '
                'Canonical v2 trajectories are saved separately. Historical attribution indices belong to the original failed trace.',
            ]
            return config

        def run_agent(self, instance, scaffold, model_name, output_dir, tools_dir=None, temperature=1.0):
            from attribution_experiments.common.model_config import TASK_MODEL
            if model_name != TASK_MODEL or temperature != 1.0:
                raise ValueError('Shared task model/sampling settings are fixed')
            validate_scaffold(scaffold)
            output_dir = Path(output_dir).resolve()
            workspace = Path(tools_dir or output_dir).resolve()
            output_dir.mkdir(parents=True, exist_ok=True)
            env, container = self.create_environment(instance, output_dir, workspace)
            request = {'protocol': PROTOCOL, 'instance_id': instance.instance_id,
                       'problem_statement': instance.problem_statement, 'scaffold': deepcopy(scaffold),
                       'step_limit': self.step_limit, 'output_dir': str(output_dir),
                       'workspace': str(workspace), 'container_name': container, 'image': env.image}
            request_path = output_dir / 'task_worker_request.json'
            request_path.write_text(json.dumps(request, indent=2) + '\n')
            try:
                with (output_dir / 'task_worker.log').open('a') as log:
                    subprocess.run([sys.executable, '-B', '-m',
                                    'attribution_experiments.recreate.shared_v2_worker',
                                    '--request', str(request_path)], cwd=ROOT, env=worker_environment(),
                                   stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                                   timeout=5400, check=True)
                reply = json.loads((output_dir / 'task_worker_result.json').read_text())
                if (reply.get('protocol') != PROTOCOL or reply.get('instance_id') != instance.instance_id
                        or reply.get('container_name') != container or reply.get('task_runtime') != '2.4.6'):
                    raise RuntimeError('Task worker returned a mismatched reply')
                canonical = output_dir / 'trajectory.v2.json'
                wrapper = SimpleNamespace(env=env, agent=TaskTrace(canonical), cleanup=env.cleanup)
                return reply['exit_status'], reply['submission'], container, wrapper
            except BaseException:
                env.cleanup()
                raise
    return SharedMiniAdapter


def install(legacy):
    import minisweagent
    if minisweagent.__version__ != '1.14.4':
        raise RuntimeError('ReCreate optimizer must stay in its legacy runtime; use a fresh process')
    legacy.UdockerSWEBenchAdapter = adapter_class(legacy)
    install_trace_saver(legacy)
    def initialize(*, workspace, runs_recreate_dir, adapter, smoke):
        from evolve_utils.scaffold_ops import init_agent_memory
        write_fixed_seed(workspace, init_agent_memory)
        return {'domain_config': adapter.get_recreate_agent_config(), 'ablations': {
            'ablation_trajectory': True, 'ablation_environment': True,
            'ablation_eval_results': True, 'ablation_modification_guidance': True}}
    legacy.initialize_workspace = initialize
    original_evolution = legacy.run_single_evolution
    def evolution(**kwargs):
        result = original_evolution(**kwargs)
        if not result.error:
            import yaml
            try:
                from .shared_v2_protocol import validate_scaffold_file
                validate_scaffold_file(kwargs['output_dir'] / 'scaffold.yaml')
            except Exception as exc:
                path = kwargs['output_dir'] / 'evolution_status.json'
                status = json.loads(path.read_text()) if path.exists() else {}
                status.update(completed=False, candidate_validation_error=str(exc))
                path.write_text(json.dumps(status, indent=2) + '\n')
                return replace(result, error='Shared-v2 candidate contract invalid: ' + str(exc))
        return result
    legacy.run_single_evolution = evolution
    original_synthesis = legacy.run_batch_synthesis
    def synthesis(**kwargs):
        pattern = 'synthesis_batch' + f"{kwargs['batch_idx']:03d}" + '_*/synthesis.traj.json'
        prior = set(kwargs['runs_recreate_dir'].glob(pattern))
        modifications = any(r.scaffold_changed or r.has_new_tools or r.has_new_memories
                            for r in kwargs['batch_results'].values())
        import yaml
        try:
            result = original_synthesis(**kwargs)
            traces = set(kwargs['runs_recreate_dir'].glob(pattern)) - prior
            if modifications and (not traces or any(json.loads(p.read_text()).get('exit_status')
                                                     not in ('Submitted', 'Resolved') for p in traces)):
                raise RuntimeError('Shared-v2 synthesis did not complete; preserve the failed candidate')
            from .shared_v2_protocol import validate_scaffold_file
            validate_scaffold_file(kwargs['new_global_dir'] / 'scaffold.yaml')
        except Exception:
            new = kwargs['new_global_dir']
            if new.exists():
                new.rename(kwargs['batch_dir'] / ('failed_shared_synthesis_' + uuid.uuid4().hex))
            raise
        return result
    legacy.run_batch_synthesis = synthesis
