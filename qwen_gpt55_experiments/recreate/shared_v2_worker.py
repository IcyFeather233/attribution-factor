"""Isolated v2.4.6 task rollout; the parent retains container ownership for analysis."""
import argparse
import json
import logging
from pathlib import Path
import sys

from .shared_v2_protocol import PROTOCOL, ROOT, V2_SOURCE, task_overrides

sys.path.insert(0, str(V2_SOURCE))
import minisweagent
from minisweagent.agents.default import DefaultAgent
from qwen_gpt55_experiments.common.runner import build_model, load_original_config, bounded_task_wall_time, redact_secret
from qwen_gpt55_experiments.common.remote_runtime import RemoteRuntime
from qwen_gpt55_experiments.common.remote_environment import RemoteDockerEnvironment
from qwen_gpt55_experiments.common.udocker_environment import UdockerEnvironmentConfig
from qwen_gpt55_experiments.common.model_config import task_api_key, TASK_MODEL


def require_v2():
    if minisweagent.__version__ != '2.4.6' or not Path(minisweagent.__file__).resolve().is_relative_to(V2_SOURCE):
        raise RuntimeError('Shared task worker imported the wrong mini-swe-agent runtime')


class BorrowedRemoteEnvironment(RemoteDockerEnvironment):
    def cleanup(self):
        """The parent owns this container, including during optimizer inspection."""
        pass


def attach_environment(request):
    original = load_original_config()['environment']
    env = BorrowedRemoteEnvironment.__new__(BorrowedRemoteEnvironment)
    env.logger = logging.getLogger(__name__)
    env.runtime = RemoteRuntime.attach(request['container_name'])
    if env.runtime.mounts != {request['workspace']: '/workspace'}:
        raise RuntimeError('Task worker mount differs from parent-owned workspace')
    env.config = UdockerEnvironmentConfig(
        image=request['image'], repo=ROOT / 'native_baseline_runs/udocker',
        executable=ROOT / 'native_baseline_runs/.venv/bin/udocker',
        cwd=original['cwd'], env=original['env'], timeout=original['timeout'],
        interpreter=original['interpreter'], volumes=env.runtime.mounts,
        container_name=env.runtime.container_name, keep_container=True)
    env.container_name = env.runtime.container_name
    env.containers_dir = env.runtime.local_dir
    env.scratch_volumes = {}
    return env


def run_task(request, *, environment=None, model=None):
    require_v2()
    if request.get('protocol') != PROTOCOL:
        raise ValueError('Wrong task worker protocol')
    output = Path(request['output_dir']).resolve()
    if not output.is_relative_to(ROOT / 'qwen_gpt55_experiments'):
        raise ValueError('Task outputs must remain in persistent experiment storage')
    output.mkdir(parents=True, exist_ok=True)
    key = task_api_key()
    agent_updates, model_updates = task_overrides(request['scaffold'])
    config = load_original_config()['agent']
    config.update(agent_updates)
    config['step_limit'] = min(250, int(request['step_limit']))
    if config['step_limit'] < 1:
        raise ValueError('Task step limit must be positive')
    config['wall_time_limit_seconds'] = min(3600, bounded_task_wall_time(config))
    canonical = output / 'trajectory.v2.json'
    config['output_path'] = canonical
    env = environment if environment is not None else attach_environment(request)
    task_model = model if model is not None else build_model(
        api_key=key, temperature=1.0, model_overrides=model_updates)
    agent = DefaultAgent(model=task_model, env=env, **config)
    try:
        result = agent.run(request['problem_statement'])
        if result.get('exit_status') not in {'Submitted', 'LimitsExceeded', 'TimeExceeded', 'RepeatedFormatError'}:
            raise RuntimeError('Task rollout ended without a gradeable terminal state: ' + str(result.get('exit_status')))
        patch = result.get('submission', '') or ''
        if agent.n_calls and not patch.strip():
            diff = env.execute({'command': 'git -c core.fileMode=false diff'}, timeout=120)
            if diff['returncode'] == 0:
                patch = diff['output']
        agent.save(canonical, {'instance_id': request['instance_id'],
                              'info': {'submission': patch, 'shared_task_protocol': PROTOCOL}})
        reply = {'protocol': PROTOCOL, 'instance_id': request['instance_id'],
                 'task_runtime': '2.4.6', 'task_model': TASK_MODEL,
                 'exit_status': result['exit_status'], 'submission': patch,
                 'trajectory': str(canonical), 'container_name': request['container_name']}
        (output / 'task_worker_result.json').write_text(json.dumps(reply, indent=2) + '\n')
        return reply
    finally:
        # Parent keeps the same container alive for the native environment-inspection stage.
        # It alone cleans the container and releases its resident-slot lock.
        redact_secret(canonical, key)
        redact_secret(output / 'task_worker_result.json', key)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--request', type=Path, required=True)
    args = parser.parse_args()
    run_task(json.loads(args.request.read_text()))


if __name__ == '__main__':
    main()
