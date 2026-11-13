"""Process-isolated Meta-Harness evaluations with a finite wall-clock budget."""
import json
import os
from pathlib import Path
import signal
import subprocess
import sys

from attribution_experiments.common.udocker_runtime import persistent_environment, runtime_root
from attribution_experiments.common.result_validity import retryable_result


def run_bounded(command, *, env, log, timeout, before_terminate=None):
    with log.open('a') as output:
        child = subprocess.Popen(command, env=env, stdout=output, stderr=subprocess.STDOUT,
                                 stdin=subprocess.DEVNULL, start_new_session=True)
        try:
            return child.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            if before_terminate is not None:
                # Freeze the whole task tree so it cannot create more containers.
                # Keep its slot locks held until remote shutdown is confirmed.
                os.killpg(child.pid, signal.SIGSTOP)
                try:
                    before_terminate()
                except Exception as exc:
                    # Do not silently release slots when remote cleanup is unknown.
                    # The stopped task remains available for explicit recovery.
                    (log.parent / 'cleanup_blocked.json').write_text(json.dumps({
                        'worker_pid': child.pid, 'state': 'frozen_holding_slots', 'error': str(exc)}))
                    raise
                os.killpg(child.pid, signal.SIGKILL)
                child.wait()
                return 124
            os.killpg(child.pid, signal.SIGTERM)
            try:
                child.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid, signal.SIGKILL)
                child.wait()
            # The leader can exit on TERM while a descendant ignores it.
            try:
                os.killpg(child.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            return 124


def stop_owned(task_dir):
    from attribution_experiments.common.remote_runtime import RemoteRuntime, LABEL
    registry = runtime_root().parent / 'remote_registry'
    for receipt in registry.glob('*.json'):
        try:
            data = json.loads(receipt.read_text())
        except FileNotFoundError:
            continue  # An unrelated task completed during the directory scan.
        if data.get('task_dir') != str(task_dir.resolve()) and not any(Path(p).is_relative_to(task_dir.resolve()) for p in data['mounts']):
            continue
        runtime = RemoteRuntime.attach(data['container_name'])
        inspected = runtime.ssh(['docker', 'inspect', runtime.container_name], check=False)
        if inspected.returncode:
            if 'No such' in inspected.stdout:
                continue
            raise RuntimeError('Unable to verify owned container state')
        info = json.loads(inspected.stdout)[0]
        if info['Config']['Labels'].get(LABEL) != runtime.container_name:
            raise RuntimeError('Container ownership mismatch')
        runtime.ssh(['docker', 'stop', '-t', '5', runtime.container_name])
        state = runtime.ssh(['docker', 'inspect', '--format', '{{.State.Running}}', runtime.container_name])
        if state.stdout.strip() != 'false':
            raise RuntimeError('Owned container did not stop; refusing to release task slots')


def run_isolated(instance, output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env.update(persistent_environment())
    env['EXPERIMENT_TASK_OUTPUT_DIR'] = str(output_dir.resolve())
    timeout = int(env.get('EXPERIMENT_TASK_TIMEOUT_SECONDS', '7200'))
    if timeout <= 0:
        raise ValueError('Task timeout must be positive')
    code = run_bounded([sys.executable, '-B', '-m', __name__, str(output_dir.resolve()),
                        instance['instance_id']], env=env, log=output_dir / 'worker.log', timeout=timeout,
                       before_terminate=lambda: stop_owned(output_dir))
    if code == 0:
        return json.loads((output_dir / 'result.json').read_text())
    stop_owned(output_dir)
    result_path = output_dir / 'result.json'
    if result_path.exists():
        try:
            result = json.loads(result_path.read_text())
        except (OSError, json.JSONDecodeError):
            result = None
        if isinstance(result, dict) and retryable_result(result):
            (output_dir / 'retryable.json').write_text(json.dumps({
                'instance_id': instance['instance_id'],
                'exit_status': result.get('exit_status'),
                'exception': result.get('exception'),
                'retryable': True,
            }, indent=2))
            raise RuntimeError(f"Retryable task infrastructure failure: {instance['instance_id']}")
    status = 'TaskTimeout' if code == 124 else 'TaskExecutionError'
    skipped = {'instance_id': instance['instance_id'], 'exit_status': status,
               'exception': status, 'skipped': True, 'count_as_failure': True, 'timeout_seconds': timeout}
    (output_dir / 'skip.json').write_text(json.dumps(skipped, indent=2))
    return skipped


if __name__ == '__main__':
    from attribution_experiments.common.dataset import load_failed_instances
    from attribution_experiments.common.runner import run_task
    from attribution_experiments.meta_harness.run_native import validate_candidate, UDOCKER_REPO, UDOCKER_EXECUTABLE
    directory = Path(sys.argv[1])
    instance = next(x for x in load_failed_instances() if x['instance_id'] == sys.argv[2])
    _, scaffold, agent_class, assets = validate_candidate(directory.parent.parent / 'search')
    result = run_task(instance=instance, output_dir=directory, udocker_repo=UDOCKER_REPO,
                      udocker_executable=UDOCKER_EXECUTABLE, agent_overrides=scaffold['agent'],
                      model_overrides=scaffold['model'], agent_class=agent_class, harness_assets_dir=assets)
    if result.get('exception'):
        raise SystemExit(2)
