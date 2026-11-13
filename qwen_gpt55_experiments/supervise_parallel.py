"""Adopt the existing remote sweep with parallel groups and a shared 48-agent target."""
import argparse
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

from .common.analysis_payloads import GROUPS, load_payloads
from .common.udocker_runtime import persistent_environment, runtime_root
from .run_pipeline import METHODS
from .supervise_remote import AdoptedProcess, now

TARGET_AGENT_CONTAINERS = 48


def atomic_json(path, data):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(data, indent=2))
    temporary.replace(path)


def pending_groups(state):
    running = {j['group'] for j in state['jobs'] if j['state'] == 'running'}
    if any(j['state'] == 'failed' for j in state['jobs']):
        return []
    started = {j['group'] for j in state['jobs']}
    return [g for g in GROUPS[1:] if g not in started][:max(0, state.get('group_concurrency', 2) - len(running))]


def legacy_running(state):
    return any(j['state'] == 'running' and not j.get('dynamic_capacity') for j in state['jobs'])


def local_memory_ok(reserve_gib):
    root = Path('/sys/fs/cgroup')
    limit = (root / 'memory.max').read_text().strip()
    return limit == 'max' or int(limit) - int((root / 'memory.current').read_text()) >= reserve_gib * 2**30


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--detach', action='store_true')
    parser.add_argument('--run-name', required=True)
    parser.add_argument('--group-concurrency', type=int, choices=(2, 3))
    args = parser.parse_args()
    if Path(args.run_name).name != args.run_name or args.run_name in ('.', '..'):
        parser.error('run-name must be a single directory name')
    os.environ.update(persistent_environment())
    package = Path(__file__).resolve().parent
    control = runtime_root().parent / 'remote_sweeps'
    sweep = control / args.run_name
    manifest = sweep / 'manifest.json'
    state = json.loads(manifest.read_text())
    if state['run_name'] != args.run_name or state['backend'] != 'remote_docker':
        raise RuntimeError('Unexpected sweep identity/backend')
    if args.detach:
        old = Path('/proc') / str(state['pid'])
        if old.exists() and (old / 'stat').read_text().rsplit(')', 1)[1].split()[0] != 'Z':
            command = (old / 'cmdline').read_bytes().decode().split('\0')
            if not any(name in command for name in ('qwen_gpt55_experiments.supervise_remote', 'qwen_gpt55_experiments.supervise_parallel')) or args.run_name not in command:
                raise RuntimeError('Expected the original supervisor; refusing to stop another process')
            os.kill(state['pid'], signal.SIGTERM)
            for _ in range(100):
                if not old.exists() or (old / 'stat').read_text().rsplit(')', 1)[1].split()[0] == 'Z':
                    break
                time.sleep(.1)
            else:
                raise RuntimeError('Original supervisor did not stop')
        with (control / 'supervisor.lock').open('a+') as lock:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
            atomic_json(sweep / ('manifest.before_parallel.' + stamp + '.json'), json.loads(manifest.read_text()))
            log = control / (args.run_name + '.parallel.' + stamp + '.log')
            with log.open('x') as output:
                proc = subprocess.Popen([sys.executable, '-B', '-m', 'qwen_gpt55_experiments.supervise_parallel',
                                         '--run-name', args.run_name, '--group-concurrency', str(args.group_concurrency or state.get('group_concurrency', 2))], cwd=package.parent,
                                        env=os.environ.copy(), stdin=subprocess.DEVNULL,
                                        stdout=output, stderr=subprocess.STDOUT, start_new_session=True)
            atomic_json(control / 'active.json', dict(pid=proc.pid, run_name=args.run_name,
                        manifest=str(manifest), log=str(log), started_at=now()))
        print(json.dumps({'pid': proc.pid, 'log': str(log)}), flush=True)
        return
    with (control / 'supervisor.lock').open('a+') as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        state = json.loads(manifest.read_text())
        state.setdefault('supervisor_history', []).append(dict(pid=state['pid'], replaced_at=now()))
        capacity = sweep / 'capacity.json'
        count = 16 if legacy_running(state) else state.get('max_agent_containers', 16)
        atomic_json(capacity, {'agent_containers': count})
        os.environ['EXPERIMENT_REMOTE_CAPACITY_FILE'] = str(capacity)
        state.update(pid=os.getpid(), group_concurrency=args.group_concurrency or state.get('group_concurrency', 2), target_agent_containers=TARGET_AGENT_CONTAINERS,
                     capacity_file=str(capacity), max_agent_containers=count)
        while True:
            for job in state['jobs']:
                if job['state'] == 'running':
                    code = AdoptedProcess(job).poll()
                    if code is not None:
                        job.update(state='complete' if code == 0 else 'failed', finished_at=now(),
                                   returncode=None, completion_detection='process exit plus native summary')
            running = [j for j in state['jobs'] if j['state'] == 'running']
            failed = any(j['state'] == 'failed' for j in state['jobs'])
            paused = (sweep / 'STOP_AFTER_GROUP').exists()
            recovery_limit = 1 if any(
                j['state'] == 'running' and j['method'] == 'harnessfix'
                for j in state['jobs']
            ) else 3
            recovery_running = sum(
                j['state'] == 'running' and bool(j.get('recovery_command'))
                for j in state['jobs']
            )
            if not failed and not paused and recovery_running < recovery_limit:
                for job in [j for j in state['jobs'] if j['state'] == 'recovery_queued'][:recovery_limit - recovery_running]:
                    command = job['recovery_command']
                    log = Path(job['recovery_log'])
                    log.parent.mkdir(parents=True, exist_ok=True)
                    with log.open('a') as output:
                        child = subprocess.Popen(
                            command,
                            cwd=package.parent,
                            env=os.environ.copy(),
                            stdin=subprocess.DEVNULL,
                            stdout=output,
                            stderr=subprocess.STDOUT,
                            start_new_session=True,
                        )
                    job.update(
                        pid=child.pid,
                        state='running',
                        command=command,
                        log=str(log),
                        started_at=now(),
                        finished_at=None,
                        returncode=None,
                        dynamic_capacity=True,
                    )
                    atomic_json(manifest, state)
                running = [j for j in state['jobs'] if j['state'] == 'running']
            state.update(state='running', active_groups=sorted({j['group'] for j in running}))
            state.pop('active_group', None)
            if failed:
                state['state'] = 'draining_after_failure' if running else 'needs_attention'
            elif paused:
                state['state'] = 'draining_before_pause' if running else 'paused_before_next_group'
            elif len(state['jobs']) == 30 and all(j['state'] == 'complete' for j in state['jobs']):
                state['state'] = 'complete'
            elif not running and any(j['state'] == 'recovery_queued' for j in state['jobs']):
                state['state'] = 'recovery_queued'
            state['capacity_transition'] = 'waiting_for_legacy_workers' if legacy_running(state) else 'ready'
            if not failed and not paused:
                try:
                    from .common.remote_runtime import RemoteRuntime
                    probe = RemoteRuntime.__new__(RemoteRuntime)
                    probe.host, probe.local_dir = 'server241', sweep
                    if count < TARGET_AGENT_CONTAINERS and not legacy_running(state) and local_memory_ok(10):
                        probe.guard()
                        count = TARGET_AGENT_CONTAINERS
                        atomic_json(capacity, {'agent_containers': count})
                    state['max_agent_containers'] = count
                    for group in pending_groups(state):
                        if not local_memory_ok(12):
                            state['admission_wait'] = 'local memory reserve below 12 GiB'
                            break
                        probe.guard()
                        if len(load_payloads(group)) != 148:
                            raise RuntimeError('Expected 148 analysis payloads')
                        if any((package / m / group / args.run_name).exists() for m in METHODS):
                            raise RuntimeError('Refusing to overwrite an existing group')
                        for method in METHODS:
                            command = [sys.executable, '-B', '-m', 'qwen_gpt55_experiments.run_remote',
                                       '--method', method, '--group', group, '--run-name', args.run_name, '--workers', '8']
                            log = sweep / f'{group}__{method}.log'
                            with log.open('x') as output:
                                child = subprocess.Popen(command, cwd=package.parent, env=os.environ.copy(),
                                    stdin=subprocess.DEVNULL, stdout=output, stderr=subprocess.STDOUT, start_new_session=True)
                            state['jobs'].append(dict(method=method, group=group, pid=child.pid, command=command,
                                log=str(log), run_dir=str(package / method / group / args.run_name),
                                started_at=now(), state='running', dynamic_capacity=True))
                            atomic_json(manifest, state)
                        state.pop('admission_wait', None)
                except Exception as exc:
                    state['admission_wait'] = f'{type(exc).__name__}: {exc}'
            state.update(updated_at=now(), active_groups=sorted({j['group'] for j in state['jobs'] if j['state'] == 'running'}))
            atomic_json(manifest, state)
            if state['state'] in ('needs_attention', 'paused_before_next_group', 'complete'):
                return
            # Reap our children; adopted jobs are checked through /proc above.
            while True:
                try:
                    pid, _ = os.waitpid(-1, os.WNOHANG)
                    if pid == 0:
                        break
                except ChildProcessError:
                    break
            time.sleep(30)


if __name__ == '__main__':
    main()
