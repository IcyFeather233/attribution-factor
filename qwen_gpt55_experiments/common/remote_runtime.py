"""Native Docker over persistent SSH; no dependency on a mini-swe-agent version.

Only verified cached images are used. All remote scratch is on Docker disk
volumes, and all local transfer files/sockets are under /volume/med. Workspace
mounts are synchronized at command boundaries, including the ReCreate sidecar.
"""
from __future__ import annotations

import fcntl
import hashlib
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import tarfile
import tempfile
from types import SimpleNamespace
import uuid
import time
from contextlib import contextmanager

from .udocker_runtime import persistent_environment, runtime_root, make_private_containers_dir, cleanup_private_containers_dir, create_slot

LABEL = 'mzchen.failure_attribution_runtime'


class RemoteInfrastructureError(RuntimeError):
    """An invalid run caused by transport/resource failure, not a failed patch."""


def remote_status_command(command, token):
    # The outer SSH command exits zero after reporting docker exec's independent
    # status. No task command is retried here: execution may have side effects.
    script = (
        "import subprocess,sys\n"
        "p=subprocess.Popen(sys.argv[2:],stdout=subprocess.PIPE,stderr=subprocess.STDOUT)\n"
        "while True:\n"
        " chunk=p.stdout.read1(65536)\n"
        " if not chunk: break\n"
        " sys.stdout.buffer.write(chunk); sys.stdout.buffer.flush()\n"
        "p.stdout.close()\n"
        "p.wait()\n"
        "code=p.returncode if p.returncode>=0 else 128-p.returncode\n"
        "sys.stdout.write('\\n__FA_EXEC_STATUS_'+sys.argv[1]+'__:'+str(code)+'\\n')\n"
        "sys.stdout.flush()\n"
    )
    return ['python3', '-c', script, token, *command]


def decode_remote_status(result, command, token):
    marker = '\n__FA_EXEC_STATUS_' + token + '__:'
    output, separator, status = str(result.stdout or '').rpartition(marker)
    if result.returncode != 0 or not separator or not status.endswith('\n') or not status[:-1].isdigit():
        raise RemoteInfrastructureError(
            f'Remote execution completion unconfirmed (SSH exit={result.returncode}); '
            f'output tail: {str(result.stdout or "")[-1000:]}')
    code = int(status[:-1])
    if not 0 <= code <= 255:
        raise RemoteInfrastructureError('Invalid remote command status envelope')
    return subprocess.CompletedProcess(command, code, output, getattr(result, 'stderr', None))


def snapshot(root):
    result = {}
    for path in sorted(Path(root).rglob('*')):
        relative = str(path.relative_to(root))
        if path.is_symlink():
            target = path.resolve()
            if not target.is_relative_to(Path(root).resolve()):
                raise ValueError(f'Workspace symlink escapes its mount: {relative}')
            result[relative] = ['link', os.readlink(path)]
        elif path.is_file():
            with path.open('rb') as handle:
                digest = hashlib.file_digest(handle, 'sha256').hexdigest()
            result[relative] = ['file', digest, path.stat().st_mode & 0o777]
        elif path.is_dir():
            result[relative] = ['dir']
        else:
            raise ValueError(f'Unsupported workspace file: {relative}')
    return result


def merge_workspace(local, staged, before):
    """Apply remote changes only; preserve concurrently written host logs."""
    local, staged = Path(local), Path(staged)
    after, current = snapshot(staged), snapshot(local)
    changed = {key for key in set(before) | set(after) if before.get(key) != after.get(key)}
    for key in changed:
        if current.get(key) not in (before.get(key), after.get(key)):
            raise RuntimeError(f'Concurrent local/remote workspace edit: {key}; remote copy retained in {staged}')
    for key in sorted(changed, key=lambda x: (-x.count('/'), x)):
        target = local / key
        if target.is_symlink() or target.is_file():
            target.unlink()
        elif target.is_dir() and after.get(key, [''])[0] != 'dir':
            # Never recursively erase a host directory that acquired new files.
            target.rmdir()
    for key in sorted(changed, key=lambda x: (x.count('/'), x)):
        if key not in after:
            continue
        src, dst = staged / key, local / key
        dst.parent.mkdir(parents=True, exist_ok=True)
        if src.is_symlink():
            dst.symlink_to(os.readlink(src))
        elif src.is_dir():
            dst.mkdir(exist_ok=True)
        else:
            shutil.copy2(src, dst)
    return after


class RemoteRuntime:
    def _remove_remote_resource(self, args, *, missing_markers, attempts=5):
        """Remove one owned Docker resource despite transient daemon races."""
        last_error = ""
        for attempt in range(attempts):
            try:
                removed = self.ssh(args, check=False)
                output = str(removed.stdout or "")
                lowered = output.lower()
                if removed.returncode == 0 or any(marker in lowered for marker in missing_markers):
                    return
                last_error = f"exit={removed.returncode}: {output[-2000:]}"
            except (OSError, subprocess.SubprocessError, RemoteInfrastructureError) as exc:
                last_error = f"{type(exc).__name__}: {exc}"
            if attempt + 1 < attempts:
                time.sleep(min(2 ** attempt, 5))
        raise RuntimeError(last_error or "remote resource removal failed without output")

    @contextmanager
    def admission(self):
        """Serialize count-and-create; survivors consume capacity until stopped."""
        deadline = time.monotonic() + float(os.environ.get('EXPERIMENT_CAPACITY_WAIT_SECONDS', '7200'))
        with (runtime_root().parent / 'remote_admission.lock').open('a+') as lock:
            while True:
                if time.monotonic() >= deadline:
                    raise RemoteInfrastructureError('Remote capacity wait timed out; task was not executed')
                try:
                    fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError:
                    time.sleep(.25)
                    continue
                try:
                    setting = 'UDOCKER_MAX_EVALUATORS' if self.role == 'evaluator' else 'UDOCKER_MAX_CONTAINERS'
                    limit = int(os.environ.get(setting, '4' if self.role == 'evaluator' else '16'))
                    if self.role == 'agent':
                        from .remote_capacity import agent_capacity
                        limit = agent_capacity(limit)
                    active = self.ssh(['docker', 'ps', '-q', '--filter', f'label={LABEL}.role={self.role}']).stdout.splitlines()
                    if len(active) < limit:
                        yield
                        return
                finally:
                    fcntl.flock(lock.fileno(), fcntl.LOCK_UN)
                time.sleep(1)

    def __init__(self, config):
        self.config = config
        self.host = os.environ.get('EXPERIMENT_REMOTE_HOST', 'server241')
        if self.host != 'server241':
            raise ValueError('This storage-verified backend currently supports server241 only')
        self.container_name = 'fa-' + uuid.uuid4().hex
        self.local_dir = make_private_containers_dir(config.container_name or 'agent-remote')
        self.volumes = []
        self.mounts = dict(config.volumes)
        self.closed = False
        self.created = False
        self.sync_failed = False
        self.failed = False
        self.role = 'evaluator' if config.container_name.startswith(('eval-', 'recreate-eval-')) else 'agent'
        self.image = ''
        self.registry = runtime_root().parent / 'remote_registry'
        self.registry.mkdir(parents=True, exist_ok=True)
        self.receipt = self.registry / (self.container_name + '.json')
        self.task_dir = os.environ.get('EXPERIMENT_TASK_OUTPUT_DIR')
        self.owner_pid = os.getpid()
        try:
            for host, target in self.mounts.items():
                if not Path(host).resolve().is_relative_to('/volume/med') or not Path(host).is_dir():
                    raise ValueError('Remote workspaces must be directories on /volume/med')
                if target != '/workspace' and not target.startswith('/workspace/'):
                    raise ValueError('Remote writable asset mounts must be under /workspace')
                if '..' in Path(target).parts:
                    raise ValueError('Invalid workspace mount')
            image_dir = Path(config.repo) / 'repos' / config.image.rsplit(':', 1)[0] / config.image.rsplit(':', 1)[1]
            self.image = json.loads((image_dir / 'manifest').read_text())['config']['digest']
            with self.admission(), create_slot():
                self.guard()
                info = json.loads(self.ssh(['docker', 'image', 'inspect', self.image]).stdout)[0]
                expected = json.loads((image_dir / self.image).read_text())['rootfs']['diff_ids']
                if info['RootFS']['Layers'] != expected:
                    raise RuntimeError('Remote image layers differ from the local reference')
                args = ['docker', 'run', '-d', '--name', self.container_name, '--label', f'{LABEL}={self.container_name}',
                        '--label', f'{LABEL}.role={self.role}',
                        '--memory', os.environ.get('EXPERIMENT_REMOTE_MEMORY', '8g'), '--memory-swap', os.environ.get('EXPERIMENT_REMOTE_MEMORY', '8g'),
                        '--cpus', os.environ.get('EXPERIMENT_REMOTE_CPUS', '4'), '--pids-limit', '512', '--log-driver', 'none',
                        '--workdir', config.cwd, '--entrypoint', '/bin/bash']
                for index, target in enumerate(['/tmp', '/var/tmp', '/dev/shm', *self.mounts.values()]):
                    volume = self.container_name + f'-v{index}'
                    self.volumes.append(volume)
                    self.save()
                    self.ssh(['docker', 'volume', 'create', '--label', f'{LABEL}={self.container_name}', volume])
                    args += ['--mount', f'type=volume,src={volume},dst={target}']
                args += [self.image, '-c', 'sleep 21600']
                self.created = True
                self.save()
                self.ssh(args)
                limit = int(os.environ.get('UDOCKER_MAX_EVALUATORS' if self.role == 'evaluator' else 'UDOCKER_MAX_CONTAINERS', '4' if self.role == 'evaluator' else '16'))
                if self.role == 'agent':
                    from .remote_capacity import agent_capacity
                    limit = agent_capacity(limit)
                active = self.ssh(['docker', 'ps', '-q', '--filter', f'label={LABEL}.role={self.role}']).stdout.splitlines()
                if len(active) > limit:
                    raise RuntimeError('Remote role capacity exceeded, possibly by survivors of a local restart; inspect registered containers')
            self.save()
            self.start_watchdog()
            for host, target in self.mounts.items():
                self.push(host, target)
        except BaseException:
            self.cleanup()
            raise

    def save(self):
        temporary = self.receipt.with_suffix('.tmp')
        temporary.write_text(json.dumps({'host': self.host, 'container_name': self.container_name,
            'task_dir': getattr(self, 'task_dir', None), 'owner_pid': getattr(self, 'owner_pid', None),
            'local_dir': str(self.local_dir), 'volumes': self.volumes, 'mounts': self.mounts,
            'image': self.image, 'cwd': self.config.cwd, 'timeout': self.config.timeout,
            'sync_failed': self.sync_failed, 'failed': self.failed,
            'failure_reason': getattr(self, 'failure_reason', None)}))
        temporary.replace(self.receipt)

    @classmethod
    def attach(cls, name):
        if not name.startswith('fa-') or len(name) != 35 or any(c not in '0123456789abcdef' for c in name[3:]):
            raise ValueError('Invalid owned container name')
        receipt = runtime_root().parent / 'remote_registry' / (name + '.json')
        data = json.loads(receipt.read_text())
        obj = cls.__new__(cls)
        obj.__dict__.update(data)
        obj.sync_failed = data.get('sync_failed', False)
        obj.failed = data.get('failed', False)
        obj.local_dir = Path(data['local_dir'])
        obj.receipt, obj.closed, obj.created = receipt, False, True
        obj.config = SimpleNamespace(cwd=data['cwd'], timeout=data['timeout'])
        return obj

    def ssh(self, args, *, timeout=120, input_text=None, check=True, stdin=None, stdout=None):
        env = os.environ.copy()
        env.update(persistent_environment())
        command = ['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=10', '-o', 'ServerAliveInterval=15',
                   '-o', 'ServerAliveCountMax=3', '-o', 'ControlMaster=auto', '-o', 'ControlPersist=30',
                   '-o', 'ControlPath=./ssh-control', self.host, shlex.join(args)]
        text_mode = stdin is None and stdout is None
        # Commands can emit arbitrary bytes (e.g. grep against repository fixtures).
        # Keep archive transfer streams binary; replace undecodable observation bytes.
        decoding = {'encoding': 'utf-8', 'errors': 'replace'} if text_mode else {}
        result = subprocess.run(command, cwd=self.local_dir, env=env, input=input_text, stdin=stdin,
            stdout=stdout or subprocess.PIPE, stderr=subprocess.PIPE if stdout else subprocess.STDOUT,
            text=text_mode, timeout=timeout, **decoding)
        if check and result.returncode:
            raise RemoteInfrastructureError(f'Remote transport failed ({result.returncode}): {str(result.stdout or result.stderr)[-2000:]}')
        return result

    def guard(self):
        script = "import shutil,json; m=dict(x.split(':',1) for x in open('/proc/meminfo')); print(json.dumps([shutil.disk_usage('/var/lib/docker').free,int(m['MemAvailable'].split()[0])*1024]))"
        disk, memory = json.loads(self.ssh(['python3', '-c', script]).stdout)
        if disk < int(os.environ.get('EXPERIMENT_REMOTE_MIN_DISK_GIB', '100')) * 2**30:
            raise RuntimeError('Remote disk below admission threshold; no new container admitted')
        if memory < int(os.environ.get('EXPERIMENT_REMOTE_MIN_MEMORY_GIB', '64')) * 2**30:
            raise RuntimeError('Remote available memory below admission threshold')

    def start_watchdog(self):
        # Runs on the remote host even if the local development container restarts.
        # Only stops this UUID-labelled container; volumes remain recoverable.
        code = '''import json,shutil,subprocess,time,sys
name,label,floor=sys.argv[1],sys.argv[2],int(sys.argv[3])
for _ in range(4320):
 p=subprocess.run(['docker','inspect',name],stdout=subprocess.PIPE,stderr=subprocess.DEVNULL)
 if p.returncode: break
 d=json.loads(p.stdout)[0]
 if d['Config']['Labels'].get(label)!=name or not d['State']['Running']: break
 memory = dict(x.split(':',1) for x in open('/proc/meminfo'))
 if shutil.disk_usage('/var/lib/docker').free < floor or int(memory['MemAvailable'].split()[0])*1024 < 16*2**30:
  subprocess.run(['docker','kill',name],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
  break
 time.sleep(5)
'''
        launcher = 'import subprocess,sys; subprocess.Popen(sys.argv[1:],stdin=subprocess.DEVNULL,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,start_new_session=True)'
        self.ssh(['python3', '-c', launcher, 'python3', '-c', code, self.container_name, LABEL,
                  str(int(os.environ.get('EXPERIMENT_REMOTE_STOP_DISK_GIB', '80')) * 2**30)])

    def push(self, host, target):
        # Transfers stream through a persistent file; no rootfs or tar in RAM.
        archive = self.local_dir / 'workspace-upload.tar'
        snapshot(host)  # Validate special files and escaping symlinks first.
        with tarfile.open(archive, 'w') as tar:
            tar.add(host, arcname='.')
        # A task's background process may remove an entry after find enumerates
        # it. Ignore only that ENOENT race; permission/transport errors still fail.
        self.ssh(['docker', 'exec', self.container_name, 'find', target,
                  '-mindepth', '1', '-ignore_readdir_race', '-delete'])
        with archive.open('rb') as handle:
            self.ssh(['docker', 'cp', '-', self.container_name + ':' + target], stdin=handle)
        archive.unlink()

    def pull(self, host, target, before):
        try:
            self._pull(host, target, before)
        except Exception as exc:
            self.sync_failed = True
            self.save()
            raise RemoteInfrastructureError(f'Workspace synchronization failed; recovery receipt: {self.receipt}') from exc

    def _pull(self, host, target, before):
        archive = self.local_dir / 'workspace-download.tar'
        with archive.open('wb') as handle:
            self.ssh(['docker', 'cp', self.container_name + ':' + target + '/.', '-'], stdout=handle)
        staged = Path(tempfile.mkdtemp(prefix='workspace-', dir=self.local_dir))
        with tarfile.open(archive) as tar:
            tar.extractall(staged, filter='data')
        merge_workspace(host, staged, before)
        shutil.rmtree(staged)
        archive.unlink()

    def run(self, args, *, timeout, input_text=None, check=False):
        try:
            return self._run(args, timeout=timeout, input_text=input_text, check=check)
        except RemoteInfrastructureError as exc:
            self.failed = True
            self.failure_reason = str(exc)
            if hasattr(self, 'receipt'):
                self.save()
            raise

    def _run(self, args, *, timeout, input_text=None, check=False):
        if getattr(self, 'failed', False):
            raise RemoteInfrastructureError('This container run was invalidated by an infrastructure failure')
        if self.sync_failed:
            raise RemoteInfrastructureError(f'Workspace recovery required; see {self.receipt}')
        if args[0] == 'rm':
            self.cleanup()
            return subprocess.CompletedProcess(args, 0, '', None)
        if args[:2] != ['run', '--nobanner']:
            raise ValueError('Unsupported remote transport operation')
        index, workdir, environment = 2, self.config.cwd, []
        while index < len(args) and args[index].startswith('--'):
            arg = args[index]
            if arg.startswith('--workdir='): workdir = arg.split('=', 1)[1]
            elif arg.startswith('--env='): environment += ['-e', arg.split('=', 1)[1]]
            elif arg.startswith('--volume='):
                host, target = arg.split('=', 1)[1].rsplit(':', 1)
                if self.mounts.get(host) != target:
                    raise ValueError('Unregistered remote mount')
            else: raise ValueError(f'Unsupported remote option: {arg}')
            index += 1
        if args[index] != self.container_name:
            raise ValueError('Command does not target its private remote container')
        with (self.local_dir / 'command.lock').open('a+') as lock:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
            before = {host: snapshot(host) for host in self.mounts}
            for host, target in self.mounts.items():
                self.push(host, target)
            command = ['docker', 'exec', '-i', '-w', workdir, *environment, self.container_name,
                       'timeout', '--signal=TERM', '--kill-after=5', str(timeout), *args[index + 1:]]
            try:
                token = uuid.uuid4().hex
                raw = self.ssh(remote_status_command(command, token), timeout=timeout + 30,
                               input_text=input_text, check=False)
                result = decode_remote_status(raw, command, token)
                if result.returncode not in (0, 124):
                    inspected = self.ssh(['docker', 'inspect', '--format', '{{json .State}}', self.container_name], check=False)
                    if inspected.returncode or not json.loads(inspected.stdout)['Running']:
                        raise RemoteInfrastructureError(f'Container stopped or disappeared: {inspected.stdout[-1000:]}')
            finally:
                for host, target in self.mounts.items():
                    self.pull(host, target, before[host])
            if result.returncode in (124, 137):
                raise subprocess.TimeoutExpired(command, timeout, output=result.stdout)
            if check and result.returncode:
                raise subprocess.CalledProcessError(result.returncode, command, output=result.stdout)
            return result

    def cleanup(self):
        if self.closed:
            return
        if self.created:
            if self.sync_failed:
                self.ssh(['docker', 'stop', '-t', '5', self.container_name], check=False)
                raise RuntimeError(f'Workspace sync failed: retaining container, volumes, and transfer files for recovery: {self.receipt}')
            try:
                self._remove_remote_resource(
                    ['docker', 'rm', '-f', self.container_name],
                    missing_markers=('no such container',),
                )
            except RuntimeError as exc:
                raise RuntimeError(f'Remote cleanup failed ({exc}); recovery receipt: {self.receipt}') from exc
            self.created = False
        for volume in list(self.volumes):
            try:
                self._remove_remote_resource(
                    ['docker', 'volume', 'rm', volume],
                    missing_markers=('no such volume',),
                )
            except RuntimeError as exc:
                raise RuntimeError(f'Remote volume cleanup failed ({exc}); recovery receipt: {self.receipt}') from exc
            self.volumes.remove(volume)
        self.receipt.unlink(missing_ok=True)
        self.closed = True
        # Close SSH master before deleting its socket directory.
        subprocess.run(['ssh', '-o', 'ControlPath=./ssh-control', '-O', 'exit', self.host],
                       cwd=self.local_dir, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=15)
        cleanup_private_containers_dir(self.local_dir)
