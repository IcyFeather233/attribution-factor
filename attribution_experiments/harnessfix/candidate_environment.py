"""Keep candidate Docker behavior, replacing only its container transport."""

from __future__ import annotations

import importlib.util
import subprocess
import uuid
from .runtime_protocol import PROTOCOL
from pathlib import Path


def candidate_environment_class(candidate_dir: Path, runtime_class=None):
    if runtime_class is None:
        from attribution_experiments.common.execution_environment import ExecutionEnvironment as UdockerEnvironment

        runtime_class = UdockerEnvironment
    source = candidate_dir / "src/minisweagent/environments/docker.py"
    spec = importlib.util.spec_from_file_location(f"harnessfix_environment_{uuid.uuid4().hex}", source)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load candidate environment: {source}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    runtimes = {}

    class Transport:
        def __getattr__(self, name):
            return getattr(subprocess, name)

        def run(self, command, **kwargs):
            if not isinstance(command, (list, tuple)) or len(command) < 2 or command[1] != "exec":
                raise RuntimeError("Candidate environment requested unsupported container transport")
            index = 2
            workdir = None
            environment = []
            while index < len(command) and command[index] in ("-w", "-e"):
                flag, value = command[index:index + 2]
                if flag == "-w":
                    workdir = value
                else:
                    environment.append(value)
                index += 2
            if index >= len(command) or command[index] not in runtimes:
                raise RuntimeError("Candidate command does not target its private container")
            runtime = runtimes[command[index]]
            allowed = {"text", "encoding", "errors", "stdout", "stderr", "timeout", "check", "capture_output"}
            if set(kwargs) - allowed or kwargs.get("text") is not True:
                raise RuntimeError("Unsupported candidate subprocess options")
            args = ["run", "--nobanner", f"--workdir={workdir or runtime.config.cwd}"]
            args.extend(f"--env={value}" for value in environment)
            for host, container in {**runtime.config.volumes, **runtime.scratch_volumes}.items():
                args.append(f"--volume={host}:{container}")
            args.extend([runtime.container_name, *command[index + 1:]])
            result = runtime._run_host(args, timeout=kwargs.get("timeout", runtime.config.timeout))
            if kwargs.get("check") and result.returncode:
                raise subprocess.CalledProcessError(result.returncode, command, output=result.stdout)
            return subprocess.CompletedProcess(command, result.returncode, result.stdout, result.stderr)

    # Only this candidate module sees the proxy, never global subprocess.
    module.subprocess = Transport()

    class CandidateUdockerEnvironment(module.DockerEnvironment):
        def __init__(self, *, repo, executable, container_name, **kwargs):
            self._runtime_options = dict(repo=repo, executable=executable, container_name=container_name)
            self._runtime = None
            super().__init__(**kwargs)

        def _start_container(self):
            self._runtime = runtime_class(
                **self._runtime_options,
                image=self.config.image,
                cwd=self.config.cwd,
                env=self.config.env,
                forward_env=self.config.forward_env,
                timeout=self.config.timeout,
                pull_timeout=self.config.pull_timeout,
                interpreter=self.config.interpreter,
            )
            self.container_id = self._runtime.container_name
            runtimes[self.container_id] = self._runtime

        def cleanup(self):
            runtime = getattr(self, "_runtime", None)
            if runtime is not None:
                runtimes.pop(self.container_id, None)
                self.container_id = None
                self._runtime = None
                runtime.cleanup()

        def execute(self, *args, **kwargs):
            result = super().execute(*args, **kwargs)
            transport = getattr(getattr(self, '_runtime', None), 'runtime', None)
            if transport is not None and transport.failed:
                from attribution_experiments.common.remote_runtime import RemoteInfrastructureError
                raise RemoteInfrastructureError('Candidate execution encountered an infrastructure failure; retry this task: '
                                                + str(getattr(transport, 'failure_reason', 'unknown cause')))
            return result

        def serialize(self):
            data = super().serialize()
            data["info"]["config"]["container_transport"] = PROTOCOL
            data["info"]["config"]["candidate_environment_source"] = str(source)
            return data

    return CandidateUdockerEnvironment
