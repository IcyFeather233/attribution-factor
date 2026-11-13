# Regression fixture: the original HarnessFix baseline submission repair.
import logging
import os
import platform
import shlex
import subprocess
import uuid
from typing import Any

from pydantic import BaseModel

from minisweagent.exceptions import Submitted
from minisweagent.utils.serialize import recursive_merge


class DockerEnvironmentConfig(BaseModel):
    image: str
    cwd: str = "/"
    """Working directory in which to execute commands."""
    env: dict[str, str] = {}
    """Environment variables to set in the container."""
    forward_env: list[str] = []
    """Environment variables to forward to the container.
    Variables are only forwarded if they are set in the host environment.
    In case of conflict with `env`, the `env` variables take precedence.
    """
    timeout: int = 30
    """Timeout for executing commands in the container."""
    executable: str = os.getenv("MSWEA_DOCKER_EXECUTABLE", "docker")
    """Path to the docker/container executable."""
    run_args: list[str] = ["--rm"]
    """Additional arguments to pass to the docker/container executable.
    Default is ["--rm"], which removes the container after it exits.
    """
    container_timeout: str = "2h"
    """Max duration to keep container running. Uses the same format as the sleep command."""
    pull_timeout: int = 120
    """Timeout in seconds for pulling images."""
    interpreter: list[str] = ["bash", "-lc"]
    """Interpreter to use to execute commands. Default is ["bash", "-lc"].
    The actual command will be appended as argument to this. Override this to e.g., modify shell flags
    (e.g., to remove the `-l` flag to disable login shell) or to use python instead of bash to interpret commands.
    """


class DockerEnvironment:
    def __init__(
        self,
        *,
        config_class: type = DockerEnvironmentConfig,
        logger: logging.Logger | None = None,
        **kwargs,
    ):
        """This class executes bash commands in a Docker container using direct docker commands.
        See `DockerEnvironmentConfig` for keyword arguments.
        """
        self.logger = logger or logging.getLogger("minisweagent.environment")
        self.container_id: str | None = None
        self.config = config_class(**kwargs)
        self._start_container()

    def get_template_vars(self, **kwargs) -> dict[str, Any]:
        return recursive_merge(self.config.model_dump(), platform.uname()._asdict(), kwargs)

    def serialize(self) -> dict:
        return {
            "info": {
                "config": {
                    "environment": self.config.model_dump(mode="json"),
                    "environment_type": f"{self.__class__.__module__}.{self.__class__.__name__}",
                }
            }
        }

    def _start_container(self):
        """Start the Docker container and return the container ID."""
        container_name = f"minisweagent-{uuid.uuid4().hex[:8]}"
        cmd = [
            self.config.executable,
            "run",
            "-d",
            "--name",
            container_name,
            "-w",
            self.config.cwd,
            *self.config.run_args,
            self.config.image,
            "sleep",
            self.config.container_timeout,
        ]
        self.logger.debug(f"Starting container with command: {shlex.join(cmd)}")
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=self.config.pull_timeout,  # docker pull might take a while
            check=True,
        )
        self.logger.info(f"Started container {container_name} with ID {result.stdout.strip()}")
        self.container_id = result.stdout.strip()

    def execute(self, action: dict, cwd: str = "", *, timeout: int | None = None) -> dict[str, Any]:
        """Execute a command in the Docker container and return the result as a dict."""
        command = action.get("command", "")
        cwd = cwd or self.config.cwd
        assert self.container_id, "Container not started"

        cmd = [self.config.executable, "exec", "-w", cwd]
        for key in self.config.forward_env:
            if (value := os.getenv(key)) is not None:
                cmd.extend(["-e", f"{key}={value}"])
        for key, value in self.config.env.items():
            cmd.extend(["-e", f"{key}={value}"])
        cmd.extend([self.container_id, *self.config.interpreter, command])

        try:
            result = subprocess.run(
                cmd,
                text=True,
                timeout=timeout or self.config.timeout,
                encoding="utf-8",
                errors="replace",
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
            )
            output = {"output": result.stdout, "returncode": result.returncode, "exception_info": ""}
        except Exception as e:
            raw_output = getattr(e, "output", None)
            raw_output = (
                raw_output.decode("utf-8", errors="replace") if isinstance(raw_output, bytes) else (raw_output or "")
            )
            output = {
                "output": raw_output,
                "returncode": -1,
                "exception_info": f"An error occurred while executing the command: {e}",
                "extra": {"exception_type": type(e).__name__, "exception": str(e)},
            }
        output = self._check_finished(output, cwd)  # ENHANCED_V1: validate final patch submissions before accepting sentinel.
        return output

    def _check_finished(self, output: dict, cwd: str) -> dict:
        """Raises Submitted if the output indicates task completion, otherwise returns the output."""
        lines = output.get("output", "").lstrip().splitlines(keepends=True)
        if not (lines and lines[0].strip() == "COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT" and output["returncode"] == 0):
            return output

        submission = "".join(lines[1:])
        if _looks_like_unified_diff(submission):  # ENHANCED_V1: accept only plausible unified git diffs.
            raise Submitted(
                {
                    "role": "exit",
                    "content": submission,
                    "extra": {"exit_status": "Submitted", "submission": submission},
                }
            )

        workspace_diff = self._collect_workspace_diff(cwd)
        if _looks_like_unified_diff(workspace_diff):  # ENHANCED_V1: recover malformed final payloads from tracked edits.
            raise Submitted(
                {
                    "role": "exit",
                    "content": workspace_diff,
                    "extra": {
                        "exit_status": "Submitted",
                        "submission": workspace_diff,
                        "submission_source": "auto_git_diff",
                        "rejected_submission_reason": "sentinel payload was not a unified diff",
                    },
                }
            )

        return _reject_invalid_submission(
            output,
            "Final submission rejected: the text after COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT was not a unified git "
            "diff, and `git diff --no-color --` did not produce a valid patch. Create a non-empty patch.txt with "
            "`git diff -- path/to/source_file > patch.txt`, inspect that it contains `diff --git`, `--- a/`, and "
            "`+++ b/` headers, then resubmit with the exact required command.",
        )

    def _collect_workspace_diff(self, cwd: str) -> str:
        """Collect tracked workspace edits from the running container without recursive completion handling."""
        if not self.container_id:
            return ""
        cmd = [
            self.config.executable,
            "exec",
            "-w",
            cwd,
            self.container_id,
            *self.config.interpreter,
            "git diff --no-color --",
        ]
        try:
            result = subprocess.run(
                cmd,
                text=True,
                timeout=max(1, min(self.config.timeout, 10)),
                encoding="utf-8",
                errors="replace",
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
            )
            return result.stdout or ""
        except Exception:
            return ""

    def cleanup(self):
        """Stop and remove the Docker container."""
        if getattr(self, "container_id", None) is not None:  # if init fails early, container_id might not be set
            cmd = f"(timeout 60 {self.config.executable} stop {self.container_id} || {self.config.executable} rm -f {self.container_id}) >/dev/null 2>&1 &"
            subprocess.Popen(cmd, shell=True)

    def __del__(self):
        """Cleanup container when object is destroyed."""
        self.cleanup()


def _looks_like_unified_diff(text: str) -> bool:
    """Return True for a conservative unified git diff shape."""  # ENHANCED_V1: shared final-submission validator.
    stripped = (text or "").strip()
    return bool(stripped and "diff --git " in stripped and "--- a/" in stripped and "+++ b/" in stripped)


def _reject_invalid_submission(output: dict, reason: str) -> dict:
    """Convert an invalid completion attempt into a recoverable tool observation."""  # ENHANCED_V1: keep agent loop alive on malformed final output.
    rejected = dict(output)
    rejected["returncode"] = 1
    rejected["output"] = reason
    rejected.setdefault("exception_info", "")
    rejected["extra"] = dict(rejected.get("extra", {}))
    rejected["extra"]["submission_rejected"] = True
    return rejected
