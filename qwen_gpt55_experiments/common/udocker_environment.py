"""mini-swe-agent environment backed by udocker rather than a Docker daemon."""

from __future__ import annotations

import fcntl
import logging
import os
import platform
import subprocess
import uuid
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from minisweagent.exceptions import Submitted
from minisweagent.utils.serialize import recursive_merge

from .udocker_runtime import (
    container_scratch_volumes,
    cleanup_private_containers_dir,
    create_slot,
    make_private_containers_dir,
    udocker_host_environment,
)


class UdockerEnvironmentConfig(BaseModel):
    image: str
    repo: Path
    executable: Path
    cwd: str = "/testbed"
    env: dict[str, str] = Field(default_factory=dict)
    forward_env: list[str] = Field(default_factory=list)
    timeout: int = 60
    pull_timeout: int = 1800
    interpreter: list[str] = Field(default_factory=lambda: ["bash", "-c"])
    volumes: dict[str, str] = Field(default_factory=dict)
    container_name: str = ""
    keep_container: bool = False
    containers_dir: Path | None = None


class UdockerEnvironment:
    """Run each agent action in the same writable udocker container tree."""

    def __init__(self, *, logger: logging.Logger | None = None, **kwargs):
        self.logger = logger or logging.getLogger("minisweagent.environment.udocker")
        self.config = UdockerEnvironmentConfig(**kwargs)
        self.config.repo.mkdir(parents=True, exist_ok=True)
        self.container_name = self.config.container_name or f"miniswe-{uuid.uuid4().hex[:12]}"
        self.containers_dir = self.config.containers_dir or make_private_containers_dir(
            self.container_name
        )
        self.scratch_volumes = container_scratch_volumes(self.containers_dir)
        try:
            self._create_container()
        except Exception:
            cleanup_private_containers_dir(self.containers_dir)
            raise

    @property
    def _base_command(self) -> list[str]:
        return [
            str(self.config.executable),
            "--allow-root",
            f"--repo={self.config.repo}",
        ]

    def _run_host(
        self,
        args: list[str],
        *,
        timeout: int,
        input_text: str | None = None,
        check: bool = False,
    ) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [*self._base_command, *args],
            input=input_text,
            text=True,
            encoding="utf-8",
            errors="replace",
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=timeout,
            check=check,
            env=udocker_host_environment(self.containers_dir),
        )

    def _run_create(self) -> subprocess.CompletedProcess[str]:
        with create_slot():
            return self._run_host(
                ["create", f"--name={self.container_name}", self.config.image],
                timeout=self.config.pull_timeout,
            )

    def _create_container(self) -> None:
        created = self._run_create()
        if created.returncode == 0:
            return

        lock_path = self.config.repo / ".image_pull.lock"
        with lock_path.open("a+") as lock_file:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
            created = self._run_create()
            if created.returncode == 0:
                return

            self.logger.info("Pulling udocker image %s", self.config.image)
            pulled = self._run_host(
                ["pull", self.config.image],
                timeout=self.config.pull_timeout,
            )
            if pulled.returncode != 0:
                raise RuntimeError(f"Failed to pull {self.config.image}:\n{pulled.stdout}")

            created = self._run_create()
            if created.returncode != 0:
                raise RuntimeError(
                    f"Failed to create {self.container_name}:\n{created.stdout}"
                )

    def get_template_vars(self, **kwargs) -> dict[str, Any]:
        return recursive_merge(self.config.model_dump(), platform.uname()._asdict(), kwargs)

    def serialize(self) -> dict:
        config = self.config.model_dump(mode="json")
        config["container_name"] = self.container_name
        return {
            "info": {
                "config": {
                    "environment": config,
                    "environment_type": f"{self.__class__.__module__}.{self.__class__.__name__}",
                }
            }
        }

    def execute(
        self,
        action: dict,
        cwd: str = "",
        *,
        timeout: int | None = None,
        input_text: str | None = None,
    ) -> dict[str, Any]:
        command = action.get("command", "")
        workdir = cwd or self.config.cwd
        args = ["run", "--nobanner", f"--workdir={workdir}"]

        for key in self.config.forward_env:
            if (value := os.getenv(key)) is not None:
                args.append(f"--env={key}={value}")
        for key, value in self.config.env.items():
            args.append(f"--env={key}={value}")
        for host_path, container_path in self.config.volumes.items():
            args.append(f"--volume={host_path}:{container_path}")
        for host_path, container_path in self.scratch_volumes.items():
            args.append(f"--volume={host_path}:{container_path}")

        args.extend([self.container_name, *self.config.interpreter, command])
        try:
            result = self._run_host(
                args,
                timeout=timeout or self.config.timeout,
                input_text=input_text,
            )
            output = {
                "output": result.stdout,
                "returncode": result.returncode,
                "exception_info": "",
            }
        except Exception as exc:
            raw_output = getattr(exc, "output", None)
            if isinstance(raw_output, bytes):
                raw_output = raw_output.decode("utf-8", errors="replace")
            output = {
                "output": raw_output or "",
                "returncode": -1,
                "exception_info": f"An error occurred while executing the command: {exc}",
                "extra": {"exception_type": type(exc).__name__, "exception": str(exc)},
            }

        self._check_finished(output)
        return output

    def _check_finished(self, output: dict) -> None:
        lines = output.get("output", "").lstrip().splitlines(keepends=True)
        if lines and lines[0].strip() == "COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT" and output["returncode"] == 0:
            submission = "".join(lines[1:])
            raise Submitted(
                {
                    "role": "exit",
                    "content": submission,
                    "extra": {"exit_status": "Submitted", "submission": submission},
                }
            )

    def cleanup(self) -> None:
        if not getattr(self, "container_name", ""):
            return
        if self.config.keep_container:
            return
        try:
            removed = self._run_host(["rm", self.container_name], timeout=120)
            if removed.returncode != 0:
                self.logger.warning(
                    "Failed to remove udocker container %s: %s",
                    self.container_name,
                    removed.stdout[-500:],
                )
        except Exception:
            self.logger.warning("Failed to remove udocker container %s", self.container_name)
        finally:
            self.container_name = ""
            cleanup_private_containers_dir(self.containers_dir)

    def __enter__(self) -> "UdockerEnvironment":
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.cleanup()

    def __del__(self) -> None:
        try:
            self.cleanup()
        except Exception:
            pass
