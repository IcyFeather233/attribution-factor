"""Run the unmodified ReCreate evolution loop on the fixed failure set.

Only the unavailable Docker daemon boundary is replaced with udocker. ReCreate's
prompts, trajectory analysis, per-instance evolution, and batch synthesis are
called directly from the upstream repository.
"""

from __future__ import annotations

import argparse
import copy
import fcntl
import importlib
import json
import os
import random
import shutil
import subprocess
import sys
import uuid
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pyarrow.parquet as pq

from attribution_experiments.common.model_config import (
    OPTIMIZER_MODEL,
    OPTIMIZER_BASE_URL,
    TASK_MODEL,
    TASK_BASE_URL,
    TASK_STEP_LIMIT,
    optimizer_api_key,
    task_api_key,
    task_model_kwargs,
)
from attribution_experiments.common.request_gate import run_task_model_request
from attribution_experiments.common.result_validity import retryable_result, retryable_text, recreate_final_problems
from attribution_experiments.common.execution_environment import backend_name
from types import SimpleNamespace
from attribution_experiments.common.udocker_runtime import (
    container_scratch_volumes,
    cleanup_private_containers_dir,
    create_slot,
    make_private_containers_dir,
    udocker_host_environment,
)

from attribution_experiments.common.analysis_payloads import (
    DEFAULT_PAYLOAD_ROOT,
    GROUPS,
    load_payloads,
    snapshot_payloads,
)

ROOT = Path(__file__).resolve().parents[2]
RECREATE_ROOT = ROOT / "baselines" / "recreate"
sys.path.insert(0, str(RECREATE_ROOT / "scripts"))
sys.path.insert(0, str(RECREATE_ROOT / "src"))

from evolve_utils import (  # noqa: E402
    BatchResult,
    create_default_scaffold,
    run_batch_synthesis,
    run_recreate_initial_creation,
    run_single_evolution,
)
from minisweagent.run.utils.save import save_traj  # noqa: E402
from recreate_agent.adapters import UnifiedInstance, UnifiedResult  # noqa: E402
from recreate_agent.adapters.swe_adapter import SWEBenchAdapter  # noqa: E402
from swebench.harness.constants import KEY_INSTANCE_ID, KEY_PREDICTION  # noqa: E402
from swebench.harness.grading import get_eval_report  # noqa: E402
from swebench.harness.test_spec.test_spec import make_test_spec  # noqa: E402


RUN_ROOT = ROOT / "attribution_experiments" / "recreate"
UDOCKER_REPO = ROOT / "native_baseline_runs" / "udocker"
UDOCKER_EXECUTABLE = ROOT / "native_baseline_runs" / ".venv" / "bin" / "udocker"
DATASET = Path(os.environ.get("ATTRIBUTION_PARQUET", ROOT / "data/tasks.parquet"))
FAILED_IDS = ROOT / "attribution_experiments" / "task_ids.txt"
MATERIALIZE_TIMEOUT_SECONDS = 4 * 60 * 60
DEFAULT_ADAPTATION_TASKS = 32
DEFAULT_ADAPTATION_BATCH_SIZE = 8
DEFAULT_ADAPTATION_WORKERS = 4
DEFAULT_FINAL_WORKERS = 3
DEFAULT_ADAPTATION_SEED = 20260902
EVOLUTION_PROTOCOL = "bounded_evolution_v1"
NATIVE_UPSTREAM_FUNCTIONS = ["run_recreate_initial_creation", "run_single_evolution", "run_batch_synthesis"]


def install_aligned_task_budget():
    """Bound task rollouts only; preserve the native ReCreate optimizer runtime."""
    if os.environ.get('EXPERIMENT_ALIGNED_TASK_BUDGET') != '1':
        return
    from minisweagent.agents import default as native_agent
    from minisweagent.agents.default import TerminatingException
    base = native_agent.DefaultAgent
    if getattr(base, '_aligned_wall_time', False):
        return
    class TimeExceeded(TerminatingException):
        pass
    class AlignedTaskAgent(base):
        _aligned_wall_time = True
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self._task_started = time.monotonic()
        def query(self):
            if isinstance(self.env, V1UdockerEnvironment) and time.monotonic() - self._task_started >= 3600:
                raise TimeExceeded('Task wall time exceeded 3600 seconds')
            return super().query()
    native_agent.DefaultAgent = AlignedTaskAgent


class V1UdockerEnvironment:
    """ReCreate's legacy mini-swe-agent Environment protocol over udocker."""

    def __init__(
        self,
        *,
        image: str,
        cwd: str = "/testbed",
        timeout: int = 100,
        workspace_host_path: Path | None = None,
        container_name: str = "",
    ):
        self.image = image
        self.cwd = cwd
        self.timeout = timeout
        self.workspace_host_path = Path(workspace_host_path).resolve() if workspace_host_path else None
        self.container_name = container_name or f"recreate-{uuid.uuid4().hex[:12]}"
        self._remote = None
        if backend_name() == 'remote_docker':
            from attribution_experiments.common.remote_runtime import RemoteRuntime
            self._remote = RemoteRuntime(SimpleNamespace(
                image=image, repo=UDOCKER_REPO, cwd=cwd, timeout=timeout,
                container_name=self.container_name,
                volumes={str(self.workspace_host_path): '/workspace'} if self.workspace_host_path else {},
            ))
            self.container_name = self._remote.container_name
            self.containers_dir = self._remote.local_dir
            self.scratch_volumes = {}
            self.config = dict(image=image, cwd=cwd, timeout=timeout, container_name=self.container_name,
                               workspace_host_path=str(self.workspace_host_path or ''), environment_transport='remote_docker_v1')
            return
        self.containers_dir = make_private_containers_dir(self.container_name)
        self.scratch_volumes = container_scratch_volumes(self.containers_dir)
        self.config = {
            "image": image,
            "cwd": cwd,
            "timeout": timeout,
            "workspace_host_path": str(self.workspace_host_path or ""),
            "container_name": self.container_name,
        }
        try:
            self._create()
        except Exception:
            cleanup_private_containers_dir(self.containers_dir)
            raise

    @property
    def base(self) -> list[str]:
        return [
            str(UDOCKER_EXECUTABLE),
            "--allow-root",
            f"--repo={UDOCKER_REPO}",
        ]

    def host_run(
        self,
        args: list[str],
        *,
        timeout: int,
        input_text: str | None = None,
    ) -> subprocess.CompletedProcess[str]:
        if self._remote is not None:
            return self._remote.run(args, timeout=timeout, input_text=input_text)
        return subprocess.run(
            [*self.base, *args],
            input=input_text,
            text=True,
            encoding="utf-8",
            errors="replace",
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=timeout,
            env=udocker_host_environment(self.containers_dir),
        )

    def _run_create(self) -> subprocess.CompletedProcess[str]:
        with create_slot():
            return self.host_run(
                ["create", f"--name={self.container_name}", self.image],
                timeout=MATERIALIZE_TIMEOUT_SECONDS,
            )

    def _create(self) -> None:
        result = self._run_create()
        if result.returncode == 0:
            return

        lock_path = UDOCKER_REPO / ".image_pull.lock"
        with lock_path.open("a+") as lock_file:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
            result = self._run_create()
            if result.returncode == 0:
                return
            pulled = self.host_run(
                ["pull", self.image], timeout=MATERIALIZE_TIMEOUT_SECONDS
            )
            if pulled.returncode != 0:
                raise RuntimeError(f"Failed to pull {self.image}:\n{pulled.stdout}")
            result = self._run_create()
            if result.returncode != 0:
                raise RuntimeError(
                    f"Failed to create {self.container_name}:\n{result.stdout}"
                )

    def execute(
        self,
        command: str,
        cwd: str = "",
        *,
        timeout: int | None = None,
        input_text: str | None = None,
    ) -> dict:
        args = ["run", "--nobanner", f"--workdir={cwd or self.cwd}"]
        if self.workspace_host_path:
            args.append(f"--volume={self.workspace_host_path}:/workspace")
        for host_path, container_path in self.scratch_volumes.items():
            args.append(f"--volume={host_path}:{container_path}")
        args.extend(
            [
                "--env=PAGER=cat",
                "--env=MANPAGER=cat",
                "--env=PIP_PROGRESS_BAR=off",
                "--env=TQDM_DISABLE=1",
                self.container_name,
                "bash",
                "-c",
                command,
            ]
        )
        try:
            result = self.host_run(
                args,
                timeout=timeout or self.timeout,
                input_text=input_text,
            )
            return {"output": result.stdout, "returncode": result.returncode}
        except subprocess.TimeoutExpired as exc:
            # ReCreate vendors mini-swe-agent v1, whose agent timeout handler
            # decodes TimeoutExpired.output itself. Preserve that bytes
            # protocol even though host_run() normally operates in text mode.
            output = exc.output or b""
            if isinstance(output, str):
                output = output.encode("utf-8", errors="replace")
            raise subprocess.TimeoutExpired(exc.cmd, exc.timeout, output=output)

    def get_template_vars(self) -> dict:
        return {
            "image": self.image,
            "cwd": self.cwd,
            "timeout": self.timeout,
        }

    def cleanup(self) -> None:
        if self._remote is not None:
            self._remote.cleanup()
            self.container_name = ''
            return
        if not self.container_name:
            return
        container_name = self.container_name
        try:
            self.host_run(["rm", container_name], timeout=120)
        except (OSError, subprocess.SubprocessError) as exc:
            print(
                f"WARNING: best-effort udocker cleanup failed for {container_name}: "
                f"{type(exc).__name__}: {exc}",
                file=sys.stderr,
            )
        finally:
            cleanup_private_containers_dir(self.containers_dir)
            self.container_name = ""


def image_for(instance_id: str) -> str:
    docker_id = instance_id.replace("__", "_1776_")
    return f"docker.io/swebench/sweb.eval.x86_64.{docker_id}:latest".lower()


def load_instances() -> list[UnifiedInstance]:
    ids = [line.strip() for line in FAILED_IDS.read_text().splitlines() if line.strip()]
    rows = pq.read_table(DATASET).to_pylist()
    by_id = {row["instance_id"]: row for row in rows}
    return [
        UnifiedInstance(
            instance_id=task_id,
            problem_statement=by_id[task_id]["problem_statement"],
            category=by_id[task_id]["repo"],
            domain_data=by_id[task_id],
        )
        for task_id in ids
    ]


def select_adaptation_instances(
    instances: list[UnifiedInstance], *, limit: int, seed: int
) -> list[UnifiedInstance]:
    """Select a deterministic repository-balanced adaptation subset."""
    if limit >= len(instances):
        return list(instances)

    by_category: dict[str, list[UnifiedInstance]] = {}
    for instance in instances:
        by_category.setdefault(instance.category, []).append(instance)

    rng = random.Random(seed)
    categories = sorted(by_category)
    rng.shuffle(categories)
    for category in categories:
        rng.shuffle(by_category[category])

    selected: list[UnifiedInstance] = []
    offsets = {category: 0 for category in categories}
    while len(selected) < limit:
        made_progress = False
        for category in categories:
            offset = offsets[category]
            candidates = by_category[category]
            if offset >= len(candidates):
                continue
            selected.append(candidates[offset])
            offsets[category] = offset + 1
            made_progress = True
            if len(selected) == limit:
                break
        if not made_progress:
            break

    if len(selected) != limit:
        raise RuntimeError(
            f"Could only select {len(selected)} of {limit} adaptation tasks"
        )
    return selected


def count_tests(report: dict, instance_id: str) -> tuple[int, int]:
    statuses = report.get(instance_id, {}).get("tests_status", {})
    passed = 0
    failed = 0
    for group in statuses.values():
        if not isinstance(group, dict):
            continue
        passed += len(group.get("success", []))
        failed += len(group.get("failure", []))
    return passed, failed


def evaluate_patch(
    *,
    instance: UnifiedInstance,
    patch: str,
    output_dir: Path,
) -> UnifiedResult:
    output_dir.mkdir(parents=True, exist_ok=True)
    task_id = instance.instance_id
    if not patch.strip():
        result = {
            "instance_id": task_id,
            "resolved": False,
            "error": "empty_patch",
        }
        (output_dir / "eval_result.json").write_text(json.dumps(result, indent=2))
        return UnifiedResult(
            instance_id=task_id,
            success=False,
            error="empty_patch",
            formatted_output=json.dumps(result, indent=2),
        )

    env = V1UdockerEnvironment(
        image=image_for(task_id),
        container_name=f"recreate-eval-{uuid.uuid4().hex[:12]}",
    )
    result: dict = {
        'execution_backend': backend_name(),
        "instance_id": task_id,
        "resolved": False,
        "patch_applied": False,
    }
    try:
        applied = env.execute("git apply --verbose -", input_text=patch, timeout=180)
        (output_dir / "patch_apply.log").write_text(applied["output"])
        if applied["returncode"] != 0:
            result["error"] = "patch_apply_failed"
        else:
            result["patch_applied"] = True
            test_spec = make_test_spec(instance.domain_data)
            tested = env.execute(
                "/bin/bash -s",
                input_text=test_spec.eval_script,
                timeout=1800,
            )
            test_output = tested["output"]
            test_log = output_dir / "test_output.txt"
            test_log.write_text(test_output)
            prediction = {
                KEY_INSTANCE_ID: task_id,
                "model_name_or_path": "recreate-native",
                KEY_PREDICTION: patch,
            }
            report = get_eval_report(
                test_spec=test_spec,
                prediction=prediction,
                test_log_path=str(test_log),
                include_tests_status=True,
            )
            result["official_report"] = report
            result["resolved"] = bool(report.get(task_id, {}).get("resolved", False))
            result["eval_returncode"] = tested["returncode"]
        passed, failed = count_tests(result.get("official_report", {}), task_id)
        result["tests_passed"] = passed
        result["tests_failed"] = failed
        (output_dir / "eval_result.json").write_text(json.dumps(result, indent=2))
        test_path = output_dir / "test_output.txt"
        formatted = json.dumps(
            {
                "resolved": result["resolved"],
                "tests_passed": passed,
                "tests_failed": failed,
                "error": result.get("error", ""),
                "test_output_tail": test_path.read_text()[-10000:] if test_path.exists() else "",
            },
            indent=2,
        )
        return UnifiedResult(
            instance_id=task_id,
            success=result["resolved"],
            score=1.0 if result["resolved"] else 0.0,
            error=result.get("error", ""),
            details={"tests_passed": passed, "tests_failed": failed},
            eval_result=result,
            formatted_output=formatted,
        )
    finally:
        env.cleanup()


class UdockerSWEBenchAdapter(SWEBenchAdapter):
    def __init__(self, *, step_limit: int, analysis_payloads: dict[str, dict]):
        super().__init__()
        self.step_limit = step_limit
        self.analysis_payloads = analysis_payloads
        self.inject_analysis_payloads = any(
            payload.get("provided_information") for payload in analysis_payloads.values()
        )
        self.environments: dict[str, V1UdockerEnvironment] = {}

    def load_dataset(self, **kwargs) -> list[UnifiedInstance]:
        return load_instances()

    def create_environment(
        self,
        instance: UnifiedInstance,
        output_dir: Path,
        tools_dir: Path | None = None,
    ):
        env = V1UdockerEnvironment(
            image=image_for(instance.instance_id),
            cwd="/testbed",
            timeout=100,
            workspace_host_path=tools_dir,
        )
        self.environments[env.container_name] = env
        return env, env.container_name

    def run_agent(
        self,
        instance: UnifiedInstance,
        scaffold: dict,
        model_name: str,
        output_dir: Path,
        tools_dir: Path | None = None,
        temperature: float = 1.0,
    ):
        bounded = dict(scaffold)
        bounded["step_limit"] = self.step_limit
        bounded["cost_limit"] = 0
        return super().run_agent(
            instance,
            bounded,
            model_name,
            output_dir,
            tools_dir,
            temperature,
        )

    def evaluate(
        self,
        container_id: str,
        instance: UnifiedInstance,
        output_dir: Path,
    ) -> UnifiedResult:
        output_dir.mkdir(parents=True, exist_ok=True)
        env = self.environments[container_id]
        diff = env.execute("git -c core.fileMode=false diff HEAD", timeout=120)
        patch = diff["output"] if diff["returncode"] == 0 else ""
        (output_dir / "patch.diff").write_text(patch)
        result = evaluate_patch(
            instance=instance,
            patch=patch,
            output_dir=output_dir,
        )
        if self.inject_analysis_payloads:
            payload = self.analysis_payloads[instance.instance_id]
            result.details = dict(result.details or {})
            result.details["provided_failure_analysis"] = payload["provided_information"]
            from attribution_experiments.common.flow_evidence import render as render_flow
            historical = render_flow(payload, instance.instance_id,
                                     current_context=f'Current candidate success={result.success}; the graph describes the original failed trace.')
            if historical:
                result.details['historical_flow_evidence'] = historical
        return result


def install_local_environment_wrapper(run_dir: Path) -> None:
    """Keep ReCreate's environment-inspection feature working with udocker."""
    shim_dir = run_dir / "runtime_bin"
    shim_dir.mkdir(parents=True, exist_ok=True)
    docker_shim = shim_dir / "docker"
    docker_shim.write_text(
        "#!/bin/bash\n"
        "if [[ \"$1\" != \"exec\" ]]; then\n"
        "  echo 'Only docker exec is supported by this experiment shim' >&2\n"
        "  exit 2\n"
        "fi\n"
        "shift\n"
        "workdir=/testbed\n"
        "if [[ \"$1\" == \"-w\" ]]; then workdir=\"$2\"; shift 2; fi\n"
        "container=\"$1\"; shift\n"
        "exec \"$RECREATE_UDOCKER\" --allow-root --repo=\"$RECREATE_UDOCKER_REPO\" "
        "run --nobanner --workdir=\"$workdir\" \"$container\" \"$@\"\n"
    )
    docker_shim.chmod(0o755)
    if backend_name() == 'remote_docker':
        docker_shim.write_text(
            '#!' + sys.executable + '\n'
            'import sys\n'
            'sys.path.insert(0, ' + repr(str(ROOT)) + ')\n'
            'from attribution_experiments.common.remote_exec import main\n'
            'main()\n'
        )

    from minisweagent.environments import local as local_module
    from recreate_agent import recreate_agent as recreate_module

    original = local_module.LocalEnvironment

    class SidecarLocalEnvironment(original):
        def __init__(self, *args, **kwargs):
            env = dict(kwargs.get("env") or {})
            env["PATH"] = f"{shim_dir}:{UDOCKER_EXECUTABLE.parent}:/usr/local/bin:/usr/bin:/bin"
            env["RECREATE_UDOCKER"] = str(UDOCKER_EXECUTABLE)
            env["RECREATE_UDOCKER_REPO"] = str(UDOCKER_REPO)
            kwargs["env"] = env
            super().__init__(*args, **kwargs)

    local_module.LocalEnvironment = SidecarLocalEnvironment
    recreate_module.LocalEnvironment = SidecarLocalEnvironment


def prepare_final_task_workspace(scaffold_dir: Path, task_dir: Path) -> None:
    """Copy evolved tools and memory into an isolated per-task mount."""
    for name in ("agent_tools", "agent_memory"):
        source = scaffold_dir / name
        if not source.is_dir():
            raise RuntimeError(f"Final ReCreate artifact is missing required {name}: {source}")
        target = task_dir / name
        if target.exists():
            shutil.rmtree(target)
        shutil.copytree(source, target)



def initialize_workspace(
    *,
    workspace: Path,
    runs_recreate_dir: Path,
    adapter: UdockerSWEBenchAdapter,
    smoke: bool,
) -> dict:
    domain_config = adapter.get_recreate_agent_config()
    ablations = {
        "ablation_trajectory": True,
        "ablation_environment": True,
        "ablation_eval_results": True,
        "ablation_modification_guidance": True,
    }
    if workspace.exists():
        shutil.rmtree(workspace)
    workspace.mkdir(parents=True)
    if smoke:
        create_default_scaffold(workspace, "swe", domain_config)
    else:
        ok = run_recreate_initial_creation(
            workspace,
            runs_recreate_dir,
            OPTIMIZER_MODEL,
            1.0,
            "swe",
            adapter.domain_description,
            adapter.get_initial_prompt_template(),
            domain_config,
            ablations,
        )
        if not ok:
            raise RuntimeError("ReCreate initial scaffold creation failed")
    return {"domain_config": domain_config, "ablations": ablations}


def redact_path(path: Path, api_key: str) -> None:
    if not api_key or api_key == "EMPTY" or not path.is_file():
        return
    try:
        text = path.read_text()
    except (OSError, UnicodeDecodeError):
        return
    if api_key in text:
        path.write_text(text.replace(api_key, "<REDACTED>"))


def redact_api_key(root: Path, api_key: str) -> None:
    if not api_key or api_key == "EMPTY" or not root.exists():
        return
    for path in root.rglob("*"):
        if path.suffix in {".json", ".jsonl", ".txt", ".log", ".yaml"}:
            redact_path(path, api_key)


def install_dual_model_router(*, optimizer_key: str, task_key: str) -> None:
    models = importlib.import_module("minisweagent.models")
    original = models.get_model
    if getattr(original, "_task_model_dual_endpoint", False):
        return

    def routed_get_model(input_model_name=None, config=None):
        routed = copy.deepcopy(config or {})
        kwargs = routed.setdefault("model_kwargs", {})
        if input_model_name == TASK_MODEL:
            kwargs.update(task_model_kwargs(api_key=task_key))
        elif input_model_name == OPTIMIZER_MODEL:
            kwargs.update({
                "api_key": optimizer_key,
                "api_base": OPTIMIZER_BASE_URL,
                "custom_llm_provider": "openai",
                "drop_params": True,
            })
        else:
            raise ValueError(f"Unconfigured ReCreate model: {input_model_name}")
        return original(input_model_name, routed)

    routed_get_model._task_model_dual_endpoint = True
    models.get_model = routed_get_model

    litellm_module = importlib.import_module("minisweagent.models.litellm_model")
    original_query = litellm_module.LitellmModel._query
    if not getattr(original_query, "_request_gated", False):
        def gated_query(model, messages, **kwargs):
            if model.config.model_name == TASK_MODEL:
                return run_task_model_request(
                    lambda: original_query(model, messages, **kwargs)
                )
            return original_query(model, messages, **kwargs)

        gated_query._request_gated = True
        litellm_module.LitellmModel._query = gated_query
    for module_name in (
        "recreate_agent.recreate_agent",
        "recreate_agent.agent_runner",
        "recreate_agent.adapters.swe_adapter",
    ):
        module = sys.modules.get(module_name)
        if module is not None and hasattr(module, "get_model"):
            module.get_model = routed_get_model


def install_redacted_traj_saver(api_keys: tuple[str, ...]) -> None:
    save_module = importlib.import_module("minisweagent.run.utils.save")
    original = save_module.save_traj
    if getattr(original, "_task_model_redacted", False):
        return

    def safe_save_traj(*args, **kwargs):
        result = original(*args, **kwargs)
        path_value = args[1] if len(args) > 1 else kwargs.get("path")
        if path_value is not None:
            
            for api_key in api_keys:
                redact_path(Path(path_value), api_key)
        return result

    safe_save_traj._task_model_redacted = True
    save_module.save_traj = safe_save_traj
    globals()["save_traj"] = safe_save_traj
    for module_name in (
        "recreate_agent.recreate_agent",
        "recreate_agent.agent_runner",
    ):
        module = sys.modules.get(module_name)
        if module is not None and hasattr(module, "save_traj"):
            module.save_traj = safe_save_traj


def run_final(
    *,
    instances: list[UnifiedInstance],
    scaffold_dir: Path,
    adapter: UdockerSWEBenchAdapter,
    output_dir: Path,
    workers: int,
) -> dict:
    from evolve_utils.scaffold_ops import load_scaffold

    scaffold = load_scaffold(scaffold_dir)

    def cached_result(instance: UnifiedInstance) -> dict | None:
        task_dir = output_dir / instance.instance_id
        evaluation_path = task_dir / "evaluation" / "eval_result.json"
        if not evaluation_path.is_file():
            return None
        try:
            evaluation = json.loads(evaluation_path.read_text())
        except (OSError, json.JSONDecodeError):
            return None
        if (
            evaluation.get("instance_id") != instance.instance_id
            or not isinstance(evaluation.get("resolved"), bool)
            or retryable_result({"evaluation": evaluation})
        ):
            return None
        exit_status = "CachedEvaluation"
        trajectory_path = task_dir / "trajectory.json"
        if trajectory_path.is_file():
            try:
                trajectory = json.loads(trajectory_path.read_text())
                exit_status = trajectory.get("info", {}).get("exit_status") or exit_status
            except (OSError, json.JSONDecodeError):
                pass
        return {
            "instance_id": instance.instance_id,
            "exit_status": exit_status,
            "resolved": bool(evaluation["resolved"]),
            "status": evaluation.get("error") or ("resolved" if evaluation["resolved"] else "unresolved"),
            "resumed_from_completed_evaluation": True,
        }

    def one(instance: UnifiedInstance) -> dict:
        cached = cached_result(instance)
        if cached is not None:
            return cached
        task_dir = output_dir / instance.instance_id
        task_dir.mkdir(parents=True, exist_ok=True)
        prepare_final_task_workspace(scaffold_dir, task_dir)
        exit_status = "Error"
        result = ""
        wrapper = None
        try:
            exit_status, result, _, wrapper = adapter.run_agent(
                instance,
                scaffold,
                TASK_MODEL,
                task_dir,
                task_dir,
                1.0,
            )
            evaluation = adapter.evaluate(
                wrapper.env.container_name,
                instance,
                task_dir / "evaluation",
            )
            if wrapper.agent:
                save_traj(
                    wrapper.agent,
                    task_dir / "trajectory.json",
                    exit_status=exit_status,
                    result=result,
                    instance_id=instance.instance_id,
                )
            return {
                "instance_id": instance.instance_id,
                "exit_status": exit_status,
                "resolved": evaluation.success,
                "status": evaluation.error or ("resolved" if evaluation.success else "unresolved"),
            }
        except Exception as exc:
            return {
                "instance_id": instance.instance_id,
                "exit_status": type(exc).__name__,
                "resolved": False,
                "status": str(exc),
            }
        finally:
            if wrapper:
                try:
                    wrapper.cleanup()
                except Exception as exc:
                    (task_dir / "cleanup_warning.log").write_text(
                        f"{type(exc).__name__}: {exc}\n"
                    )

    by_id = {}
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(one, item): item.instance_id for item in instances}
        for future in as_completed(futures):
            by_id[futures[future]] = future.result()
    results = [by_id[item.instance_id] for item in instances]
    retryable = [item["instance_id"] for item in results if retryable_text(item.get("status"))]
    problems = recreate_final_problems(output_dir, results)
    if problems:
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / 'incomplete_results.json').write_text(json.dumps(
            {'state': 'incomplete', 'problems': problems, 'results': results}, indent=2) + '\n')
    if retryable:
        raise RuntimeError(
            f"{len(retryable)} ReCreate final tasks have retryable infrastructure results: {retryable}"
        )
    if problems:
        raise RuntimeError(f'ReCreate final evaluation incomplete; missing or invalid reports: {problems}')
    summary = {
        "resolved": sum(item["resolved"] for item in results),
        "total": len(results),
        "results": results,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-name", default="native_round1")
    parser.add_argument("--ablation-group", choices=GROUPS, required=True)
    parser.add_argument("--payload-root", type=Path, default=DEFAULT_PAYLOAD_ROOT)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--resume-final", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--adaptation-tasks", type=int, default=DEFAULT_ADAPTATION_TASKS)
    parser.add_argument("--adaptation-repeats", type=int, default=1)
    parser.add_argument(
        "--adaptation-batch-size", type=int, default=DEFAULT_ADAPTATION_BATCH_SIZE
    )
    parser.add_argument(
        "--adaptation-workers", type=int, default=DEFAULT_ADAPTATION_WORKERS
    )
    parser.add_argument("--final-workers", type=int, default=DEFAULT_FINAL_WORKERS)
    parser.add_argument("--adaptation-seed", type=int, default=DEFAULT_ADAPTATION_SEED)
    args = parser.parse_args()

    for name in (
        "adaptation_tasks",
        "adaptation_repeats",
        "adaptation_batch_size",
        "adaptation_workers",
        "final_workers",
    ):
        if getattr(args, name) < 1:
            raise ValueError(f"--{name.replace('_', '-')} must be positive")

    optimizer_key = optimizer_api_key()
    task_key = task_api_key()
    os.environ.pop("MSWEA_MODEL_API_KEY", None)
    os.environ.pop("LLM_API_BASE", None)
    install_dual_model_router(optimizer_key=optimizer_key, task_key=task_key)
    install_aligned_task_budget()
    install_redacted_traj_saver((optimizer_key, task_key))
    os.environ["OPENAI_API_KEY"] = optimizer_key
    os.environ["LITELLM_LOG"] = "ERROR"

    run_dir = RUN_ROOT / args.ablation_group / args.run_name
    from attribution_experiments.common.optimization_budget import pin_budget, adaptation_batches
    pin_budget(run_dir, {
        'protocol': 'recreate_repeated_adaptation_v2', 'evolution_protocol': EVOLUTION_PROTOCOL,
        'group': args.ablation_group, 'adaptation_tasks': args.adaptation_tasks,
        'batch_size': args.adaptation_batch_size, 'repeats': args.adaptation_repeats,
        'seed': args.adaptation_seed, 'smoke': args.smoke,
    }, has_progress=any((run_dir / 'workspace').glob('batch_*')) or (run_dir / 'summary.json').exists())
    from attribution_experiments.common.execution_protocol import record_execution_backend
    record_execution_backend(run_dir)
    analysis_payloads = load_payloads(args.ablation_group, args.payload_root.resolve())
    snapshot_payloads(run_dir, args.ablation_group, analysis_payloads)
    from attribution_experiments.common.flow_evidence import write_receipt
    write_receipt(run_dir, analysis_payloads, 'native per-case analysis and batch synthesis evidence handoff')
    install_local_environment_wrapper(run_dir)

    step_limit = 3 if args.smoke else TASK_STEP_LIMIT
    adapter = UdockerSWEBenchAdapter(
        step_limit=step_limit,
        analysis_payloads=analysis_payloads,
    )
    instances = load_instances()
    if args.adaptation_tasks > len(instances):
        raise ValueError(
            f"--adaptation-tasks={args.adaptation_tasks} exceeds {len(instances)} tasks"
        )
    adaptation_instances = (
        instances[:1]
        if args.smoke
        else select_adaptation_instances(
            instances, limit=args.adaptation_tasks, seed=args.adaptation_seed
        )
    )
    final_instances = instances[:1] if args.smoke else instances
    batch_size = 1 if args.smoke else args.adaptation_batch_size
    adaptation_workers = 1 if args.smoke else args.adaptation_workers
    final_workers = 1 if args.smoke else args.final_workers
    if args.resume_final:
        current = run_dir / "workspace" / "current"
        if not current.exists():
            raise FileNotFoundError(f"Missing evolved ReCreate scaffold: {current}")
        final_scaffold = current.resolve()
        if not (final_scaffold / "scaffold.yaml").is_file():
            raise FileNotFoundError(f"Missing final scaffold.yaml: {final_scaffold}")
        adapter.inject_analysis_payloads = False
        summary = run_final(
            instances=instances,
            scaffold_dir=final_scaffold,
            adapter=adapter,
            output_dir=run_dir / "final",
            workers=final_workers,
        )
        global_versions = sorted((run_dir / "workspace").glob("global_v*"))
        output = {
            "baseline": "ReCreate",
            "task_model": TASK_MODEL,
            "task_step_limit": TASK_STEP_LIMIT,
            "optimizer_model": OPTIMIZER_MODEL,
            "ablation_group": args.ablation_group,
            "analysis_payloads_injected_at": (
                "none" if args.ablation_group == "baseline"
                else "per-case ReCreate-Agent analysis evidence (dedicated result.json field)"
            ),
            "native_upstream_functions": NATIVE_UPSTREAM_FUNCTIONS,
            "analysis_json_injected": args.ablation_group != "baseline",
            "evolution_protocol": EVOLUTION_PROTOCOL,
            "adaptation_tasks": len(adaptation_instances),
            "adaptation_task_ids": [item.instance_id for item in adaptation_instances],
            "adaptation_pool_tasks": len(instances),
            "adaptation_batch_size": batch_size,
            "adaptation_repeats": args.adaptation_repeats,
            "adaptation_workers": adaptation_workers,
            "adaptation_seed": args.adaptation_seed,
            "final_workers": final_workers,
            "global_versions": len(global_versions),
            "resume": {
                "stage": "final_benchmark_only",
                "final_scaffold": str(final_scaffold),
                "completed_evaluations_reused": sum(
                    bool(item.get("resumed_from_completed_evaluation"))
                    for item in summary["results"]
                ),
            },
            "final": summary,
        }
        (run_dir / "summary.json").write_text(json.dumps(output, indent=2) + "\n")
        redact_api_key(run_dir, optimizer_key)
        redact_api_key(run_dir, task_key)
        print(json.dumps(output, indent=2))
        return
    if args.resume:
        if not (run_dir / 'resume32_drained.json').is_file():
            raise RuntimeError('ReCreate adaptation resume requires an orderly-drain checkpoint')
        setup = {'domain_config': adapter.get_recreate_agent_config(), 'ablations': {
            'ablation_trajectory': True, 'ablation_environment': True,
            'ablation_eval_results': True, 'ablation_modification_guidance': True}}
    else:
        setup = initialize_workspace(
            workspace=run_dir / "workspace",
            runs_recreate_dir=run_dir / "runs_recreate",
            adapter=adapter,
            smoke=args.smoke,
        )
    workspace = run_dir / "workspace"
    global_version = 0
    evolution_results = []

    repeats = 1 if args.smoke else args.adaptation_repeats
    for batch_idx, repeat_idx, batch in adaptation_batches(adaptation_instances, batch_size, repeats):
        current = workspace / f"global_v{global_version:03d}"
        batch_dir = workspace / f"batch_{batch_idx:03d}"
        batch_dir.mkdir(parents=True, exist_ok=True)
        results: dict[str, BatchResult] = {}
        if args.resume:
            from .resume_state import recovered_batch_result
            for instance in batch:
                task_dir = batch_dir / instance.instance_id
                saved = recovered_batch_result(task_dir, current, instance.instance_id)
                if saved is not None:
                    results[instance.instance_id] = BatchResult(**saved)
                    print(f'Reused completed ReCreate evolution: {instance.instance_id}', flush=True)
                elif task_dir.exists():
                    # A task waiting for a slot may already have copied its scaffold.
                    task_dir.rename(task_dir.with_name(task_dir.name + '.before_resume32_' + uuid.uuid4().hex[:10]))
        with ThreadPoolExecutor(
            max_workers=min(adaptation_workers, len(batch))
        ) as executor:
            futures = {
                executor.submit(
                    run_single_evolution,
                    instance=instance,
                    global_scaffold_dir=current,
                    output_dir=batch_dir / instance.instance_id,
                    adapter=adapter,
                    agent_model=TASK_MODEL,
                    recreate_model=OPTIMIZER_MODEL,
                    agent_temp=1.0,
                    recreate_temp=1.0,
                    domain="swe",
                    domain_config=setup["domain_config"],
                    runs_recreate_dir=run_dir / "runs_recreate",
                    ablation_settings=setup["ablations"],
                ): instance.instance_id
                for instance in batch if instance.instance_id not in results
            }
            for future in as_completed(futures):
                task_id = futures[future]
                results[task_id] = future.result()
        from dataclasses import asdict
        for task_id, result in results.items():
            (batch_dir / task_id / 'batch_result.json').write_text(json.dumps(asdict(result), indent=2))
            evolution_results.append({'instance_id': task_id, 'success': result.success,
                                      'repeat': repeat_idx + 1, 'batch_index': batch_idx,
                                      'scaffold_changed': result.scaffold_changed, 'error': result.error})

        failures = {task_id: result.error for task_id, result in results.items() if result.error}
        if failures:
            (batch_dir / 'incomplete.json').write_text(json.dumps(failures, indent=2))
            raise RuntimeError(f'ReCreate adaptation batch {batch_idx} incomplete; preserved valid peers: {failures}')

        new_global = workspace / f"global_v{global_version + 1:03d}"
        if args.resume and new_global.is_dir():
            if not (new_global / 'scaffold.yaml').is_file():
                raise RuntimeError('Incomplete ReCreate batch synthesis')
            current_link = workspace / 'current'
            if current_link.exists() or current_link.is_symlink():
                current_link.unlink()
            current_link.symlink_to(new_global.name)
            global_version += 1
            continue
        from attribution_experiments.common.flow_evidence import render as render_flow, handoff as flow_handoff
        flow_inputs = {}
        for task_id in results:
            historical = render_flow(analysis_payloads[task_id], task_id)
            if historical:
                evidence_path = batch_dir / task_id / 'evolution_summary.md'
                reasoning = evidence_path.read_text() if evidence_path.is_file() else ''
                flow_inputs[task_id] = {'text': historical, 'handoff': flow_handoff(analysis_payloads[task_id], task_id, reasoning)}
        run_batch_synthesis(
            workspace=workspace,
            batch_dir=batch_dir,
            batch_results=results,
            provided_flow_evidence=flow_inputs,
            current_global_dir=current,
            new_global_dir=new_global,
            batch_idx=batch_idx,
            recreate_model=OPTIMIZER_MODEL,
            recreate_temp=1.0,
            domain="swe",
            domain_config=setup["domain_config"],
            runs_recreate_dir=run_dir / "runs_recreate",
            ablation_settings=setup["ablations"],
        )
        current_link = workspace / "current"
        if current_link.exists() or current_link.is_symlink():
            current_link.unlink()
        current_link.symlink_to(new_global.name)
        global_version += 1

    final_scaffold = workspace / f"global_v{global_version:03d}"
    shutil.copy(final_scaffold / "scaffold.yaml", run_dir / "final_scaffold.yaml")
    adapter.inject_analysis_payloads = False
    summary = run_final(
        instances=final_instances,
        scaffold_dir=final_scaffold,
        adapter=adapter,
        output_dir=run_dir / "final",
        workers=final_workers,
    )
    output = {
        "baseline": "ReCreate",
        "task_model": TASK_MODEL,
        "task_step_limit": TASK_STEP_LIMIT,
        "optimizer_model": OPTIMIZER_MODEL,
        "ablation_group": args.ablation_group,
        "analysis_payloads_injected_at": (
            "none" if args.ablation_group == "baseline"
            else "per-case ReCreate-Agent analysis evidence (dedicated result.json field)"
        ),
        "native_upstream_functions": NATIVE_UPSTREAM_FUNCTIONS,
        "analysis_json_injected": args.ablation_group != "baseline",
        "evolution_protocol": EVOLUTION_PROTOCOL,
        "adaptation_tasks": len(adaptation_instances),
        "adaptation_task_ids": [item.instance_id for item in adaptation_instances],
        "adaptation_pool_tasks": len(instances),
        "adaptation_batch_size": batch_size,
        "adaptation_repeats": repeats,
        "adaptation_task_visits": len(adaptation_instances) * repeats,
        "adaptation_workers": adaptation_workers,
        "adaptation_seed": args.adaptation_seed,
        "final_workers": final_workers,
        "scaffold_changes": sum(item["scaffold_changed"] for item in evolution_results),
        "global_versions": global_version + 1,
        "evolution_results": evolution_results,
        "final": summary,
    }
    (run_dir / "summary.json").write_text(json.dumps(output, indent=2))
    redact_api_key(run_dir, optimizer_key)
    redact_api_key(run_dir, task_key)
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
