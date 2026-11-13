"""Run the original mini-swe-agent harness with a supplied native artifact."""

from __future__ import annotations

import json
import os
import shutil
import traceback
import uuid
from copy import deepcopy
from pathlib import Path
from typing import Type

from minisweagent.agents.default import DefaultAgent
from minisweagent.models.litellm_model import LitellmModel

from .dataset import WORKSPACE, image_for_instance
from .evaluator import evaluate_patch, normalize_patch_transport
from .model_config import (
    OPTIMIZER_BASE_URL,
    OPTIMIZER_MODEL,
    TASK_BASE_URL,
    TASK_MODEL,
    TASK_STEP_LIMIT,
    task_api_key,
    task_model_kwargs,
)
from .request_gate import run_task_model_request
from .execution_environment import ExecutionEnvironment as UdockerEnvironment
from .execution_environment import backend_name


REFERENCE_TRACE = Path(os.environ.get("ATTRIBUTION_REFERENCE_TRACE", WORKSPACE / "data" / "reference.traj.json"))


def bounded_task_wall_time(agent_config: dict) -> float:
    """Cap a whole task without lengthening a stricter scaffold limit."""
    maximum = float(os.environ.get("TASK_MODEL_WALL_TIME_SECONDS", "3600"))
    if maximum <= 0:
        raise ValueError("TASK_MODEL_WALL_TIME_SECONDS must be positive")
    configured = float(agent_config.get("wall_time_limit_seconds", 0) or 0)
    return maximum if configured <= 0 else min(configured, maximum)


class RedactedLitellmModel(LitellmModel):
    def _query(self, messages: list[dict[str, str]], **kwargs):
        parent_query = super()._query
        if self.config.model_name == TASK_MODEL:
            return run_task_model_request(lambda: parent_query(messages, **kwargs))
        return parent_query(messages, **kwargs)

    def serialize(self) -> dict:
        data = super().serialize()
        model_config = data.get("info", {}).get("config", {}).get("model", {})
        model_kwargs = model_config.get("model_kwargs", {})
        if "api_key" in model_kwargs:
            model_kwargs["api_key"] = "<REDACTED>"
        return data


def redact_secret(path: Path, secret: str | None) -> None:
    if not secret or secret == "EMPTY" or not path.is_file():
        return
    text = path.read_text()
    if secret in text:
        path.write_text(text.replace(secret, "<REDACTED>"))


def load_original_config(trace_path: Path = REFERENCE_TRACE) -> dict:
    """Recover the exact mini-swe templates and limits used by the failed run."""
    trace = json.loads(trace_path.read_text())
    config = trace["info"]["config"]
    return {
        "agent": deepcopy(config["agent"]),
        "model": deepcopy(config["model"]),
        "environment": deepcopy(config["environment"]),
    }


def build_model(
    *,
    api_key: str,
    temperature: float = 1.0,
    model_overrides: dict | None = None,
) -> LitellmModel:
    base = load_original_config()["model"]
    mutable_model_fields = {
        "format_error_template",
        "observation_template",
        "multimodal_regex",
    }
    if model_overrides:
        base.update(
            {
                key: value
                for key, value in model_overrides.items()
                if key in mutable_model_fields
            }
        )
    return RedactedLitellmModel(
        model_name=TASK_MODEL,
        model_kwargs=task_model_kwargs(api_key=api_key, temperature=temperature),
        cost_tracking="ignore_errors",
        format_error_template=base["format_error_template"],
        observation_template=base["observation_template"],
        multimodal_regex=base.get("multimodal_regex", ""),
    )


def build_optimizer_model(
    *,
    api_key: str,
    temperature: float = 1.0,
) -> LitellmModel:
    base = load_original_config()["model"]
    return RedactedLitellmModel(
        model_name=OPTIMIZER_MODEL,
        model_kwargs={
            "api_key": api_key,
            "api_base": OPTIMIZER_BASE_URL,
            "custom_llm_provider": "openai",
            "drop_params": True,
            "parallel_tool_calls": True,
            "temperature": temperature,
        },
        cost_tracking="ignore_errors",
        format_error_template=base["format_error_template"],
        observation_template=base["observation_template"],
        multimodal_regex=base.get("multimodal_regex", ""),
    )


def run_task(
    *,
    instance: dict,
    output_dir: Path,
    udocker_repo: Path,
    udocker_executable: Path,
    api_key: str | None = None,
    system_template: str | None = None,
    instance_template: str | None = None,
    agent_overrides: dict | None = None,
    model_overrides: dict | None = None,
    template_vars: dict | None = None,
    agent_class: Type[DefaultAgent] = DefaultAgent,
    harness_assets_dir: Path | None = None,
    step_limit: int = TASK_STEP_LIMIT,
    temperature: float = 1.0,
    evaluate: bool = True,
) -> dict:
    """Run one task and optionally grade the resulting patch in a fresh image."""
    output_dir.mkdir(parents=True, exist_ok=True)
    api_key = api_key or task_api_key()

    original = load_original_config()
    agent_config = original["agent"]
    if agent_overrides:
        allowed = {
            "system_template",
            "instance_template",
            "step_limit",
            "cost_limit",
            "max_consecutive_format_errors",
            "wall_time_limit_seconds",
        }
        agent_config.update(
            {
                key: value
                for key, value in agent_overrides.items()
                if key in allowed
            }
        )
    agent_config["system_template"] = system_template or agent_config["system_template"]
    agent_config["instance_template"] = instance_template or agent_config["instance_template"]
    agent_config["step_limit"] = step_limit
    agent_config["wall_time_limit_seconds"] = bounded_task_wall_time(agent_config)
    agent_config["output_path"] = output_dir / "trajectory.json"

    env_config = original["environment"]
    volumes: dict[str, str] = {}
    if harness_assets_dir is not None:
        source_assets = Path(harness_assets_dir)
        if source_assets.is_dir():
            isolated_assets = output_dir / "harness_assets"
            if isolated_assets.exists():
                shutil.rmtree(isolated_assets)
            shutil.copytree(source_assets, isolated_assets)
            volumes[str(isolated_assets.resolve())] = "/workspace/meta_harness"

    env = UdockerEnvironment(
        image=image_for_instance(instance),
        repo=udocker_repo,
        executable=udocker_executable,
        cwd=env_config["cwd"],
        env=env_config["env"],
        timeout=env_config["timeout"],
        interpreter=env_config["interpreter"],
        volumes=volumes,
        pull_timeout=3600,
        container_name=f"agent-{uuid.uuid4().hex[:12]}",
    )

    if not issubclass(agent_class, DefaultAgent):
        raise TypeError("agent_class must subclass minisweagent.agents.default.DefaultAgent")
    agent = agent_class(
        model=build_model(
            api_key=api_key,
            temperature=temperature,
            model_overrides=model_overrides,
        ),
        env=env,
        **agent_config,
    )
    if template_vars:
        agent.extra_template_vars.update(deepcopy(template_vars))
    result: dict = {
        'execution_backend': backend_name(),
        "instance_id": instance["instance_id"],
        "model": TASK_MODEL,
        "temperature": temperature,
        "step_limit": step_limit,
    }
    patch = ""
    try:
        info = agent.run(instance["problem_statement"])
        result["exit_status"] = info.get("exit_status", "")
        patch = info.get("submission", "") or ""
    except Exception as exc:
        result["exit_status"] = type(exc).__name__
        result["exception"] = str(exc)
        result["traceback"] = traceback.format_exc()
    finally:
        try:
            if not patch.strip() and not result.get('exception'):
                diff = env.execute({"command": "git -c core.fileMode=false diff"}, timeout=120)
                if diff["returncode"] == 0:
                    patch = diff["output"]
            agent.save(
                output_dir / "trajectory.json",
                {"instance_id": instance["instance_id"], "info": {"submission": patch}},
            )
            redact_secret(output_dir / "trajectory.json", api_key)
        finally:
            try:
                env.cleanup()
            except Exception as cleanup_exc:
                result["cleanup_error"] = f"{type(cleanup_exc).__name__}: {cleanup_exc}"
                runtime = getattr(env, "runtime", None)
                if getattr(runtime, "sync_failed", False):
                    result["exit_status"] = "RemoteInfrastructureError"
                    result["exception"] = result["cleanup_error"]
                    result["traceback"] = traceback.format_exc()

    patch = normalize_patch_transport(patch)
    (output_dir / "patch.diff").write_text(patch)
    result["api_calls"] = agent.n_calls
    result["patch_chars"] = len(patch)
    if evaluate and not result.get('exception'):
        result["evaluation"] = evaluate_patch(
            instance=instance,
            patch=patch,
            output_dir=output_dir / "evaluation",
            udocker_repo=udocker_repo,
            udocker_executable=udocker_executable,
        )
    (output_dir / "result.json").write_text(json.dumps(result, indent=2))
    return result
