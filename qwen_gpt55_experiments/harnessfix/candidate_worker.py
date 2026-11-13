"""Run one HarnessFix candidate in an isolated mini-swe-agent import context."""

from __future__ import annotations

import argparse
import json
import os
import sys
import traceback
import uuid
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from qwen_gpt55_experiments.common.model_config import (
    TASK_MODEL,
    TASK_BASE_URL,
    TASK_STEP_LIMIT,
    task_api_key,
    task_model_kwargs,
)
from qwen_gpt55_experiments.common.qwen_request_gate import run_task_model_request
from qwen_gpt55_experiments.harnessfix.runtime_protocol import PROTOCOL
UDOCKER_REPO = ROOT / "native_baseline_runs" / "udocker"
UDOCKER_EXECUTABLE = ROOT / "native_baseline_runs" / ".venv" / "bin" / "udocker"
FIXED_STEP_LIMIT = int(os.environ.get("HARNESSFIX_STEP_LIMIT", str(TASK_STEP_LIMIT)))


def install_redacted_serializer(model, api_key: str) -> None:
    """Prevent every intermediate trajectory save from serializing the API key."""
    original_serialize = model.serialize

    def redacted_serialize() -> dict:
        data = original_serialize()
        model_kwargs = (
            data.get("info", {})
            .get("config", {})
            .get("model", {})
            .get("model_kwargs", {})
        )
        if "api_key" in model_kwargs:
            model_kwargs["api_key"] = "<REDACTED>"
        return data

    model.serialize = redacted_serialize


def redact_secret(path: Path, secret: str) -> None:
    if not secret or secret == "EMPTY" or not path.is_file():
        return
    try:
        content = path.read_text()
        if secret in content:
            path.write_text(content.replace(secret, "<REDACTED>"))
    except (OSError, UnicodeDecodeError):
        return


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run one isolated HarnessFix candidate task")
    parser.add_argument("--candidate-dir", type=Path, required=True)
    parser.add_argument("--instance-id")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--check-only", action="store_true")
    return parser.parse_args()


def candidate_config(candidate_dir: Path) -> dict:
    path = (
        candidate_dir
        / "src"
        / "minisweagent"
        / "config"
        / "benchmarks"
        / "swebench.yaml"
    )
    if not path.is_file():
        raise FileNotFoundError(f"Candidate SWE-bench config is missing: {path}")
    config = yaml.safe_load(path.read_text())
    if not isinstance(config, dict):
        raise TypeError(f"Candidate config must be a mapping: {path}")
    for section in ("agent", "environment", "model"):
        if not isinstance(config.get(section), dict):
            raise TypeError(f"Candidate config section {section!r} must be a mapping")
    return config


def activate_candidate(candidate_dir: Path) -> tuple[type, type]:
    source = (candidate_dir / "src").resolve()
    if not source.is_dir():
        raise FileNotFoundError(f"Candidate source directory is missing: {source}")
    sys.path.insert(0, str(source))
    sys.path.insert(1, str(ROOT))

    from minisweagent.agents.default import DefaultAgent
    from minisweagent.models.litellm_model import LitellmModel

    imported = Path(sys.modules["minisweagent"].__file__).resolve()
    if source not in imported.parents:
        raise RuntimeError(f"Imported minisweagent from {imported}, expected under {source}")
    return DefaultAgent, LitellmModel


def finish_candidate(agent, env, output_dir, api_key, result, patch):
    """Always retain diagnostics and clean up, even when diff recovery fails."""
    from qwen_gpt55_experiments.common.result_validity import retryable_result
    try:
        if not patch.strip() and not result.get('exception') and not retryable_result(result):
            try:
                diff = env.execute({'command': 'git -c core.fileMode=false diff'}, timeout=120)
                if diff['returncode'] == 0:
                    patch = diff['output']
            except Exception as exc:
                result.update(exit_status=type(exc).__name__, exception=str(exc), traceback=traceback.format_exc())
    finally:
        try:
            agent.save(output_dir / 'trajectory.json', {'instance_id': result['instance_id'], 'info': {'submission': patch}})
            redact_secret(output_dir / 'trajectory.json', api_key)
        except Exception as exc:
            result.setdefault('exception', f'Trajectory save failed: {type(exc).__name__}: {exc}')
        finally:
            try:
                env.cleanup()
            except Exception as exc:
                result['cleanup_error'] = f'{type(exc).__name__}: {exc}'
    (output_dir / 'patch.diff').write_text(patch)
    result['api_calls'] = agent.n_calls
    result['patch_chars'] = len(patch)
    (output_dir / 'result.json').write_text(json.dumps(result, indent=2))
    return patch


def run_candidate(args: argparse.Namespace) -> dict:
    candidate_dir = args.candidate_dir.resolve()
    config = candidate_config(candidate_dir)
    if config["environment"].get("environment_class", "docker") != "docker":
        raise ValueError("HarnessFix persistent transport currently requires the candidate Docker environment")
    agent_class, model_class = activate_candidate(candidate_dir)
    from qwen_gpt55_experiments.harnessfix.candidate_environment import candidate_environment_class

    environment_class = candidate_environment_class(candidate_dir)

    if args.check_only:
        return {
            "checked": True,
            "candidate_dir": str(candidate_dir),
            "agent_class": f"{agent_class.__module__}.{agent_class.__name__}",
            "model_class": f"{model_class.__module__}.{model_class.__name__}",
            "step_limit": FIXED_STEP_LIMIT,
            "environment_transport": PROTOCOL,
        }
    if not args.instance_id or not args.output_dir:
        raise ValueError("--instance-id and --output-dir are required unless --check-only is used")

    api_key = task_api_key()

    from qwen_gpt55_experiments.common.dataset import load_failed_instances, image_for_instance
    from qwen_gpt55_experiments.common.evaluator import evaluate_patch

    instances = {item["instance_id"]: item for item in load_failed_instances()}
    if args.instance_id not in instances:
        raise KeyError(f"Instance is not in the fixed failure set: {args.instance_id}")
    instance = instances[args.instance_id]
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    agent_config = dict(config["agent"])
    agent_config["step_limit"] = FIXED_STEP_LIMIT
    if os.environ.get('HARNESSFIX_NATIVE_PROFILE') == '1':
        agent_config['wall_time_limit_seconds'] = 3600
    agent_config["output_path"] = output_dir / "trajectory.json"

    model_config = dict(config["model"])
    model = model_class(
        model_name=TASK_MODEL,
        model_kwargs=task_model_kwargs(api_key=api_key),
        cost_tracking="ignore_errors",
        format_error_template=model_config["format_error_template"],
        observation_template=model_config["observation_template"],
        multimodal_regex=model_config.get("multimodal_regex", ""),
    )
    original_query = model._query
    model._query = lambda messages, **kwargs: run_task_model_request(
        lambda: original_query(messages, **kwargs)
    )
    install_redacted_serializer(model, api_key)

    env_config = dict(config["environment"])
    env = environment_class(
        image=image_for_instance(instance),
        repo=UDOCKER_REPO,
        executable=UDOCKER_EXECUTABLE,
        cwd=env_config.get("cwd", "/testbed"),
        env=env_config.get("env", {}),
        forward_env=env_config.get("forward_env", []),
        timeout=env_config.get("timeout", 60),
        interpreter=env_config.get("interpreter", ["bash", "-c"]),
        pull_timeout=3600,
        container_name=f"harnessfix-{uuid.uuid4().hex[:12]}",
    )
    agent = agent_class(model=model, env=env, **agent_config)
    result: dict = {
        "instance_id": args.instance_id,
        "model": TASK_MODEL,
        "temperature": 1.0,
        "step_limit": FIXED_STEP_LIMIT,
        "candidate_dir": str(candidate_dir),
        "environment_transport": PROTOCOL,
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
        patch = finish_candidate(agent, env, output_dir, api_key, result, patch)
    from qwen_gpt55_experiments.common.result_validity import retryable_result
    if result.get('exception') or retryable_result(result):
        return result
    result["evaluation"] = evaluate_patch(
        instance=instance,
        patch=patch,
        output_dir=output_dir / "evaluation",
        udocker_repo=UDOCKER_REPO,
        udocker_executable=UDOCKER_EXECUTABLE,
    )
    (output_dir / "result.json").write_text(json.dumps(result, indent=2))
    return result


def main() -> None:
    args = parse_args()
    result = run_candidate(args)
    print(json.dumps(result, indent=2))
    if not args.check_only and (result.get('exception') or not isinstance(result.get('evaluation', {}).get('resolved'), bool)):
        raise SystemExit(2)


if __name__ == "__main__":
    main()
