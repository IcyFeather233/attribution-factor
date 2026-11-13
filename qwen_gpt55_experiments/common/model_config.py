"""Fixed optimizer and task-model endpoints for the Qwen repair experiments."""

from __future__ import annotations

import os
from pathlib import Path


OPTIMIZER_MODEL = os.environ.get("OPTIMIZER_MODEL", "gpt-5.5-ppio")
OPTIMIZER_LITELLM_MODEL = f"openai/{OPTIMIZER_MODEL}"
OPTIMIZER_BASE_URL = os.environ.get("OPTIMIZER_BASE_URL", "http://localhost:8000/v1")

TASK_MODEL = os.environ.get("TASK_MODEL", "qwen3.5-35b-a3b")
TASK_BASE_URL = os.environ.get(
    "QWEN_TASK_BASE_URL",
    "http://localhost:8001/v1",
)
TASK_API_KEY_FILE = (
    Path(__file__).resolve().parents[1] / ".secrets" / "dashscope.env"
)
TASK_API_KEY_SOURCE = "QWEN_TASK_API_KEY or SiFlow placeholder EMPTY"
TASK_TEMPERATURE = 1.0
TASK_TOP_P = 0.95
TASK_TOP_K = 20
TASK_ENABLE_THINKING = True
TASK_STEP_LIMIT = 250


def optimizer_api_key() -> str:
    key = os.environ.get("SIDECAR_API_KEY")
    if not key:
        raise RuntimeError("SIDECAR_API_KEY is required for optimizer/evolve calls")
    return key


def task_api_key() -> str:
    key = os.environ.get("QWEN_TASK_API_KEY")
    if key:
        return key
    if "siflow-changliu.siflow.cn/" in TASK_BASE_URL:
        return "EMPTY"
    key = os.environ.get("DASHSCOPE_API_KEY")
    if not key and TASK_API_KEY_FILE.is_file():
        for line in TASK_API_KEY_FILE.read_text().splitlines():
            stripped = line.strip()
            if not stripped or stripped.startswith("#") or "=" not in stripped:
                continue
            name, value = stripped.split("=", 1)
            if name.strip() in {"DASHSCOPE_API_KEY", "QWEN_TASK_API_KEY"}:
                key = value.strip().strip(chr(34)).strip(chr(39))
                if key:
                    break
    if not key:
        raise RuntimeError(
            "DASHSCOPE_API_KEY or QWEN_TASK_API_KEY is required for task-agent rollouts"
        )
    return key


def task_model_kwargs(*, api_key: str, temperature: float = TASK_TEMPERATURE) -> dict:
    """Return the inference settings used by the original Qwen trajectories."""
    return {
        "api_key": api_key,
        "api_base": TASK_BASE_URL,
        "custom_llm_provider": "openai",
        "drop_params": True,
        "parallel_tool_calls": True,
        "temperature": temperature,
        "timeout": float(os.environ.get("QWEN_TASK_REQUEST_TIMEOUT_SECONDS", "300")),
        "top_p": TASK_TOP_P,
        "extra_body": {
            "top_k": TASK_TOP_K,
            "enable_thinking": TASK_ENABLE_THINKING,
        },
    }
