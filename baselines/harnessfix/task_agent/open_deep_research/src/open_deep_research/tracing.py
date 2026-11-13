from __future__ import annotations

import json
import re
from contextlib import contextmanager
from contextvars import ContextVar
from threading import Lock
from dataclasses import asdict, is_dataclass
from enum import Enum
from pathlib import Path
from typing import Any

from smolagents import LiteLLMModel
from smolagents.models import ChatMessage, TokenUsage, remove_content_after_stop_sequences


SENSITIVE_MODEL_ARGUMENT_KEYS = {"api_key", "authorization", "Authorization"}
SECRET_PATTERNS = (
    re.compile(r"sk-[A-Za-z0-9_\-]+"),
    re.compile(r"Bearer\s+[A-Za-z0-9_\-.]+", re.IGNORECASE),
)

_CURRENT_RECORDER: ContextVar["ModelTraceRecorder | None"] = ContextVar(
    "open_deep_research_model_trace_recorder", default=None
)


def _redact_secret_text(text: str) -> str:
    redacted = text
    for pattern in SECRET_PATTERNS:
        redacted = pattern.sub("<redacted>", redacted)
    return redacted


def _to_jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, float, str)):
        return _redact_secret_text(value) if isinstance(value, str) else value
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Path):
        return str(value)
    if is_dataclass(value) and not isinstance(value, type):
        return _to_jsonable(asdict(value))
    if isinstance(value, dict):
        result: dict[str, Any] = {}
        for key, item in value.items():
            key_str = str(key)
            if key_str in SENSITIVE_MODEL_ARGUMENT_KEYS:
                result[key_str] = "<redacted>"
            else:
                result[key_str] = _to_jsonable(item)
        return result
    if isinstance(value, (list, tuple, set)):
        return [_to_jsonable(item) for item in value]
    if hasattr(value, "model_dump") and callable(value.model_dump):
        try:
            return _to_jsonable(value.model_dump())
        except Exception:
            pass
    if hasattr(value, "dict") and callable(value.dict):
        try:
            return _to_jsonable(value.dict())
        except Exception:
            pass
    if hasattr(value, "__dict__"):
        try:
            return _to_jsonable(vars(value))
        except Exception:
            pass
    return _redact_secret_text(str(value))


def _message_to_dict(message: ChatMessage) -> dict[str, Any]:
    return {
        "role": getattr(message.role, "value", str(message.role)),
        "content": _to_jsonable(message.content),
        "tool_calls": _to_jsonable(message.tool_calls),
    }


def _usage_to_dict(usage: TokenUsage | None) -> dict[str, Any]:
    if usage is None:
        return {}
    return {
        "input_tokens": getattr(usage, "input_tokens", None),
        "output_tokens": getattr(usage, "output_tokens", None),
    }


class ModelTraceRecorder:
    """Per-task collector for GAIA model calls across manager, sub-agent, and tools."""

    def __init__(self, task_id: str):
        self.task_id = task_id
        self.model_calls: list[dict[str, Any]] = []
        self._lock = Lock()

    def append_call(
        self,
        *,
        component: str,
        model: str,
        request_messages: list[dict[str, Any]],
        response_message: dict[str, Any],
        usage: dict[str, Any],
        request_options: dict[str, Any],
        raw_response: Any | None = None,
        error: str | None = None,
    ) -> None:
        with self._lock:
            call_index = len(self.model_calls) + 1
            item = {
                "call_index": call_index,
                "component": component,
                "model": model,
                "request_messages": _to_jsonable(request_messages),
                "response_message": _to_jsonable(response_message),
                "usage": _to_jsonable(usage),
                "request_options": _to_jsonable(request_options),
                "source_ref": f"model_calls[{call_index - 1}]",
            }
            if raw_response is not None:
                item["raw_response"] = _to_jsonable(raw_response)
            if error:
                item["error"] = _redact_secret_text(error)
            self.model_calls.append(item)


@contextmanager
def model_trace_context(task_id: str):
    recorder = ModelTraceRecorder(task_id)
    token = _CURRENT_RECORDER.set(recorder)
    try:
        yield recorder
    finally:
        _CURRENT_RECORDER.reset(token)


def current_model_trace_recorder() -> ModelTraceRecorder | None:
    return _CURRENT_RECORDER.get()


class TracingLiteLLMModel(LiteLLMModel):
    """LiteLLMModel variant that records complete request/response messages."""

    def __init__(
        self,
        *args: Any,
        trace_component: str = "model",
        trace_recorder: ModelTraceRecorder | None = None,
        **kwargs: Any,
    ):
        super().__init__(*args, **kwargs)
        self.trace_component = trace_component
        self.trace_recorder = trace_recorder

    def generate(
        self,
        messages: list[ChatMessage | dict],
        stop_sequences: list[str] | None = None,
        response_format: dict[str, str] | None = None,
        tools_to_call_from: list[Any] | None = None,
        **kwargs: Any,
    ) -> ChatMessage:
        completion_kwargs = self._prepare_completion_kwargs(
            messages=messages,
            stop_sequences=stop_sequences,
            response_format=response_format,
            tools_to_call_from=tools_to_call_from,
            model=self.model_id,
            api_base=self.api_base,
            api_key=self.api_key,
            convert_images_to_image_urls=True,
            custom_role_conversions=self.custom_role_conversions,
            **kwargs,
        )
        request_messages = completion_kwargs.get("messages", [])
        request_options = {key: value for key, value in completion_kwargs.items() if key != "messages"}
        recorder = self.trace_recorder or current_model_trace_recorder()

        try:
            self._apply_rate_limit()
            response = self.retryer(self.client.completion, **completion_kwargs)
            if not response.choices:
                raise RuntimeError(
                    f"Unexpected API response: model '{self.model_id}' returned no choices. "
                    "This may indicate a possible API or upstream issue. "
                    f"Response details: {response.model_dump()}"
                )
            content = response.choices[0].message.content
            if stop_sequences is not None and not self.supports_stop_parameter:
                content = remove_content_after_stop_sequences(content, stop_sequences)
            message = ChatMessage(
                role=response.choices[0].message.role,
                content=content,
                tool_calls=response.choices[0].message.tool_calls,
                raw=response,
                token_usage=TokenUsage(
                    input_tokens=response.usage.prompt_tokens,
                    output_tokens=response.usage.completion_tokens,
                ),
            )
            if recorder is not None:
                recorder.append_call(
                    component=self.trace_component,
                    model=str(completion_kwargs.get("model", self.model_id)),
                    request_messages=request_messages,
                    response_message=_message_to_dict(message),
                    usage=_usage_to_dict(message.token_usage),
                    request_options=request_options,
                    raw_response={
                        "id": getattr(response, "id", None),
                        "model": getattr(response, "model", None),
                        "created": getattr(response, "created", None),
                        "usage": getattr(response, "usage", None),
                        "finish_reason": response.choices[0].finish_reason,
                    },
                )
            return message
        except Exception as exc:
            if recorder is not None:
                recorder.append_call(
                    component=self.trace_component,
                    model=str(completion_kwargs.get("model", self.model_id)),
                    request_messages=request_messages,
                    response_message={},
                    usage={},
                    request_options=request_options,
                    error=repr(exc),
                )
            raise
