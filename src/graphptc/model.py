from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from typing import Any

from openai import APIConnectionError, APIStatusError, APITimeoutError, OpenAI

from .config import ModelConfig


@dataclass(frozen=True)
class TokenUsage:
    input_tokens: int = 0
    output_tokens: int = 0
    cached_input_tokens: int = 0
    reasoning_tokens: int = 0

    def __add__(self, other: "TokenUsage") -> "TokenUsage":
        return TokenUsage(
            input_tokens=self.input_tokens + other.input_tokens,
            output_tokens=self.output_tokens + other.output_tokens,
            cached_input_tokens=self.cached_input_tokens + other.cached_input_tokens,
            reasoning_tokens=self.reasoning_tokens + other.reasoning_tokens,
        )


@dataclass(frozen=True)
class ToolCall:
    id: str
    name: str
    input: dict[str, Any]


@dataclass(frozen=True)
class ModelAttempt:
    attempt: int
    duration_ms: float
    status: str
    status_code: int | None = None
    error_type: str | None = None
    error: str | None = None
    retry_delay_seconds: float | None = None


@dataclass(frozen=True)
class ModelTurn:
    assistant_message: dict[str, Any]
    text: str
    tool_calls: list[ToolCall]
    usage: TokenUsage
    stop_reason: str | None
    attempts: tuple[ModelAttempt, ...] = ()


class OpenAIChatModel:
    """OpenAI-compatible Chat Completions adapter used by MiMo and later GPT runs."""

    def __init__(self, config: ModelConfig, api_key: str) -> None:
        self.config = config
        self._client = OpenAI(
            api_key=api_key,
            base_url=config.base_url,
            max_retries=0,
            timeout=config.timeout_seconds,
        )

    def create_turn(
        self,
        *,
        system: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        timeout_seconds: float | None = None,
        max_completion_tokens: int | None = None,
        thinking: str | None = None,
    ) -> ModelTurn:
        request_messages = list(messages)
        if system:
            request_messages.insert(0, {"role": "system", "content": system})
        request: dict[str, Any] = {
            "model": self.config.model,
            "max_completion_tokens": (
                self.config.max_completion_tokens
                if max_completion_tokens is None
                else max_completion_tokens
            ),
            "messages": request_messages,
        }
        if tools:
            request["tools"] = tools
            request["tool_choice"] = "auto"
        if self.config.temperature is not None:
            request["temperature"] = self.config.temperature
        if self.config.top_p is not None:
            request["top_p"] = self.config.top_p
        effective_thinking = self.config.thinking if thinking is None else thinking
        if effective_thinking:
            request["extra_body"] = {"thinking": {"type": effective_thinking}}

        response, attempts = self._create_with_retries(request, timeout_seconds)
        choice = response.choices[0]
        message = choice.message

        tool_calls: list[ToolCall] = []
        serialized_calls: list[dict[str, Any]] = []
        for call in message.tool_calls or []:
            try:
                arguments = json.loads(call.function.arguments)
            except json.JSONDecodeError:
                arguments = {"_invalid_json": call.function.arguments}
            if not isinstance(arguments, dict):
                arguments = {"_invalid_json": call.function.arguments}
            tool_calls.append(
                ToolCall(id=call.id, name=call.function.name, input=arguments)
            )
            serialized_calls.append(
                {
                    "id": call.id,
                    "type": "function",
                    "function": {
                        "name": call.function.name,
                        "arguments": call.function.arguments,
                    },
                }
            )

        assistant_message: dict[str, Any] = {
            "role": "assistant",
            "content": message.content,
        }
        if serialized_calls:
            assistant_message["tool_calls"] = serialized_calls

        reasoning_content = _extra_message_field(message, "reasoning_content")
        if reasoning_content is not None:
            assistant_message["reasoning_content"] = reasoning_content

        usage_data = response.usage
        prompt_details = getattr(usage_data, "prompt_tokens_details", None)
        completion_details = getattr(usage_data, "completion_tokens_details", None)
        usage = TokenUsage(
            input_tokens=getattr(usage_data, "prompt_tokens", 0) or 0,
            output_tokens=getattr(usage_data, "completion_tokens", 0) or 0,
            cached_input_tokens=getattr(prompt_details, "cached_tokens", 0) or 0,
            reasoning_tokens=getattr(completion_details, "reasoning_tokens", 0) or 0,
        )
        return ModelTurn(
            assistant_message=assistant_message,
            text=(message.content or "").strip(),
            tool_calls=tool_calls,
            usage=usage,
            stop_reason=choice.finish_reason,
            attempts=attempts,
        )

    def _create_with_retries(
        self, request: dict[str, Any], timeout_seconds: float | None
    ) -> tuple[Any, tuple[ModelAttempt, ...]]:
        request_budget = (
            self.config.timeout_seconds if timeout_seconds is None else timeout_seconds
        )
        if request_budget <= 0:
            raise TimeoutError("Model request deadline exhausted")
        deadline = time.monotonic() + request_budget
        attempts: list[ModelAttempt] = []
        attempt = 0
        while self.config.max_retries < 0 or attempt <= self.config.max_retries:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError("Model request deadline exhausted")
            client = self._client.with_options(
                timeout=min(self.config.timeout_seconds, remaining),
                max_retries=0,
            )
            attempt_started = time.perf_counter()
            try:
                response = client.chat.completions.create(**request)
                if self.config.retry_all_errors and not _has_response_payload(response):
                    raise ValueError("Model returned an empty response without tool calls")
                attempts.append(
                    ModelAttempt(
                        attempt=attempt + 1,
                        duration_ms=(time.perf_counter() - attempt_started) * 1_000,
                        status="success",
                    )
                )
                return response, tuple(attempts)
            except Exception as exc:
                retryable = self.config.retry_all_errors or _retryable(exc)
                if (
                    (self.config.max_retries >= 0 and attempt >= self.config.max_retries)
                    or not retryable
                ):
                    attempts.append(_failed_attempt(attempt, attempt_started, exc, None))
                    raise
                remaining = deadline - time.monotonic()
                retry_after = _retry_after_seconds(exc)
                configured_backoff = self.config.retry_backoff_seconds
                delay = min(
                    configured_backoff
                    if configured_backoff is not None
                    else retry_after
                    if retry_after is not None
                    else float(2**attempt),
                    max(0.0, remaining),
                )
                attempts.append(_failed_attempt(attempt, attempt_started, exc, delay))
                if delay <= 0:
                    raise TimeoutError("Model request deadline exhausted") from exc
                time.sleep(delay)
                attempt += 1
        raise AssertionError("unreachable")


def _has_response_payload(response: Any) -> bool:
    choices = getattr(response, "choices", None) or []
    if not choices:
        return False
    message = getattr(choices[0], "message", None)
    if message is None:
        return False
    content = getattr(message, "content", None)
    return bool((isinstance(content, str) and content.strip()) or getattr(message, "tool_calls", None))


def _extra_message_field(message: Any, name: str) -> Any:
    value = getattr(message, name, None)
    if value is not None:
        return value
    extra = getattr(message, "model_extra", None) or {}
    return extra.get(name)


def _retryable(exc: Exception) -> bool:
    if isinstance(exc, (APIConnectionError, APITimeoutError)):
        return True
    if isinstance(exc, APIStatusError):
        return exc.status_code in {408, 409, 429} or exc.status_code >= 500
    return False


def _failed_attempt(
    attempt: int,
    started: float,
    exc: Exception,
    retry_delay: float | None,
) -> ModelAttempt:
    return ModelAttempt(
        attempt=attempt + 1,
        duration_ms=(time.perf_counter() - started) * 1_000,
        status="failed",
        status_code=getattr(exc, "status_code", None),
        error_type=type(exc).__name__,
        error=str(exc),
        retry_delay_seconds=retry_delay,
    )


def _retry_after_seconds(exc: Exception) -> float | None:
    response = getattr(exc, "response", None)
    headers = getattr(response, "headers", None)
    if headers is None:
        return None
    value = headers.get("retry-after")
    if value is None:
        return None
    try:
        return max(0.0, float(value))
    except (TypeError, ValueError):
        return None


def usage_to_dict(usage: TokenUsage) -> dict[str, int]:
    return asdict(usage)
