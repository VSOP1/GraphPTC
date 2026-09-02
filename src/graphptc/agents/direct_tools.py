from __future__ import annotations

import json
import time
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict
from typing import Any

from ..config import RuntimeConfig
from ..model import usage_to_dict
from .original_ptc import (
    AgentResult,
    MessagesModel,
    ModelRequestTrace,
)

DIRECT_FINALIZE_PROMPT = """All benchmark tools are now unavailable. Use only the observations
already present in the conversation and return the best final answer in the format requested by the
task. Do not call tools or propose additional tool use."""


class DirectToolAgent:
    """Benchmark-neutral native function-calling baseline."""

    def __init__(
        self,
        *,
        model: MessagesModel,
        runtime: RuntimeConfig,
        system_prompt: str,
        user_prompt_template: str,
        functions: Mapping[str, Callable[..., Any]],
        tool_specs: Sequence[Mapping[str, Any]],
        finalize_prompt: str = DIRECT_FINALIZE_PROMPT,
    ) -> None:
        self._model = model
        self._runtime = runtime
        self._system_prompt = system_prompt
        self._user_prompt_template = user_prompt_template
        self._functions = dict(functions)
        self._tool_specs = [dict(spec) for spec in tool_specs]
        self._finalize_prompt = finalize_prompt
        _validate_tool_contract(self._functions, self._tool_specs)

    def run(self, task: str) -> AgentResult:
        started = time.perf_counter()
        result = AgentResult()
        messages: list[dict[str, Any]] = [
            {
                "role": "user",
                "content": self._user_prompt_template.format(task=task),
            }
        ]
        force_finalize = False
        finalization_requested = False
        tool_rounds = 0
        direct_calls = 0
        observation_chars = 0
        observation_truncations = 0
        tool_errors = 0
        tool_call_distribution: Counter[str] = Counter()
        tool_call_trace: list[dict[str, Any]] = []

        try:
            for turn_number in range(1, self._runtime.max_turns + 1):
                remaining = self._runtime.task_timeout_seconds - (
                    time.perf_counter() - started
                )
                if remaining <= 0:
                    result.finish_reason = "task_timeout"
                    result.error = "Task wall-clock budget exhausted before a final answer"
                    break
                if (
                    self._runtime.max_total_output_tokens is not None
                    and result.usage.output_tokens >= self._runtime.max_total_output_tokens
                ):
                    force_finalize = True
                    result.budget_trigger = result.budget_trigger or "total_output_tokens"
                tools_available = not force_finalize and turn_number < self._runtime.max_turns
                if not tools_available and not finalization_requested:
                    messages.append({"role": "user", "content": self._finalize_prompt})
                    finalization_requested = True

                context_chars = len(
                    json.dumps(messages, ensure_ascii=False, separators=(",", ":"))
                )
                request_started = time.perf_counter()
                result.model_requests += 1
                turn = self._model.create_turn(
                    system=self._system_prompt,
                    messages=messages,
                    tools=self._tool_specs if tools_available else [],
                    timeout_seconds=remaining,
                    max_completion_tokens=(
                        self._runtime.finalization_max_tokens
                        if finalization_requested
                        else None
                    ),
                )
                result.usage = result.usage + turn.usage
                result.requests.append(
                    ModelRequestTrace(
                        turn=turn_number,
                        kind="agent",
                        tools_available=tools_available,
                        context_chars=context_chars,
                        duration_ms=(time.perf_counter() - request_started) * 1_000,
                        stop_reason=turn.stop_reason,
                        tool_calls=len(turn.tool_calls),
                        usage=usage_to_dict(turn.usage),
                        attempts=[asdict(attempt) for attempt in turn.attempts],
                    )
                )
                messages.append(turn.assistant_message)

                if not turn.tool_calls:
                    if (
                        turn.stop_reason in {"length", "max_tokens"}
                        and not finalization_requested
                        and turn_number < self._runtime.max_turns
                    ):
                        force_finalize = True
                        result.budget_trigger = result.budget_trigger or "completion_tokens"
                        continue
                    result.answer = turn.text
                    result.finish_reason = turn.stop_reason
                    if turn.text and turn.stop_reason == "stop":
                        result.status = "success"
                    elif turn.text:
                        result.error = (
                            "Model did not finish normally "
                            f"(finish_reason={turn.stop_reason})"
                        )
                    else:
                        result.error = f"Model stopped without an answer ({turn.stop_reason})"
                    break

                tool_rounds += 1
                for call in turn.tool_calls:
                    direct_calls += 1
                    tool_call_distribution[call.name] += 1
                    tool_started = time.perf_counter()
                    error_type: str | None = None
                    try:
                        function = self._functions[call.name]
                        value = function(**call.input)
                        observation = _serialize_tool_result(value)
                    except Exception as exc:  # noqa: BLE001 - tool failures are observations
                        tool_errors += 1
                        error_type = type(exc).__name__
                        observation = "DIRECT_TOOL_ERROR " + json.dumps(
                            {
                                "tool": call.name,
                                "error_type": type(exc).__name__,
                                "message": str(exc),
                            },
                            ensure_ascii=False,
                            separators=(",", ":"),
                        )
                    full_observation_chars = len(observation)
                    if full_observation_chars > self._runtime.max_stdout_chars:
                        observation_truncations += 1
                        observation = _truncate_tool_result(
                            observation,
                            self._runtime.max_stdout_chars,
                        )
                    observation_chars += full_observation_chars
                    tool_call_trace.append(
                        {
                            "turn": turn_number,
                            "tool_call_id": call.id,
                            "tool": call.name,
                            "success": error_type is None,
                            "error_type": error_type,
                            "duration_ms": (time.perf_counter() - tool_started) * 1_000,
                            "observation_chars": full_observation_chars,
                            "observation_truncated": (
                                full_observation_chars > self._runtime.max_stdout_chars
                            ),
                        }
                    )
                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": call.id,
                            "content": observation,
                        }
                    )
            else:
                result.error = "Model turn budget exhausted before a final answer"
        except Exception as exc:  # noqa: BLE001 - task failures belong in AgentResult
            result.error = f"{type(exc).__name__}: {exc}"
        finally:
            result.duration_ms = (time.perf_counter() - started) * 1_000
            result.runtime_session = {
                "mode": "direct_tool_calling",
                "persistent": False,
                "tool_rounds": tool_rounds,
                "direct_tool_calls": direct_calls,
                "tool_errors": tool_errors,
                "tool_call_distribution": dict(tool_call_distribution),
                "tool_call_trace": tool_call_trace,
                "tool_observation_chars": observation_chars,
                "tool_observation_truncations": observation_truncations,
            }
        return result


def _serialize_tool_result(value: Any) -> str:
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False, default=repr)


def _truncate_tool_result(value: str, maximum: int) -> str:
    notice = "DIRECT_TOOL_OUTPUT_TRUNCATED " + json.dumps(
        {"original_chars": len(value), "limit_chars": maximum},
        separators=(",", ":"),
    ) + "\n"
    if len(notice) >= maximum:
        return notice[:maximum]
    return notice + value[: maximum - len(notice)]


def _validate_tool_contract(
    functions: Mapping[str, Callable[..., Any]],
    tool_specs: Sequence[Mapping[str, Any]],
) -> None:
    declared: list[str] = []
    for index, spec in enumerate(tool_specs):
        function = spec.get("function")
        name = function.get("name") if isinstance(function, Mapping) else None
        if spec.get("type") != "function" or not isinstance(name, str) or not name:
            raise ValueError(f"tool_specs[{index}] is not a named function tool")
        declared.append(name)
    duplicates = sorted(name for name, count in Counter(declared).items() if count > 1)
    if duplicates:
        raise ValueError(f"duplicate direct-tool specs: {duplicates}")
    missing = sorted(set(declared) - set(functions))
    extra = sorted(set(functions) - set(declared))
    if missing or extra:
        raise ValueError(
            "direct-tool function/spec mismatch "
            f"(missing_functions={missing}, missing_specs={extra})"
        )
