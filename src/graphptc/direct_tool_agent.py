from __future__ import annotations

import json
import time
from dataclasses import asdict
from typing import Any, Callable, Mapping

from .config import RuntimeConfig
from .model import usage_to_dict
from .ptc import AgentResult, FINALIZE_PROMPT, ModelRequestTrace


class DirectToolAgent:
    """Native function-calling baseline with search and fetch exposed directly."""

    def __init__(
        self,
        *,
        model: Any,
        search_tools: Any,
        runtime: RuntimeConfig,
        system_prompt: str,
        user_prompt_template: str,
        functions: Mapping[str, Callable[..., Any]],
        tool_specs: list[dict[str, Any]],
    ) -> None:
        self._model = model
        self._search_tools = search_tools
        self._runtime = runtime
        self._system_prompt = system_prompt
        self._user_prompt_template = user_prompt_template
        self._functions = dict(functions)
        self._tool_specs = tool_specs

    def run(self, task: str) -> AgentResult:
        started = time.perf_counter()
        result = AgentResult()
        messages: list[dict[str, Any]] = [
            {
                "role": "user",
                "content": self._user_prompt_template.format(question=task),
            }
        ]
        force_finalize = False
        finalization_requested = False
        tool_rounds = 0
        direct_calls = 0
        observation_chars = 0

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
                    messages.append({"role": "user", "content": FINALIZE_PROMPT})
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
                    thinking="disabled" if finalization_requested else None,
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
                    try:
                        function = self._functions[call.name]
                        value = function(**call.input)
                        observation = json.dumps(value, ensure_ascii=False)
                    except Exception as exc:
                        observation = "DIRECT_TOOL_ERROR " + json.dumps(
                            {
                                "tool": call.name,
                                "error_type": type(exc).__name__,
                                "message": str(exc),
                            },
                            ensure_ascii=False,
                            separators=(",", ":"),
                        )
                    observation_chars += len(observation)
                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": call.id,
                            "content": observation,
                        }
                    )
            else:
                result.error = "Model turn budget exhausted before a final answer"
        except Exception as exc:
            result.error = f"{type(exc).__name__}: {exc}"
        finally:
            result.duration_ms = (time.perf_counter() - started) * 1_000
            result.search_calls = self._search_tools.calls
            result.runtime_session = {
                "mode": "direct_tool_calling",
                "persistent": False,
                "tool_rounds": tool_rounds,
                "direct_tool_calls": direct_calls,
                "tool_observation_chars": observation_chars,
            }
        return result
