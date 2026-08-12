from __future__ import annotations

import ast
import copy
import json
import re
import time
from dataclasses import asdict, dataclass, field
from collections.abc import Callable, Iterable
from typing import Any, Mapping, Protocol

from toolregistry import ToolRegistry

from .config import RuntimeConfig
from .model import ModelTurn, TokenUsage, usage_to_dict
from .search import TavilySearchTools


PTC_TOOL_SPEC: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "programmatic_tool_call",
        "description": (
            "Execute one Python program that can call the web research functions described in "
            "the system prompt multiple times. Intermediate tool results remain inside the program; "
            "only stdout is returned. Use print() for concise evidence needed by your next reasoning step."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "code": {
                    "type": "string",
                    "description": "Python source code. Registered research functions are globals.",
                }
            },
            "required": ["code"],
            "additionalProperties": False,
        },
        "strict": True,
    },
}


SYSTEM_PROMPT = """You are a research agent with access to programmatic_tool_call. Its Python
environment provides these global functions:

- search_web(*, query: str, max_results: int = 10) -> list[dict]
- search_web_batch(*, queries: list[str], max_results: int = 10) -> dict[str, list[dict]]
- fetch_url(*, url: str, query: str = "", max_chars: int = 1000000) -> dict
- fetch_urls(*, urls: list[str], query: str = "", max_chars: int = 1000000) -> list[dict]

Search results contain title, url, content, and score. Fetched pages contain url, content, and a
truncated flag. The program may import safe computation modules such as json, re, collections,
itertools, statistics, datetime, and csv. Always call runtime tools with keyword arguments. Batch
functions accept at most 20 unique queries or URLs; split larger work into multiple calls. You must
not access files, the shell, environment variables, or the network except through these runtime tools.
Use batch functions for independent queries. Keep stdout focused: print distilled facts and source
URLs, not entire pages. You decide whether tools are needed and how many PTC blocks to generate. Each
block must be a self-contained research program because local Python state does not persist between
blocks."""


USER_PROMPT_TEMPLATE = """I want you to answer the following question.

<question>{question}</question>

First plan out your response. This part can be as long as needed. You may need to run many searches,
this is totally fine.
Then provide a short and concise answer in <result> tags. For questions expecting multiple answers,
separate them with commas."""

FINALIZE_PROMPT = """The research phase is complete. All research tools are now unavailable.
Do not emit tool-call markup, function names, Python code, or a plan for more research. Synthesize the
best final answer supported by the evidence already present in the conversation. Answer the user's
question directly and concisely inside <result> and </result> tags."""

COMPACTION_SYSTEM_PROMPT = """Summarize an in-progress research trajectory for the same agent.
Do not answer the research question and do not call tools. Begin by preserving the original question
verbatim inside <original_question> tags. Preserve the required answer format, verified facts with
source identifiers, candidate and excluded answers, attempted searches, remaining uncertainty, and
the next research direction. Omit raw tool output and private reasoning. Return only a compact
<compacted_state>...</compacted_state> block."""

COMPACTION_CONTINUE_PROMPT = """Continue the original research task from the compacted state. Use
programmatic_tool_call when more evidence is needed, and return the final answer in <result> tags."""

COMPACTION_REQUEST_PROMPT = """Create the compacted state now. Follow the compaction instructions
exactly, do not call tools, and do not answer the research question."""


_RESULT_TAG_RE = re.compile(r"<result>(.*?)</result>", re.DOTALL | re.IGNORECASE)


def extract_result_tag(text: str) -> str | None:
    """Return the contents of the last complete result tag."""
    matches = _RESULT_TAG_RE.findall(text)
    if not matches:
        return None
    result = matches[-1].strip()
    return result or None


class MessagesModel(Protocol):
    def create_turn(
        self,
        *,
        system: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        timeout_seconds: float | None = None,
        max_completion_tokens: int | None = None,
        thinking: str | None = None,
    ) -> ModelTurn: ...


@dataclass(frozen=True)
class PTCBlockTrace:
    turn: int
    tool_call_id: str | None
    code: str
    stdout: str
    stdout_chars: int
    stdout_truncated: bool
    success: bool
    duration_ms: float
    invocation_id: str | None
    runtime_calls: int
    program_analysis: dict[str, Any] = field(default_factory=dict)
    runtime_trace: dict[str, Any] = field(default_factory=dict)
    error_type: str | None = None
    error_message: str | None = None


@dataclass(frozen=True)
class ModelRequestTrace:
    turn: int
    kind: str
    tools_available: bool
    context_chars: int
    duration_ms: float
    stop_reason: str | None
    tool_calls: int
    usage: dict[str, int]
    attempts: list[dict[str, Any]] = field(default_factory=list)


@dataclass(frozen=True)
class CompactionTrace:
    turn: int
    success: bool
    before_chars: int
    after_chars: int
    estimated_tokens_before: int
    summary_chars: int
    duration_ms: float
    usage: dict[str, int]
    error: str | None = None


@dataclass
class AgentResult:
    answer: str = ""
    status: str = "failed"
    error: str | None = None
    finish_reason: str | None = None
    duration_ms: float = 0.0
    model_requests: int = 0
    compaction_requests: int = 0
    ptc_blocks: int = 0
    usage: TokenUsage = field(default_factory=TokenUsage)
    blocks: list[PTCBlockTrace] = field(default_factory=list)
    requests: list[ModelRequestTrace] = field(default_factory=list)
    compactions: list[CompactionTrace] = field(default_factory=list)
    search_calls: list[dict[str, Any]] = field(default_factory=list)
    runtime_session: dict[str, Any] = field(default_factory=dict)
    budget_trigger: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "answer": self.answer,
            "status": self.status,
            "error": self.error,
            "finish_reason": self.finish_reason,
            "duration_ms": self.duration_ms,
            "model_requests": self.model_requests,
            "compaction_requests": self.compaction_requests,
            "ptc_blocks": self.ptc_blocks,
            "usage": usage_to_dict(self.usage),
            "blocks": [asdict(block) for block in self.blocks],
            "requests": [asdict(request) for request in self.requests],
            "compactions": [asdict(item) for item in self.compactions],
            "search_calls": self.search_calls,
            "runtime_session": self.runtime_session,
            "budget_trigger": self.budget_trigger,
        }


class OriginalPTCAgent:
    """Original PTC loop with local subprocess execution and no managed container."""

    def __init__(
        self,
        model: MessagesModel,
        search_tools: TavilySearchTools,
        runtime: RuntimeConfig,
        *,
        system_prompt: str = SYSTEM_PROMPT,
        user_prompt_template: str = USER_PROMPT_TEMPLATE,
        runtime_functions: Iterable[Callable[..., Any]] | None = None,
        checkpoint_callback: Callable[[dict[str, Any]], None] | None = None,
        ptc_tool_spec: dict[str, Any] = PTC_TOOL_SPEC,
        demonstration_messages: Iterable[dict[str, Any]] = (),
        post_block_message_factory: Callable[[PTCBlockTrace], str | None] | None = None,
        post_block_message_on_error: bool = False,
        block_observation_factory: Callable[[PTCBlockTrace], str | None] | None = None,
        ptc_call_metadata_callback: (
            Callable[[dict[str, Any]], Mapping[str, Any] | None] | None
        ) = None,
        adaptation_initial_observation: Callable[[], str] | None = None,
        message_projection_callback: Callable[[list[dict[str, Any]]], None] | None = None,
    ) -> None:
        self._model = model
        self._search_tools = search_tools
        self._runtime = runtime
        self._system_prompt = system_prompt
        self._user_prompt_template = user_prompt_template
        functions = tuple(
            runtime_functions
            or (
                search_tools.search_web,
                search_tools.search_web_batch,
                search_tools.fetch_url,
                search_tools.fetch_urls,
            )
        )
        self._runtime_tool_names = {function.__name__ for function in functions}
        self._registry = self._create_registry(functions)
        self._checkpoint_callback = checkpoint_callback
        self._ptc_tool_spec = ptc_tool_spec
        self._demonstration_messages = tuple(demonstration_messages)
        self._post_block_message_factory = post_block_message_factory
        self._post_block_message_on_error = post_block_message_on_error
        self._block_observation_factory = block_observation_factory
        self._ptc_call_metadata_callback = ptc_call_metadata_callback
        self._adaptation_initial_observation = adaptation_initial_observation
        self._message_projection_callback = message_projection_callback

    def run(self, task: str) -> AgentResult:
        started = time.perf_counter()
        result = AgentResult()
        task_message = {
            "role": "user",
            "content": self._user_prompt_template.format(question=task),
        }
        messages: list[dict[str, Any]] = [
            *copy.deepcopy(self._demonstration_messages),
            task_message,
        ]
        if self._adaptation_initial_observation is not None:
            messages.append(
                {"role": "user", "content": self._adaptation_initial_observation()}
            )
        force_finalize = False
        finalization_requested = False

        try:
            for turn_number in range(1, self._runtime.max_turns + 1):
                remaining = self._runtime.task_timeout_seconds - (
                    time.perf_counter() - started
                )
                if remaining <= 0:
                    result.finish_reason = "task_timeout"
                    result.error = (
                        "Task wall-clock budget exhausted before a final answer "
                        f"({self._runtime.task_timeout_seconds:g}s)"
                    )
                    break
                if (
                    self._runtime.max_total_output_tokens is not None
                    and result.usage.output_tokens
                    >= self._runtime.max_total_output_tokens
                ):
                    force_finalize = True
                    result.budget_trigger = result.budget_trigger or "total_output_tokens"
                tools_available = (
                    not force_finalize
                    and result.ptc_blocks < self._runtime.max_ptc_blocks
                    and turn_number < self._runtime.max_turns
                )
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
                    tools=[self._ptc_tool_spec] if tools_available else [],
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
                if time.perf_counter() - started >= self._runtime.task_timeout_seconds:
                    result.finish_reason = "task_timeout"
                    result.error = (
                        "Task wall-clock budget exhausted during a model request "
                        f"({self._runtime.task_timeout_seconds:g}s)"
                    )
                    break
                messages.append(turn.assistant_message)

                if not turn.tool_calls:
                    needs_finalization = turn.stop_reason in {
                        "length",
                        "max_tokens",
                        "repetition_truncation",
                    } or (turn.stop_reason == "stop" and not turn.text)
                    if (
                        needs_finalization
                        and not finalization_requested
                        and turn_number < self._runtime.max_turns
                    ):
                        messages = self._maybe_compact(
                            messages,
                            result,
                            turn_number=turn_number,
                            started=started,
                            task_message=task_message,
                            force=True,
                        )
                        force_finalize = True
                        continue
                    if not tools_available and _looks_like_textual_tool_call(turn.text):
                        result.answer = turn.text
                        result.finish_reason = turn.stop_reason
                        result.error = (
                            "Model emitted textual tool-call markup after "
                            "the PTC budget was exhausted"
                        )
                        break
                    result.finish_reason = turn.stop_reason
                    result.answer = turn.text
                    if not turn.text:
                        result.status = "failed"
                        result.error = f"Model stopped without an answer ({turn.stop_reason})"
                    elif turn.stop_reason != "stop":
                        result.status = "failed"
                        result.error = (
                            "Model did not finish normally "
                            f"(finish_reason={turn.stop_reason})"
                        )
                    else:
                        result.status = "success"
                    break

                for call in turn.tool_calls:
                    trace: PTCBlockTrace | None = None
                    if finalization_requested:
                        output = "Error: research tools are unavailable during finalization"
                        is_error = True
                    elif call.name != "programmatic_tool_call":
                        output = f"Error: unknown tool: {call.name}"
                        is_error = True
                    elif result.ptc_blocks >= self._runtime.max_ptc_blocks:
                        output = "Error: PTC block budget exhausted"
                        is_error = True
                    else:
                        remaining = self._runtime.task_timeout_seconds - (
                            time.perf_counter() - started
                        )
                        if remaining <= 0:
                            result.finish_reason = "task_timeout"
                            result.error = (
                                "Task wall-clock budget exhausted before a PTC block "
                                f"({self._runtime.task_timeout_seconds:g}s)"
                            )
                            break
                        self._active_code_timeout_seconds = min(
                            self._runtime.code_timeout_seconds, remaining
                        )
                        self._active_tool_call_id = call.id
                        call_input = dict(call.input)
                        if self._ptc_call_metadata_callback is not None:
                            prepared = self._ptc_call_metadata_callback(call_input)
                            if prepared is not None:
                                call_input.update(prepared)
                        output, is_error, trace = self._execute_block(
                            turn_number, call_input.get("code")
                        )
                        result.blocks.append(trace)
                        result.ptc_blocks += 1
                    if is_error and not output.startswith(("Error:", "PTC_ERROR ")):
                        output = f"Error: {output}"
                    if trace is not None and self._block_observation_factory is not None:
                        block_observation = self._block_observation_factory(trace)
                        if block_observation:
                            output = f"{output.rstrip()}\n\n{block_observation}"
                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": call.id,
                            "content": output,
                        }
                    )
                    if trace is not None and self._message_projection_callback is not None:
                        self._message_projection_callback(messages)
                    if (
                        (not is_error or self._post_block_message_on_error)
                        and trace is not None
                        and self._post_block_message_factory is not None
                    ):
                        post_block_message = self._post_block_message_factory(trace)
                        if post_block_message:
                            messages.append(
                                {"role": "user", "content": post_block_message}
                            )
                if result.finish_reason == "task_timeout":
                    break
                self._checkpoint(messages, result, turn_number + 1)
                if finalization_requested:
                    result.finish_reason = turn.stop_reason
                    result.error = "Model attempted a tool call during finalization"
                    break
                messages = self._maybe_compact(
                    messages,
                    result,
                    turn_number=turn_number,
                    started=started,
                    task_message=task_message,
                )
            else:
                result.error = "Model turn budget exhausted before a final answer"
        except Exception as exc:
            result.error = f"{type(exc).__name__}: {exc}"
        finally:
            result.duration_ms = (time.perf_counter() - started) * 1_000
            result.search_calls = self._search_tools.calls

        return result

    def _maybe_compact(
        self,
        messages: list[dict[str, Any]],
        result: AgentResult,
        *,
        turn_number: int,
        started: float,
        task_message: dict[str, Any],
        force: bool = False,
    ) -> list[dict[str, Any]]:
        trigger = self._runtime.compaction_trigger_input_tokens
        if trigger is None or len(result.compactions) >= self._runtime.max_compactions:
            return messages
        if any(not item.success for item in result.compactions):
            return messages
        estimated_tokens = _estimate_context_tokens(messages)
        if result.requests:
            last_usage = result.requests[-1].usage
            estimated_tokens = max(
                estimated_tokens,
                int(last_usage.get("input_tokens", 0))
                + int(last_usage.get("output_tokens", 0)),
            )
        if not force and estimated_tokens < trigger:
            return messages

        remaining = self._runtime.task_timeout_seconds - (time.perf_counter() - started)
        if remaining <= 0:
            return messages
        before_chars = len(
            json.dumps(messages, ensure_ascii=False, separators=(",", ":"))
        )
        summary_messages = [
            *messages,
            {"role": "user", "content": COMPACTION_REQUEST_PROMPT},
        ]
        request_started = time.perf_counter()
        result.model_requests += 1
        result.compaction_requests += 1
        try:
            turn = self._model.create_turn(
                system=COMPACTION_SYSTEM_PROMPT,
                messages=summary_messages,
                tools=[],
                timeout_seconds=remaining,
                max_completion_tokens=self._runtime.compaction_max_tokens,
                thinking="disabled",
            )
            duration_ms = (time.perf_counter() - request_started) * 1_000
            result.usage = result.usage + turn.usage
            result.requests.append(
                ModelRequestTrace(
                    turn=turn_number,
                    kind="compaction",
                    tools_available=False,
                    context_chars=before_chars,
                    duration_ms=duration_ms,
                    stop_reason=turn.stop_reason,
                    tool_calls=len(turn.tool_calls),
                    usage=usage_to_dict(turn.usage),
                    attempts=[asdict(attempt) for attempt in turn.attempts],
                )
            )
            success = bool(turn.text) and not turn.tool_calls and turn.stop_reason == "stop"
            if success:
                compacted = [
                    task_message,
                    {"role": "assistant", "content": turn.text},
                    {"role": "user", "content": COMPACTION_CONTINUE_PROMPT},
                ]
                after_chars = len(
                    json.dumps(compacted, ensure_ascii=False, separators=(",", ":"))
                )
                error = None
            else:
                compacted = messages
                after_chars = before_chars
                error = (
                    "Compaction model did not return a complete text summary "
                    f"(finish_reason={turn.stop_reason})"
                )
            result.compactions.append(
                CompactionTrace(
                    turn=turn_number,
                    success=success,
                    before_chars=before_chars,
                    after_chars=after_chars,
                    estimated_tokens_before=estimated_tokens,
                    summary_chars=len(turn.text),
                    duration_ms=duration_ms,
                    usage=usage_to_dict(turn.usage),
                    error=error,
                )
            )
        except Exception as exc:
            duration_ms = (time.perf_counter() - request_started) * 1_000
            result.compactions.append(
                CompactionTrace(
                    turn=turn_number,
                    success=False,
                    before_chars=before_chars,
                    after_chars=before_chars,
                    estimated_tokens_before=estimated_tokens,
                    summary_chars=0,
                    duration_ms=duration_ms,
                    usage=usage_to_dict(TokenUsage()),
                    error=f"{type(exc).__name__}: {exc}",
                )
            )
            compacted = messages
        self._checkpoint(compacted, result, turn_number + 1)
        return compacted

    def _checkpoint(
        self,
        messages: list[dict[str, Any]],
        result: AgentResult,
        next_turn: int,
    ) -> None:
        if self._checkpoint_callback is None:
            return
        self._checkpoint_callback(
            {
                "schema_version": 1,
                "next_turn": next_turn,
                "messages": messages,
                "agent": result.to_dict(),
            }
        )

    def _execute_block(
        self, turn_number: int, code: Any
    ) -> tuple[str, bool, PTCBlockTrace]:
        started = time.perf_counter()
        calls_before = len(self._search_tools.calls)
        invocation_id = None
        analysis = _analyze_program(code, self._runtime_tool_names)
        error_type = None
        error_message = None
        if not isinstance(code, str) or not code.strip():
            output = "PTC code must be a non-empty string"
            success = False
        else:
            try:
                output = self._registry.invoke(
                    "programmatic_tool_call",
                    {
                        "code": code,
                        "timeout": getattr(
                            self,
                            "_active_code_timeout_seconds",
                            self._runtime.code_timeout_seconds,
                        ),
                    },
                )
                output = str(output)
                invocation_id = self._registry.ptc.last_invocation_id
                success = bool(output.strip()) and not output.lstrip().startswith("Error:")
                if not output.strip():
                    output = "PTC program produced no stdout"
            except Exception as exc:
                output = f"{type(exc).__name__}: {exc}"
                success = False

        if not success:
            error_type, error_message = _runtime_error_details(output)
            output = "PTC_ERROR " + json.dumps(
                {
                    "stage": "programmatic_tool_call",
                    "error_type": error_type,
                    "message": error_message,
                },
                ensure_ascii=False,
                separators=(",", ":"),
            )

        stdout_chars = len(output)
        stdout_truncated = stdout_chars > self._runtime.max_stdout_chars
        output = _truncate(output, self._runtime.max_stdout_chars)
        runtime_calls = len(self._search_tools.calls) - calls_before
        trace = PTCBlockTrace(
            turn=turn_number,
            tool_call_id=getattr(self, "_active_tool_call_id", None),
            code=code if isinstance(code, str) else repr(code),
            stdout=output,
            stdout_chars=stdout_chars,
            stdout_truncated=stdout_truncated,
            success=success,
            duration_ms=(time.perf_counter() - started) * 1_000,
            invocation_id=invocation_id,
            runtime_calls=runtime_calls,
            program_analysis=analysis,
            error_type=error_type,
            error_message=error_message,
        )
        return output, not success, trace

    def _create_registry(
        self,
        functions: Iterable[Callable[..., Any]],
    ) -> ToolRegistry:
        registry = ToolRegistry()
        for function in functions:
            registry.register(function)
        registry.ptc.enable(timeout=self._runtime.code_timeout_seconds)
        return registry


def _truncate(value: str, maximum: int) -> str:
    if len(value) <= maximum:
        return value
    omitted = len(value) - maximum
    notice = (
        "PTC_STDOUT_TRUNCATED "
        + json.dumps(
            {
                "original_chars": len(value),
                "limit_chars": maximum,
                "omitted_at_least": omitted,
                "guidance": (
                    "The observation is incomplete. Filter and aggregate inside Python before "
                    "doing more research."
                ),
            },
            separators=(",", ":"),
        )
        + "\n"
    )
    if len(notice) >= maximum:
        return notice[:maximum]
    return notice + value[: maximum - len(notice)]


def _looks_like_textual_tool_call(value: str) -> bool:
    lowered = value.lower()
    return "<tool_call" in lowered or (
        "programmatic_tool_call" in lowered
        and ("<function" in lowered or "<parameter" in lowered)
    )


def _estimate_context_tokens(messages: list[dict[str, Any]]) -> int:
    serialized = json.dumps(messages, ensure_ascii=False, separators=(",", ":"))
    return max(1, (len(serialized) + 3) // 4)


def _runtime_error_details(output: str) -> tuple[str, str]:
    if output == "PTC program produced no stdout":
        return "NoStdout", output
    match = re.search(r"\b([A-Za-z_][A-Za-z0-9_]*(?:Error|Exception))\s*:\s*(.+)", output, re.DOTALL)
    if match:
        return match.group(1), match.group(2).strip()
    if output.startswith("PTC code must"):
        return "InvalidCode", output
    return "ExecutionError", output.strip()


class _ProgramVisitor(ast.NodeVisitor):
    def __init__(self, tool_names: set[str]) -> None:
        self.tool_names = tool_names
        self.stack: list[ast.AST] = []
        self.tool_calls = 0
        self.tool_calls_in_loops = 0
        self.conditional_tool_calls = 0
        self.print_calls = 0
        self.has_dedup = False
        self.has_filter = False
        self.has_aggregation = False

    def generic_visit(self, node: ast.AST) -> None:
        self.stack.append(node)
        super().generic_visit(node)
        self.stack.pop()

    def visit_Call(self, node: ast.Call) -> None:
        name = node.func.id if isinstance(node.func, ast.Name) else None
        attribute = node.func.attr if isinstance(node.func, ast.Attribute) else None
        if name in self.tool_names:
            self.tool_calls += 1
            if any(isinstance(parent, (ast.For, ast.While, ast.comprehension)) for parent in self.stack):
                self.tool_calls_in_loops += 1
            if any(isinstance(parent, (ast.If, ast.IfExp)) for parent in self.stack):
                self.conditional_tool_calls += 1
        if name == "print":
            self.print_calls += 1
        if name == "set" or attribute in {"add", "setdefault"}:
            self.has_dedup = True
        if name in {"sorted", "sum", "len", "min", "max", "any", "all"} or attribute in {
            "sort",
            "most_common",
        }:
            self.has_aggregation = True
        self.generic_visit(node)

    def visit_If(self, node: ast.If) -> None:
        self.has_filter = True
        self.generic_visit(node)

    def visit_Compare(self, node: ast.Compare) -> None:
        if any(isinstance(operator, (ast.In, ast.NotIn)) for operator in node.ops):
            self.has_dedup = True
        self.generic_visit(node)

    def visit_comprehension(self, node: ast.comprehension) -> None:
        if node.ifs:
            self.has_filter = True
        self.generic_visit(node)

    def visit_SetComp(self, node: ast.SetComp) -> None:
        self.has_dedup = True
        self.generic_visit(node)

    def visit_AugAssign(self, node: ast.AugAssign) -> None:
        if isinstance(node.op, (ast.Add, ast.Sub, ast.Mult, ast.Div)):
            self.has_aggregation = True
        self.generic_visit(node)


def _analyze_program(code: Any, tool_names: set[str]) -> dict[str, Any]:
    if not isinstance(code, str) or not code.strip():
        return {"syntax_valid": False, "syntax_error": "empty code"}
    try:
        tree = ast.parse(code)
    except SyntaxError as exc:
        return {
            "syntax_valid": False,
            "syntax_error": exc.msg,
            "syntax_error_line": exc.lineno,
        }
    visitor = _ProgramVisitor(tool_names)
    visitor.visit(tree)
    return {
        "syntax_valid": True,
        "has_loop": any(isinstance(node, (ast.For, ast.While)) for node in ast.walk(tree)),
        "tool_calls": visitor.tool_calls,
        "tool_calls_in_loops": visitor.tool_calls_in_loops,
        "conditional_tool_calls": visitor.conditional_tool_calls,
        "has_dedup": visitor.has_dedup,
        "has_filter": visitor.has_filter,
        "has_aggregation": visitor.has_aggregation,
        "print_calls": visitor.print_calls,
    }
