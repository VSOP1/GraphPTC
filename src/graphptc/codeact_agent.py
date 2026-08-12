from __future__ import annotations

import json
from collections.abc import Callable, Iterable
from typing import Any, Mapping

from toolregistry import ToolRegistry

from .config import RuntimeConfig
from .observability import ExecutionObserver
from .persistent_runtime import PersistentIpcRuntime
from .ptc import PTC_TOOL_SPEC, OriginalPTCAgent, PTCBlockTrace, _truncate
from .search import TavilySearchTools


CODEACT_SYSTEM_PROMPT = """You are a research agent. Your only directly callable tool is
programmatic_tool_call, which runs Python in a task-scoped session. Variables and imports persist
between blocks for this task and are reset before the next task. The Python environment provides:

- search(*, query: str) -> list[dict] with docid, url, score, and snippet
- fetch(*, docid: str) -> dict with docid, url, and content

Use Python as the action language: call tools, loop over candidates, deduplicate results, filter
irrelevant evidence, and print only the compact observations needed for the next decision. A block
may make any number of useful calls. Continue in the same block while the next actions follow from
program state; return to the model when the observations require a new semantic judgment or you can
answer. Always use keyword arguments for tool calls. You may import safe computation modules, but
must not access files, the shell, environment variables, or the network outside the provided tools.
You decide whether tools are needed and whether to emit zero, one, or multiple blocks. Answer
concisely inside <result> and </result> tags; separate multiple answers with commas."""


CODEACT_STATELESS_SYSTEM_PROMPT = CODEACT_SYSTEM_PROMPT.replace(
    "Variables and imports persist\nbetween blocks for this task and are reset before the next task.",
    "Each block starts a fresh Python process, so define every variable and import it uses.",
)


CODEACT_FEW_SHOT_SUFFIX = r"""

Illustrative action examples (the names and facts are unrelated to the current task):

Example 1, mechanically inspect several hypotheses in one block:
```python
import json
queries = ["Redwood permit", "Juniper permit", "Willow permit"]
seen = {}
for query in queries:
    for hit in search(query=query):
        seen.setdefault(hit["docid"], hit)
shortlist = [hit for hit in seen.values() if "permit" in hit["snippet"].lower()]
evidence = []
for hit in shortlist[:8]:
    page = fetch(docid=hit["docid"])
    lines = [line for line in page["content"].splitlines() if "permit" in line.lower()]
    if lines:
        evidence.append({"docid": hit["docid"], "lines": lines[:3]})
print(json.dumps(evidence, ensure_ascii=False))
```

Example 2, reuse task state only after an observation changes the semantic direction:
```python
# Earlier blocks stored candidate_hits and printed their compact labels.
selected_terms = ["oscillator", "replacement"]
followup = []
for hit in candidate_hits:
    if any(term in hit["snippet"].lower() for term in selected_terms):
        followup.append(fetch(docid=hit["docid"]))
print([{"docid": page["docid"], "excerpt": page["content"][:500]} for page in followup])
```
These examples demonstrate Python control flow, not a required number of calls or a fixed template."""


CODEACT_USER_PROMPT_TEMPLATE = """Answer the following question using the research environment.

<question>{question}</question>

Return the final concise answer in <result> tags."""


class CodeActPTCAgent(OriginalPTCAgent):
    """Training-free CodeAct-style PTC variant with task-scoped Python state."""

    def __init__(
        self,
        model: Any,
        search_tools: TavilySearchTools,
        runtime: RuntimeConfig,
        *,
        system_prompt: str = CODEACT_SYSTEM_PROMPT,
        user_prompt_template: str = CODEACT_USER_PROMPT_TEMPLATE,
        runtime_functions: Iterable[Callable[..., Any]] | None = None,
        checkpoint_callback: Callable[[dict[str, Any]], None] | None = None,
        persistent: bool = True,
        structured_observation: bool = False,
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
        observer: ExecutionObserver | None = None,
        active_repair_callback: (
            Callable[[str, PersistentIpcRuntime], dict[str, Any]] | None
        ) = None,
    ) -> None:
        if active_repair_callback is not None and observer is None:
            raise ValueError("active repair requires an execution observer")
        self._observer = observer
        self._active_repair_callback = active_repair_callback
        self._active_repair_attempted = False
        self._persistent_runtime = (
            PersistentIpcRuntime(observer=observer) if persistent else None
        )
        self._structured_observation = structured_observation
        self._seen_docids: set[str] = set()
        self._observed_block_count = 0
        super().__init__(
            model=model,
            search_tools=search_tools,
            runtime=runtime,
            system_prompt=system_prompt,
            user_prompt_template=user_prompt_template,
            runtime_functions=runtime_functions,
            checkpoint_callback=checkpoint_callback,
            ptc_tool_spec=ptc_tool_spec,
            demonstration_messages=demonstration_messages,
            post_block_message_factory=post_block_message_factory,
            post_block_message_on_error=post_block_message_on_error,
            block_observation_factory=block_observation_factory,
            ptc_call_metadata_callback=ptc_call_metadata_callback,
            adaptation_initial_observation=adaptation_initial_observation,
            message_projection_callback=message_projection_callback,
        )

    def _create_registry(
        self, functions: Iterable[Callable[..., Any]]
    ) -> ToolRegistry:
        registry = ToolRegistry()
        for function in functions:
            registry.register(function)
        registry.ptc.enable(
            timeout=self._runtime.code_timeout_seconds,
            runtime=self._persistent_runtime,
        )
        return registry

    def run(self, task: str):  # type: ignore[no-untyped-def]
        result = None
        self._observed_block_count = 0
        self._active_repair_attempted = False
        if self._observer is not None:
            self._observer.emit("episode.started", data={"task": task})
        try:
            result = super().run(task)
        finally:
            if self._persistent_runtime is not None:
                self._persistent_runtime.close()
        if result is not None and self._persistent_runtime is not None:
            result.runtime_session = self._persistent_runtime.telemetry()
        if self._observer is not None:
            self._observer.emit(
                "episode.finished",
                data={
                    "status": result.status if result is not None else "failed",
                    "answer": result.answer if result is not None else "",
                    "error": result.error if result is not None else "Agent run failed",
                    "ptc_blocks": result.ptc_blocks if result is not None else 0,
                },
            )
        return result

    def _execute_block(
        self, turn_number: int, code: Any
    ) -> tuple[str, bool, PTCBlockTrace]:
        self._observed_block_count += 1
        block_id = (
            f"{self._observer.episode_id}:block:{self._observed_block_count}"
            if self._observer is not None
            else None
        )
        if self._observer is not None:
            self._observer.emit(
                "block.started",
                block_id=block_id,
                data={
                    "turn": turn_number,
                    "tool_call_id": getattr(self, "_active_tool_call_id", None),
                    "code": code if isinstance(code, str) else repr(code),
                },
            )
        if self._persistent_runtime is not None:
            self._persistent_runtime.active_block_id = block_id
        calls_before = len(self._search_tools.calls)
        try:
            output, is_error, trace = super()._execute_block(turn_number, code)
        finally:
            if self._persistent_runtime is not None:
                self._persistent_runtime.active_block_id = None
        if self._observer is not None:
            runtime_trace = (
                self._persistent_runtime.last_execution_trace
                if self._persistent_runtime is not None
                else {}
            )
            self._observer.emit(
                "block.finished",
                block_id=block_id,
                data={**trace.__dict__, "runtime_trace": runtime_trace},
            )
        else:
            runtime_trace = (
                self._persistent_runtime.last_execution_trace
                if self._persistent_runtime is not None
                else {}
            )
        trace = PTCBlockTrace(
            **{
                **trace.__dict__,
                "runtime_trace": runtime_trace,
            }
        )
        if (
            is_error
            and block_id is not None
            and self._persistent_runtime is not None
            and self._active_repair_callback is not None
            and not self._active_repair_attempted
        ):
            self._active_repair_attempted = True
            try:
                repair = self._active_repair_callback(
                    block_id, self._persistent_runtime
                )
            except Exception as exc:
                repair = {
                    "status": "active_repair_error",
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                }
            if self._observer is not None:
                self._observer.emit(
                    "repair.finished",
                    block_id=block_id,
                    data={key: value for key, value in repair.items() if key != "output"},
                )
            if repair.get("status") == "repaired_active":
                repaired_output = str(repair["output"])
                stdout_chars = len(repaired_output)
                stdout_truncated = stdout_chars > self._runtime.max_stdout_chars
                output = _truncate(repaired_output, self._runtime.max_stdout_chars)
                trace = PTCBlockTrace(
                    **{
                        **trace.__dict__,
                        "code": str(repair["patched_code"]),
                        "stdout": output,
                        "stdout_chars": stdout_chars,
                        "stdout_truncated": stdout_truncated,
                        "success": True,
                        "runtime_calls": int(
                            repair.get("replay", {}).get(
                                "executed_tool_call_count", 0
                            )
                        ),
                        "error_type": None,
                        "error_message": None,
                    }
                )
                is_error = False
        if not self._structured_observation or is_error:
            return output, is_error, trace

        calls = self._search_tools.calls[calls_before:]
        returned_docids = {
            str(docid)
            for call in calls
            for docid in call.get("docids", ())
        }
        new_docids = sorted(returned_docids - self._seen_docids)
        repeated_docids = sorted(returned_docids & self._seen_docids)
        self._seen_docids.update(returned_docids)
        observation = "PTC_OBSERVATION " + json.dumps(
            {
                "status": "ok",
                "stdout": output,
                "tool_calls": len(calls),
                "search_calls": sum(
                    call.get("operation") == "search" for call in calls
                ),
                "fetch_calls": sum(
                    call.get("operation") == "fetch" for call in calls
                ),
                "new_docids": new_docids,
                "repeated_docids": repeated_docids,
                "state": (
                    self._persistent_runtime.last_state
                    if self._persistent_runtime is not None
                    else {}
                ),
            },
            ensure_ascii=False,
            separators=(",", ":"),
        )
        observation_chars = len(observation)
        observation_truncated = (
            trace.stdout_truncated
            or observation_chars > self._runtime.max_stdout_chars
        )
        observation = _truncate(observation, self._runtime.max_stdout_chars)
        updated_trace = PTCBlockTrace(
            **{
                **trace.__dict__,
                "stdout": observation,
                "stdout_chars": observation_chars,
                "stdout_truncated": observation_truncated,
            }
        )
        return observation, False, updated_trace
