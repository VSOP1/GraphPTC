from __future__ import annotations

import json
from typing import Any

from graphptc.codeact_agent import CodeActPTCAgent
from graphptc.config import RuntimeConfig
from graphptc.model import ModelTurn, TokenUsage, ToolCall
from graphptc.observability import ExecutionObserver, InMemoryEventSink
from graphptc.persistent_runtime import PersistentIpcRuntime


class ScriptedModel:
    def __init__(self, turns: list[ModelTurn]) -> None:
        self._turns = iter(turns)
        self.messages_seen: list[list[dict[str, Any]]] = []

    def create_turn(self, **kwargs: Any) -> ModelTurn:
        self.messages_seen.append(list(kwargs["messages"]))
        return next(self._turns)


class FakeLocalTools:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def search(self, *, query: str) -> list[dict[str, Any]]:
        docid = f"doc-{query}"
        self.calls.append(
            {"operation": "search", "query": query, "docids": (docid,)}
        )
        return [{"docid": docid, "snippet": query}]

    def fetch(self, *, docid: str) -> dict[str, Any]:
        self.calls.append(
            {"operation": "fetch", "docid": docid, "docids": (docid,)}
        )
        return {"docid": docid, "content": f"content for {docid}"}


def tool_turn(call_id: str, code: str) -> ModelTurn:
    return ModelTurn(
        assistant_message={"role": "assistant", "content": None},
        text="",
        tool_calls=[
            ToolCall(
                id=call_id,
                name="programmatic_tool_call",
                input={"code": code},
            )
        ],
        usage=TokenUsage(input_tokens=10, output_tokens=5),
        stop_reason="tool_calls",
    )


def answer_turn() -> ModelTurn:
    return ModelTurn(
        assistant_message={"role": "assistant", "content": "<result>done</result>"},
        text="<result>done</result>",
        tool_calls=[],
        usage=TokenUsage(input_tokens=10, output_tokens=5),
        stop_reason="stop",
    )


def test_persistent_runtime_keeps_state_between_blocks() -> None:
    runtime = PersistentIpcRuntime()
    try:
        first = runtime.execute("items = [1, 2]\nprint(len(items))", timeout=5)
        second = runtime.execute("items.append(3)\nprint(sum(items))", timeout=5)
    finally:
        runtime.close()

    assert first.stdout.strip() == "2"
    assert second.stdout.strip() == "6"
    assert runtime.last_state["items"] == "list"


def test_persistent_runtime_forwards_tools() -> None:
    runtime = PersistentIpcRuntime()
    try:
        result = runtime.execute(
            "values = [lookup(query=q) for q in ['a', 'b']]\nprint(values)",
            namespace={"lookup": lambda *, query: query.upper()},
            timeout=5,
        )
    finally:
        runtime.close()

    assert result.return_code == 0
    assert result.stdout.strip() == "['A', 'B']"


def test_observed_runtime_records_exact_call_sites_and_state_access() -> None:
    sink = InMemoryEventSink()
    observer = ExecutionObserver(
        sink,
        episode_id="episode-runtime",
        task_id="task-runtime",
    )
    runtime = PersistentIpcRuntime(observer=observer)
    runtime.active_block_id = "block-1"
    try:
        first = runtime.execute(
            "items = [lookup(query=q) for q in ['a', 'b']]\nprint(items)",
            namespace={"lookup": lambda *, query: query.upper()},
            timeout=5,
        )
        first_trace = runtime.last_execution_trace.copy()
        runtime.active_block_id = "block-2"
        second = runtime.execute("print(items)", timeout=5)
    finally:
        runtime.close()

    assert first.stdout.strip() == "['A', 'B']"
    assert second.stdout.strip() == "['A', 'B']"
    calls = [event for event in sink.events if event["type"] == "tool.called"]
    assert len(calls) == 2
    assert {event["data"]["call_site"]["line"] for event in calls} == {1}
    assert {event["data"]["call_site"]["column"] for event in calls} == {9}
    assert runtime.last_execution_trace["state_before"] == {"items": "list"}
    assert "items" in runtime.last_execution_trace["loaded_names"]
    assert runtime.last_execution_trace["stored_names"] == []
    assert {
        "line": 1,
        "column": 9,
        "end_line": 1,
        "end_column": 24,
    } in first_trace["executed_spans"]


def test_observed_runtime_records_error_location() -> None:
    observer = ExecutionObserver(
        InMemoryEventSink(),
        episode_id="episode-error",
        task_id="task-error",
    )
    runtime = PersistentIpcRuntime(observer=observer)
    try:
        result = runtime.execute("items = [1]\nprint(items[2])", timeout=5)
        trace = runtime.last_execution_trace.copy()
    finally:
        runtime.close()

    assert result.return_code == 1
    assert trace["error_location"] == {
        "line": 2,
        "column": 6,
        "end_line": 2,
        "end_column": 14,
    }


def test_persistent_runtime_preserves_unicode_without_desynchronizing() -> None:
    runtime = PersistentIpcRuntime()
    try:
        first = runtime.execute(
            "print(lookup(query='x'))",
            namespace={"lookup": lambda *, query: "minus \N{MINUS SIGN}"},
            timeout=5,
        )
        second = runtime.execute("print('still synchronized')", timeout=5)
    finally:
        runtime.close()

    assert first.stdout.strip() == "minus \N{MINUS SIGN}"
    assert second.stdout.strip() == "still synchronized"


def test_codeact_agent_reuses_state_and_returns_structured_observation() -> None:
    tools = FakeLocalTools()
    model = ScriptedModel(
        [
            tool_turn(
                "call-1",
                "hits = search(query='alpha')\nprint(hits[0]['docid'])",
            ),
            tool_turn(
                "call-2",
                "page = fetch(docid=hits[0]['docid'])\nprint(page['content'])",
            ),
            answer_turn(),
        ]
    )
    agent = CodeActPTCAgent(
        model=model,
        search_tools=tools,  # type: ignore[arg-type]
        runtime=RuntimeConfig(max_turns=4, max_ptc_blocks=3),
        runtime_functions=(tools.search, tools.fetch),
        structured_observation=True,
    )

    result = agent.run("test")

    assert result.status == "success"
    assert [block.success for block in result.blocks] == [True, True]
    first = json.loads(result.blocks[0].stdout.removeprefix("PTC_OBSERVATION "))
    second = json.loads(result.blocks[1].stdout.removeprefix("PTC_OBSERVATION "))
    assert first["new_docids"] == ["doc-alpha"]
    assert first["state"]["hits"] == "list"
    assert second["repeated_docids"] == ["doc-alpha"]
    assert "content for doc-alpha" in second["stdout"]
    assert model.messages_seen[1][-1]["content"].startswith("PTC_OBSERVATION ")
    assert result.runtime_session == {
        "persistent": True,
        "process_starts": 1,
        "executions": 2,
        "timeouts": 0,
        "protocol_errors": 0,
        "tool_calls": 2,
        "closed": True,
        "state": {"hits": "list", "page": "dict"},
    }


def test_codeact_agent_raw_stdout_preserves_persistent_state_and_truncation() -> None:
    tools = FakeLocalTools()
    model = ScriptedModel(
        [
            tool_turn("call-1", "items = ['alpha']\nprint('x' * 500)"),
            tool_turn("call-2", "items.append('beta')\nprint(','.join(items))"),
            answer_turn(),
        ]
    )
    agent = CodeActPTCAgent(
        model=model,
        search_tools=tools,  # type: ignore[arg-type]
        runtime=RuntimeConfig(max_turns=4, max_ptc_blocks=3, max_stdout_chars=240),
        runtime_functions=(tools.search, tools.fetch),
        structured_observation=False,
    )

    result = agent.run("test")

    first, second = result.blocks
    assert first.stdout_truncated is True
    assert first.stdout_chars == 501
    assert len(first.stdout) == 240
    assert first.stdout.startswith("PTC_STDOUT_TRUNCATED ")
    assert second.stdout == "alpha,beta\n"
    assert second.stdout_truncated is False
    assert model.messages_seen[1][-1]["content"] == first.stdout
    assert model.messages_seen[2][-1]["content"] == second.stdout
    assert "PTC_OBSERVATION" not in first.stdout + second.stdout


def test_active_repair_replaces_error_observation_and_runtime_state() -> None:
    tools = FakeLocalTools()
    model = ScriptedModel(
        [
            tool_turn("call-1", "items = [1]\nprint(missing)"),
            tool_turn("call-2", "items.append(2)\nprint(sum(items))"),
            answer_turn(),
        ]
    )
    sink = InMemoryEventSink()
    observer = ExecutionObserver(sink, episode_id="active", task_id="active")
    calls = 0

    def repair(block_id: str, runtime: PersistentIpcRuntime) -> dict[str, Any]:
        nonlocal calls
        calls += 1
        assert block_id == "active:block:1"
        runtime.close()
        execution = runtime.execute("items = [1]\nprint('repaired')", timeout=5)
        return {
            "status": "repaired_active",
            "patched_code": "items = [1]\nprint('repaired')",
            "output": execution.stdout,
            "replay": {"executed_tool_call_count": 0},
        }

    agent = CodeActPTCAgent(
        model=model,
        search_tools=tools,  # type: ignore[arg-type]
        runtime=RuntimeConfig(max_turns=4, max_ptc_blocks=3),
        runtime_functions=(tools.search, tools.fetch),
        observer=observer,
        active_repair_callback=repair,
    )

    result = agent.run("test")

    assert result.status == "success"
    assert calls == 1
    assert model.messages_seen[1][-1]["content"] == "repaired\n"
    assert result.blocks[0].success is True
    assert result.blocks[1].stdout == "3\n"
    repairs = [event for event in sink.events if event["type"] == "repair.finished"]
    assert len(repairs) == 1
    assert repairs[0]["data"]["status"] == "repaired_active"


def test_active_repair_preserves_stdout_truncation_contract() -> None:
    tools = FakeLocalTools()
    model = ScriptedModel(
        [
            tool_turn("call-1", "print(missing)"),
            answer_turn(),
        ]
    )
    observer = ExecutionObserver(
        InMemoryEventSink(), episode_id="active-cap", task_id="active-cap"
    )

    def repair(block_id: str, runtime: PersistentIpcRuntime) -> dict[str, Any]:
        return {
            "status": "repaired_active",
            "patched_code": "print('x' * 500)",
            "output": "x" * 501,
            "replay": {"executed_tool_call_count": 0},
        }

    agent = CodeActPTCAgent(
        model=model,
        search_tools=tools,  # type: ignore[arg-type]
        runtime=RuntimeConfig(max_turns=3, max_ptc_blocks=2, max_stdout_chars=240),
        runtime_functions=(tools.search, tools.fetch),
        observer=observer,
        active_repair_callback=repair,
    )

    result = agent.run("test")

    assert len(model.messages_seen[1][-1]["content"]) == 240
    assert model.messages_seen[1][-1]["content"].startswith(
        "PTC_STDOUT_TRUNCATED "
    )
    assert result.blocks[0].stdout_chars == 501
    assert result.blocks[0].stdout_truncated is True


def test_separate_persistent_runtimes_do_not_share_state() -> None:
    first = PersistentIpcRuntime()
    second = PersistentIpcRuntime()
    try:
        first.execute("secret = 42", timeout=5)
        result = second.execute("print('secret' in globals())", timeout=5)
    finally:
        first.close()
        second.close()

    assert result.stdout.strip() == "False"


def test_timeout_kills_session_and_next_block_starts_clean() -> None:
    runtime = PersistentIpcRuntime()
    try:
        timed_out = runtime.execute("value = 7\nwhile True: pass", timeout=0.1)
        restarted = runtime.execute("print('value' in globals())", timeout=5)
    finally:
        runtime.close()

    assert timed_out.timed_out is True
    assert restarted.stdout.strip() == "False"
