from __future__ import annotations

import copy
from typing import Any

from graphptc.agents.codeact import CodeActPTCAgent
from graphptc.config import RuntimeConfig
from graphptc.benchmarks.browsecomp_plus.ptc_fewshot import PTC_FEW_SHOT_MESSAGES
from graphptc.model import ModelTurn, TokenUsage, ToolCall
from graphptc.runtime.observability import ExecutionObserver, InMemoryEventSink


class RecordingModel:
    def __init__(self, turns: list[ModelTurn]) -> None:
        self._turns = iter(turns)
        self.requests: list[dict[str, Any]] = []

    def create_turn(self, **kwargs: Any) -> ModelTurn:
        self.requests.append(copy.deepcopy(kwargs))
        return next(self._turns)


class FakeSearchTools:
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


def _tool_turn(call_id: str, code: str) -> ModelTurn:
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


def _answer_turn(answer: str = "done") -> ModelTurn:
    text = f"<result>{answer}</result>"
    return ModelTurn(
        assistant_message={"role": "assistant", "content": text},
        text=text,
        tool_calls=[],
        usage=TokenUsage(input_tokens=10, output_tokens=5),
        stop_reason="stop",
    )


def _agent(
    model: RecordingModel,
    tools: FakeSearchTools,
    *,
    observer: ExecutionObserver | None = None,
    max_stdout_chars: int = 4_000,
    demonstration_messages: tuple[dict[str, Any], ...] = (),
) -> CodeActPTCAgent:
    return CodeActPTCAgent(
        model=model,
        search_tools=tools,  # type: ignore[arg-type]
        runtime=RuntimeConfig(
            max_turns=5,
            max_ptc_blocks=4,
            max_stdout_chars=max_stdout_chars,
        ),
        runtime_functions=(tools.search, tools.fetch),
        observer=observer,
        demonstration_messages=demonstration_messages,
    )


def test_observer_is_behaviorally_transparent_for_matched_runs(
    monkeypatch: Any,
) -> None:
    monkeypatch.setattr("graphptc.agents.original_ptc.time.perf_counter", lambda: 100.0)
    turns = [
        _tool_turn(
            "call-1",
            "hits = search(query='alpha')\n"
            "page = fetch(docid=hits[0]['docid'])\n"
            "print(page['content'])",
        ),
        _tool_turn("call-2", "print(hits[0]['docid'])"),
        _answer_turn("alpha"),
    ]
    baseline_model = RecordingModel(copy.deepcopy(turns))
    observed_model = RecordingModel(copy.deepcopy(turns))
    baseline_tools = FakeSearchTools()
    observed_tools = FakeSearchTools()
    sink = InMemoryEventSink()
    observer = ExecutionObserver(
        sink,
        episode_id="episode-1",
        task_id="task-1",
    )

    baseline = _agent(
        baseline_model,
        baseline_tools,
        demonstration_messages=PTC_FEW_SHOT_MESSAGES,
    ).run("research alpha")
    observed = _agent(
        observed_model,
        observed_tools,
        observer=observer,
        demonstration_messages=PTC_FEW_SHOT_MESSAGES,
    ).run("research alpha")

    assert observed_model.requests == baseline_model.requests
    assert [block.code for block in observed.blocks] == [
        block.code for block in baseline.blocks
    ]
    assert [block.stdout for block in observed.blocks] == [
        block.stdout for block in baseline.blocks
    ]
    assert observed.answer == baseline.answer
    assert observed_tools.calls == baseline_tools.calls
    assert all("GRAPHPTC" not in str(request["messages"]) for request in observed_model.requests)

    events = sink.events
    assert [event["type"] for event in events] == [
        "episode.started",
        "block.started",
        "tool.called",
        "tool.called",
        "block.finished",
        "block.started",
        "block.finished",
        "episode.finished",
    ]
    assert [event["sequence"] for event in events] == list(range(1, 9))
    assert {event["episode_id"] for event in events} == {"episode-1"}
    tool_events = [event for event in events if event["type"] == "tool.called"]
    assert [event["data"]["tool"] for event in tool_events] == ["search", "fetch"]
    assert len({event["block_id"] for event in tool_events}) == 1


def test_observer_records_runtime_error_and_stdout_truncation() -> None:
    model = RecordingModel(
        [
            _tool_turn("call-1", "print('x' * 500)"),
            _tool_turn("call-2", "raise ValueError('bad code')"),
            _answer_turn(),
        ]
    )
    tools = FakeSearchTools()
    sink = InMemoryEventSink()
    result = _agent(
        model,
        tools,
        observer=ExecutionObserver(
            sink,
            episode_id="episode-errors",
            task_id="task-errors",
        ),
        max_stdout_chars=240,
    ).run("exercise failures")

    finished = [event for event in sink.events if event["type"] == "block.finished"]
    assert [event["data"]["success"] for event in finished] == [True, False]
    assert finished[0]["data"]["stdout_truncated"] is True
    assert finished[0]["data"]["stdout"] == result.blocks[0].stdout
    assert finished[1]["data"]["error_type"] == "ValueError"
    assert finished[1]["data"]["stdout"] == result.blocks[1].stdout


def test_observed_agent_resets_runtime_state_between_tasks() -> None:
    first_model = RecordingModel(
        [
            _tool_turn("call-1", "secret = 42\nprint(secret)"),
            _answer_turn("first"),
        ]
    )
    second_model = RecordingModel(
        [
            _tool_turn("call-2", "print('secret' in globals())"),
            _answer_turn("second"),
        ]
    )
    tools = FakeSearchTools()
    first_sink = InMemoryEventSink()
    first_agent = _agent(
        first_model,
        tools,
        observer=ExecutionObserver(
            first_sink,
            episode_id="episode-first",
            task_id="task-first",
        ),
    )

    first = first_agent.run("first task")
    second_sink = InMemoryEventSink()
    second_agent = _agent(
        second_model,
        tools,
        observer=ExecutionObserver(
            second_sink,
            episode_id="episode-second",
            task_id="task-second",
        ),
    )
    second = second_agent.run("second task")

    assert first.blocks[0].stdout == "42\n"
    assert second.blocks[0].stdout == "False\n"
    assert first_sink.events[-1]["type"] == "episode.finished"
    assert second_sink.events[0]["sequence"] == 1
    assert second_sink.events[0]["episode_id"] == "episode-second"
