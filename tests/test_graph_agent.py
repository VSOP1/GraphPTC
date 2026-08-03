from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from graphptc.config import RuntimeConfig
from graphptc.graph_agent import GraphPTCAgent
from graphptc.model import ModelTurn, TokenUsage, ToolCall
from graphptc.observability import JsonlEventSink, ProgramAnalyzer


@dataclass
class FakeTools:
    calls: list[dict[str, Any]]

    def lookup(self, query: str) -> list[dict[str, str]]:
        """Return deterministic evidence."""
        self.calls.append({"query": query})
        return [{"value": query.upper()}]

    def failing_lookup(self, query: str) -> list[dict[str, str]]:
        """Raise a deterministic tool error."""
        raise ValueError(f"bad query: {query}")


class ScriptedModel:
    def __init__(self, turns: list[ModelTurn]) -> None:
        self._turns = iter(turns)

    def create_turn(self, **kwargs: Any) -> ModelTurn:
        return next(self._turns)


def _tool_turn(code: str) -> ModelTurn:
    return ModelTurn(
        assistant_message={
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "call-1",
                    "type": "function",
                    "function": {
                        "name": "programmatic_tool_call",
                        "arguments": json.dumps({"code": code}),
                    },
                }
            ],
        },
        text="",
        tool_calls=[
            ToolCall(
                id="call-1",
                name="programmatic_tool_call",
                input={"code": code},
            )
        ],
        usage=TokenUsage(input_tokens=10, output_tokens=5),
        stop_reason="tool_calls",
    )


def _answer_turn(text: str) -> ModelTurn:
    return ModelTurn(
        assistant_message={"role": "assistant", "content": text},
        text=text,
        tool_calls=[],
        usage=TokenUsage(input_tokens=5, output_tokens=3),
        stop_reason="stop",
    )


def _multi_tool_turn(codes: list[str]) -> ModelTurn:
    tool_calls = [
        ToolCall(
            id=f"call-{index}",
            name="programmatic_tool_call",
            input={"code": code},
        )
        for index, code in enumerate(codes, 1)
    ]
    return ModelTurn(
        assistant_message={
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": call.id,
                    "type": "function",
                    "function": {
                        "name": call.name,
                        "arguments": json.dumps(call.input),
                    },
                }
                for call in tool_calls
            ],
        },
        text="",
        tool_calls=tool_calls,
        usage=TokenUsage(input_tokens=10, output_tokens=10),
        stop_reason="tool_calls",
    )


def test_graph_agent_records_episode_block_and_tool_lifecycle(
    tmp_path: Path,
) -> None:
    tools = FakeTools(calls=[])
    code = "result = lookup(query='alpha')\nprint(result[0]['value'])"
    sink_path = tmp_path / "events.jsonl"
    agent = GraphPTCAgent(
        model=ScriptedModel([_tool_turn(code), _answer_turn("done")]),
        search_tools=tools,  # type: ignore[arg-type]
        runtime=RuntimeConfig(max_turns=3, max_ptc_blocks=2),
        runtime_functions=(tools.lookup,),
        event_sink=JsonlEventSink(sink_path),
    )

    result = agent.run("research alpha")

    assert result.agent.status == "success"
    assert result.agent.blocks[0].stdout.strip() == "ALPHA"
    assert [event.kind for event in result.events] == [
        "episode.started",
        "block.started",
        "block.analyzed",
        "tool.started",
        "tool.finished",
        "block.finished",
        "episode.finished",
    ]
    assert [event.sequence for event in result.events] == list(range(1, 8))
    tool_finished = next(
        event for event in result.events if event.kind == "tool.finished"
    )
    assert tool_finished.status == "success"
    assert tool_finished.payload["source_mapping"] == "unique_static_candidate"
    assert tool_finished.payload["static_callsite_id"].startswith("callsite_")
    assert tool_finished.payload["result"]["sha256"]

    persisted = [
        json.loads(line) for line in sink_path.read_text(encoding="utf-8").splitlines()
    ]
    assert [event["event_id"] for event in persisted] == [
        event.event_id for event in result.events
    ]


def test_graph_agent_observes_multiple_ptc_blocks_from_one_model_turn() -> None:
    tools = FakeTools(calls=[])
    codes = [
        "print(lookup(query='alpha')[0]['value'])",
        "print(lookup(query='beta')[0]['value'])",
    ]
    agent = GraphPTCAgent(
        model=ScriptedModel([_multi_tool_turn(codes), _answer_turn("done")]),
        search_tools=tools,  # type: ignore[arg-type]
        runtime=RuntimeConfig(max_turns=3, max_ptc_blocks=3),
        runtime_functions=(tools.lookup,),
    )

    result = agent.run("research both")

    assert result.agent.status == "success"
    assert result.agent.ptc_blocks == 2
    assert [block.turn for block in result.agent.blocks] == [1, 1]
    assert [event.kind for event in result.events].count("block.started") == 2
    assert [event.kind for event in result.events].count("block.finished") == 2
    assert [event.kind for event in result.events].count("tool.finished") == 2
    block_ids = {
        event.block_id for event in result.events if event.kind == "block.started"
    }
    assert len(block_ids) == 2


def test_graph_agent_records_tool_and_block_failure_then_recovery() -> None:
    tools = FakeTools(calls=[])
    code = "print(failing_lookup(query='broken'))"
    agent = GraphPTCAgent(
        model=ScriptedModel([_tool_turn(code), _answer_turn("recovered")]),
        search_tools=tools,  # type: ignore[arg-type]
        runtime=RuntimeConfig(max_turns=3, max_ptc_blocks=2),
        runtime_functions=(tools.failing_lookup,),
    )

    result = agent.run("recover")

    assert result.agent.status == "success"
    assert result.agent.blocks[0].success is False
    tool_finished = next(
        event for event in result.events if event.kind == "tool.finished"
    )
    block_finished = next(
        event for event in result.events if event.kind == "block.finished"
    )
    assert tool_finished.status == "error"
    assert tool_finished.payload["exception_type"] == "ValueError"
    assert block_finished.status == "failed"
    assert result.events[-1].status == "success"


def test_program_analysis_marks_multiple_same_tool_calls_as_candidates() -> None:
    analysis = ProgramAnalyzer().analyze(
        "a = lookup(query='a')\nb = lookup(query='b')", {"lookup"}
    )

    assert analysis.syntax_error is None
    assert len(analysis.callsites) == 2
    assert [site.span.line for site in analysis.callsites] == [1, 2]
    assert len({site.callsite_id for site in analysis.callsites}) == 2


def test_program_analysis_preserves_syntax_failure_as_observation() -> None:
    analysis = ProgramAnalyzer().analyze("if :", {"lookup"})

    assert analysis.callsites == ()
    assert "invalid syntax" in (analysis.syntax_error or "")
