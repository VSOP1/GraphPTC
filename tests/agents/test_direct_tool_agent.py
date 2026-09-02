from __future__ import annotations

from typing import Any

import pytest

from graphptc.agents.direct_tools import DIRECT_FINALIZE_PROMPT, DirectToolAgent
from graphptc.config import RuntimeConfig
from graphptc.model import ModelTurn, TokenUsage, ToolCall


class ScriptedModel:
    def __init__(self, turns: list[ModelTurn]) -> None:
        self.turns = iter(turns)
        self.calls: list[dict[str, Any]] = []

    def create_turn(self, **kwargs: Any) -> ModelTurn:
        self.calls.append(kwargs)
        return next(self.turns)


class FakeTools:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def search(self, *, query: str) -> list[dict[str, Any]]:
        self.calls.append({"operation": "search", "query": query, "docids": ["d1"]})
        return [{"docid": "d1", "score": 1.0, "snippet": "hit"}]

    def fetch(self, *, docid: str) -> dict[str, Any]:
        self.calls.append({"operation": "fetch", "docid": docid, "docids": [docid]})
        return {"docid": docid, "content": "evidence"}


def _turn(
    *, calls: list[ToolCall] | None = None, text: str = "", reason: str = "tool_calls"
) -> ModelTurn:
    calls = calls or []
    assistant: dict[str, Any] = {"role": "assistant", "content": text or None}
    if calls:
        assistant["tool_calls"] = [
            {
                "id": call.id,
                "type": "function",
                "function": {"name": call.name, "arguments": "{}"},
            }
            for call in calls
        ]
    return ModelTurn(
        assistant_message=assistant,
        text=text,
        tool_calls=calls,
        usage=TokenUsage(input_tokens=10, output_tokens=2, cached_input_tokens=4),
        stop_reason=reason,
    )


def test_direct_agent_exposes_and_executes_search_fetch_without_ptc() -> None:
    tools = FakeTools()
    specs = [
        {"type": "function", "function": {"name": "search"}},
        {"type": "function", "function": {"name": "fetch"}},
    ]
    model = ScriptedModel(
        [
            _turn(
                calls=[
                    ToolCall("s1", "search", {"query": "alpha"}),
                    ToolCall("f1", "fetch", {"docid": "d1"}),
                ]
            ),
            _turn(text="<result>answer</result>", reason="stop"),
        ]
    )
    agent = DirectToolAgent(
        model=model,
        runtime=RuntimeConfig(max_turns=3),
        system_prompt="system",
        user_prompt_template="<question>{task}</question>",
        functions={"search": tools.search, "fetch": tools.fetch},
        tool_specs=specs,
    )

    result = agent.run("task")

    assert result.status == "success"
    assert result.answer == "<result>answer</result>"
    assert result.ptc_blocks == 0
    assert result.search_calls == []
    assert model.calls[0]["tools"] == specs
    assert model.calls[1]["messages"][-3]["content"].startswith("[")
    assert model.calls[1]["messages"][-2]["content"].startswith("{")
    assert result.runtime_session["mode"] == "direct_tool_calling"
    assert result.runtime_session["persistent"] is False
    assert result.runtime_session["tool_rounds"] == 1
    assert result.runtime_session["direct_tool_calls"] == 2
    assert result.runtime_session["tool_errors"] == 0
    assert result.runtime_session["tool_call_distribution"] == {"search": 1, "fetch": 1}
    assert [item["tool"] for item in result.runtime_session["tool_call_trace"]] == [
        "search",
        "fetch",
    ]
    assert result.runtime_session["tool_observation_chars"] > 0


def test_direct_agent_finalizes_after_truncated_completion() -> None:
    tools = FakeTools()
    model = ScriptedModel(
        [
            _turn(text="partial", reason="length"),
            _turn(text="<result>answer</result>", reason="stop"),
        ]
    )
    agent = DirectToolAgent(
        model=model,
        runtime=RuntimeConfig(max_turns=3),
        system_prompt="system",
        user_prompt_template="<question>{task}</question>",
        functions={"search": tools.search},
        tool_specs=[{"type": "function", "function": {"name": "search"}}],
    )

    result = agent.run("task")

    assert result.status == "success"
    assert model.calls[1]["tools"] == []
    assert model.calls[1]["messages"][-2]["content"] == DIRECT_FINALIZE_PROMPT


def test_direct_agent_supports_non_retrieval_tools_and_task_template() -> None:
    state = {"value": 0}

    def advance(*, amount: int) -> str:
        state["value"] += amount
        return f"state={state['value']}"

    specs = [{"type": "function", "function": {"name": "advance"}}]
    model = ScriptedModel(
        [
            _turn(calls=[ToolCall("a1", "advance", {"amount": 2})]),
            _turn(text="done", reason="stop"),
        ]
    )
    agent = DirectToolAgent(
        model=model,
        runtime=RuntimeConfig(max_turns=3),
        system_prompt="system",
        user_prompt_template="Task: {task}",
        functions={"advance": advance},
        tool_specs=specs,
        finalize_prompt="Finish this benchmark task.",
    )

    result = agent.run("update state")

    assert result.status == "success"
    assert state == {"value": 2}
    assert model.calls[0]["messages"][0]["content"] == "Task: update state"
    assert model.calls[1]["messages"][-2]["content"] == "state=2"
    assert result.runtime_session["tool_call_distribution"] == {"advance": 1}


def test_direct_agent_bounds_large_tool_observations() -> None:
    def large_result() -> str:
        return "x" * 1_000

    model = ScriptedModel(
        [
            _turn(calls=[ToolCall("large", "large_result", {})]),
            _turn(text="done", reason="stop"),
        ]
    )
    agent = DirectToolAgent(
        model=model,
        runtime=RuntimeConfig(max_turns=3, max_stdout_chars=128),
        system_prompt="system",
        user_prompt_template="{task}",
        functions={"large_result": large_result},
        tool_specs=[{"type": "function", "function": {"name": "large_result"}}],
    )

    result = agent.run("task")

    observation = model.calls[1]["messages"][-2]["content"]
    assert len(observation) == 128
    assert observation.startswith("DIRECT_TOOL_OUTPUT_TRUNCATED ")
    assert result.runtime_session["tool_observation_truncations"] == 1
    assert result.runtime_session["tool_call_trace"][0]["observation_truncated"] is True


def test_direct_agent_rejects_function_spec_mismatches() -> None:
    with pytest.raises(ValueError, match="function/spec mismatch"):
        DirectToolAgent(
            model=ScriptedModel([]),
            runtime=RuntimeConfig(),
            system_prompt="system",
            user_prompt_template="{task}",
            functions={"actual": lambda: None},
            tool_specs=[{"type": "function", "function": {"name": "declared"}}],
        )
