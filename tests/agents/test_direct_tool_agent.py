from __future__ import annotations

from typing import Any

from graphptc.config import RuntimeConfig
from graphptc.direct_tool_agent import DirectToolAgent
from graphptc.model import ModelTurn, TokenUsage, ToolCall
from graphptc.ptc import FINALIZE_PROMPT


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
        search_tools=tools,
        runtime=RuntimeConfig(max_turns=3),
        system_prompt="system",
        user_prompt_template="<question>{question}</question>",
        functions={"search": tools.search, "fetch": tools.fetch},
        tool_specs=specs,
    )

    result = agent.run("task")

    assert result.status == "success"
    assert result.answer == "<result>answer</result>"
    assert result.ptc_blocks == 0
    assert [call["operation"] for call in result.search_calls] == ["search", "fetch"]
    assert model.calls[0]["tools"] == specs
    assert model.calls[1]["messages"][-3]["content"].startswith("[")
    assert model.calls[1]["messages"][-2]["content"].startswith("{")
    assert result.runtime_session["mode"] == "direct_tool_calling"
    assert result.runtime_session["persistent"] is False
    assert result.runtime_session["tool_rounds"] == 1
    assert result.runtime_session["direct_tool_calls"] == 2
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
        search_tools=tools,
        runtime=RuntimeConfig(max_turns=3),
        system_prompt="system",
        user_prompt_template="<question>{question}</question>",
        functions={"search": tools.search, "fetch": tools.fetch},
        tool_specs=[{"type": "function", "function": {"name": "search"}}],
    )

    result = agent.run("task")

    assert result.status == "success"
    assert model.calls[1]["tools"] == []
    assert model.calls[1]["messages"][-2]["content"] == FINALIZE_PROMPT
