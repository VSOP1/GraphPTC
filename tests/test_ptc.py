from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from graphptc.config import RuntimeConfig
from graphptc.model import ModelTurn, TokenUsage
from graphptc.ptc import OriginalPTCAgent, _analyze_program, extract_result_tag


@dataclass
class FakeSearchTools:
    calls: list[dict[str, Any]]

    def search_web(self, query: str, max_results: int = 10) -> list[dict[str, Any]]:
        """Return deterministic search results."""
        self.calls.append({"operation": "search", "query": query})
        return [{"title": query, "url": f"https://example.com/{query}", "content": "ok"}]

    def search_web_batch(
        self, queries: list[str], max_results: int = 10
    ) -> dict[str, list[dict[str, Any]]]:
        """Return deterministic batched results."""
        return {query: self.search_web(query, max_results) for query in queries}

    def fetch_url(
        self, url: str, query: str = "", max_chars: int = 20_000
    ) -> dict[str, Any]:
        """Return deterministic page content."""
        self.calls.append({"operation": "fetch", "url": url})
        return {"url": url, "content": "page"}

    def fetch_urls(
        self, urls: list[str], query: str = "", max_chars: int = 20_000
    ) -> list[dict[str, Any]]:
        """Return deterministic page content for multiple URLs."""
        return [self.fetch_url(url, query, max_chars) for url in urls]


class ScriptedModel:
    def __init__(self, turns: list[ModelTurn]) -> None:
        self._turns = iter(turns)
        self.messages_seen: list[list[dict[str, Any]]] = []
        self.requests_seen: list[dict[str, Any]] = []

    def create_turn(self, **kwargs: Any) -> ModelTurn:
        self.requests_seen.append(kwargs)
        self.messages_seen.append(list(kwargs["messages"]))
        return next(self._turns)


def _tool_turn(call_id: str, code: str) -> ModelTurn:
    assistant_message = {
        "role": "assistant",
        "content": None,
        "reasoning_content": "I need another research program.",
        "tool_calls": [
            {
                "id": call_id,
                "type": "function",
                "function": {
                    "name": "programmatic_tool_call",
                    "arguments": "{...}",
                },
            }
        ],
    }
    from graphptc.model import ToolCall

    return ModelTurn(
        assistant_message=assistant_message,
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


def _multi_tool_turn(calls: list[tuple[str, str]]) -> ModelTurn:
    from graphptc.model import ToolCall

    return ModelTurn(
        assistant_message={
            "role": "assistant",
            "content": None,
            "reasoning_content": "Run both independent programs.",
            "tool_calls": [
                {
                    "id": call_id,
                    "type": "function",
                    "function": {
                        "name": "programmatic_tool_call",
                        "arguments": "{...}",
                    },
                }
                for call_id, _ in calls
            ],
        },
        text="",
        tool_calls=[
            ToolCall(
                id=call_id,
                name="programmatic_tool_call",
                input={"code": code},
            )
            for call_id, code in calls
        ],
        usage=TokenUsage(input_tokens=10, output_tokens=5),
        stop_reason="tool_calls",
    )


def _answer_turn(answer: str, stop_reason: str = "stop") -> ModelTurn:
    return ModelTurn(
        assistant_message={"role": "assistant", "content": answer},
        text=answer,
        tool_calls=[],
        usage=TokenUsage(input_tokens=20, output_tokens=7),
        stop_reason=stop_reason,
    )


def _answer_turn_with_usage(
    answer: str,
    *,
    stop_reason: str = "stop",
    input_tokens: int,
) -> ModelTurn:
    return ModelTurn(
        assistant_message={"role": "assistant", "content": answer},
        text=answer,
        tool_calls=[],
        usage=TokenUsage(input_tokens=input_tokens, output_tokens=7),
        stop_reason=stop_reason,
    )


def test_agent_autonomously_runs_multiple_ptc_blocks() -> None:
    tools = FakeSearchTools(calls=[])
    model = ScriptedModel(
        [
            _tool_turn("tool-1", "a = search_web(query='alpha')\nprint(a[0]['title'])"),
            _tool_turn("tool-2", "b = search_web(query='beta')\nprint(b[0]['title'])"),
            _answer_turn("alpha and beta"),
        ]
    )
    agent = OriginalPTCAgent(
        model=model,
        search_tools=tools,  # type: ignore[arg-type]
        runtime=RuntimeConfig(max_turns=4, max_ptc_blocks=3),
    )

    result = agent.run("research both")

    assert result.status == "success"
    assert result.answer == "alpha and beta"
    assert result.ptc_blocks == 2
    assert [block.stdout.strip() for block in result.blocks] == ["alpha", "beta"]
    assert result.model_requests == 3
    assert result.usage.input_tokens == 40
    assert [request.tool_calls for request in result.requests] == [1, 1, 0]
    assert all(request.context_chars > 0 for request in result.requests)
    assert [block.runtime_calls for block in result.blocks] == [1, 1]
    assert result.blocks[0].tool_call_id == "tool-1"
    assert result.blocks[0].stdout_chars == len(result.blocks[0].stdout)
    assert result.blocks[0].stdout_truncated is False
    assert model.messages_seen[1][-1]["content"].strip() == "alpha"
    assert model.messages_seen[1][1]["reasoning_content"] == (
        "I need another research program."
    )
    assert "force_tool" not in model.requests_seen[0]
    exposed_tools = model.requests_seen[0]["tools"]
    assert len(exposed_tools) == 1
    assert exposed_tools[0]["function"]["name"] == "programmatic_tool_call"


def test_agent_executes_multiple_ptc_blocks_from_one_api_turn() -> None:
    tools = FakeSearchTools(calls=[])
    model = ScriptedModel(
        [
            _multi_tool_turn(
                [
                    ("tool-1", "print(search_web(query='alpha')[0]['title'])"),
                    ("tool-2", "print(search_web(query='beta')[0]['title'])"),
                ]
            ),
            _answer_turn("<result>alpha and beta</result>"),
        ]
    )
    agent = OriginalPTCAgent(
        model=model,
        search_tools=tools,  # type: ignore[arg-type]
        runtime=RuntimeConfig(max_turns=3, max_ptc_blocks=3),
    )

    result = agent.run("research both")

    assert result.status == "success"
    assert result.ptc_blocks == 2
    assert result.model_requests == 2
    assert [block.stdout.strip() for block in result.blocks] == ["alpha", "beta"]
    tool_messages = [
        message for message in model.messages_seen[1] if message["role"] == "tool"
    ]
    assert [message["tool_call_id"] for message in tool_messages] == [
        "tool-1",
        "tool-2",
    ]
    assert [message["content"].strip() for message in tool_messages] == [
        "alpha",
        "beta",
    ]


def test_block_trace_reports_programmatic_python_semantics() -> None:
    tools = FakeSearchTools(calls=[])
    code = """queries = ['alpha', 'beta']
seen = {}
for query in queries:
    for hit in search_web(query=query):
        if hit['url'] not in seen:
            seen[hit['url']] = hit
print(sorted(seen))
"""
    model = ScriptedModel([_tool_turn("tool-1", code), _answer_turn("done")])
    agent = OriginalPTCAgent(
        model=model,
        search_tools=tools,  # type: ignore[arg-type]
        runtime=RuntimeConfig(max_turns=3, max_ptc_blocks=2),
    )

    result = agent.run("research")

    analysis = result.blocks[0].program_analysis
    assert result.blocks[0].runtime_calls == 2
    assert analysis["tool_calls_in_loops"] == 1
    assert analysis["has_dedup"] is True
    assert analysis["has_filter"] is True
    assert analysis["has_aggregation"] is True


def test_program_analysis_detects_count_and_accumulator_aggregation() -> None:
    analysis = _analyze_program(
        "count = len(rows)\ntotal = 0\nfor row in rows:\n    total += row['value']\nprint(count, total)",
        {"search", "fetch"},
    )

    assert analysis["has_aggregation"] is True


def test_block_error_feedback_is_structured() -> None:
    tools = FakeSearchTools(calls=[])
    model = ScriptedModel(
        [_tool_turn("tool-1", "print(undefined_name)"), _answer_turn("done")]
    )
    agent = OriginalPTCAgent(
        model=model,
        search_tools=tools,  # type: ignore[arg-type]
        runtime=RuntimeConfig(max_turns=3, max_ptc_blocks=2),
    )

    result = agent.run("research")

    assert result.blocks[0].success is False
    assert result.blocks[0].error_type == "NameError"
    assert result.blocks[0].stdout.startswith("PTC_ERROR ")
    assert "undefined_name" in result.blocks[0].stdout


def test_truncated_stdout_leads_with_actionable_feedback() -> None:
    tools = FakeSearchTools(calls=[])
    model = ScriptedModel(
        [_tool_turn("tool-1", "print('x' * 500)"), _answer_turn("done")]
    )
    agent = OriginalPTCAgent(
        model=model,
        search_tools=tools,  # type: ignore[arg-type]
        runtime=RuntimeConfig(max_turns=3, max_ptc_blocks=2, max_stdout_chars=240),
    )

    result = agent.run("research")

    block = result.blocks[0]
    assert block.stdout_truncated is True
    assert block.stdout_chars == 501
    assert len(block.stdout) == 240
    assert block.stdout.startswith("PTC_STDOUT_TRUNCATED ")
    assert "Filter and aggregate inside Python" in block.stdout
    assert model.messages_seen[1][-1]["content"] == block.stdout


def test_agent_pairs_all_blocks_when_one_api_turn_exceeds_block_budget() -> None:
    tools = FakeSearchTools(calls=[])
    model = ScriptedModel(
        [
            _multi_tool_turn(
                [
                    ("tool-1", "print('first')"),
                    ("tool-2", "print('second')"),
                ]
            ),
            _answer_turn("<result>first</result>"),
        ]
    )
    agent = OriginalPTCAgent(
        model=model,
        search_tools=tools,  # type: ignore[arg-type]
        runtime=RuntimeConfig(max_turns=3, max_ptc_blocks=1),
    )

    result = agent.run("research")

    assert result.status == "success"
    assert result.ptc_blocks == 1
    tool_messages = [
        message for message in model.messages_seen[1] if message["role"] == "tool"
    ]
    assert [message["tool_call_id"] for message in tool_messages] == [
        "tool-1",
        "tool-2",
    ]
    assert tool_messages[0]["content"].strip() == "first"
    assert tool_messages[1]["content"] == "Error: PTC block budget exhausted"


def test_agent_reports_program_failure_to_model() -> None:
    tools = FakeSearchTools(calls=[])
    model = ScriptedModel(
        [
            _tool_turn("tool-1", "raise ValueError('bad code')"),
            _answer_turn("recovered"),
        ]
    )
    agent = OriginalPTCAgent(
        model=model,
        search_tools=tools,  # type: ignore[arg-type]
        runtime=RuntimeConfig(max_turns=3, max_ptc_blocks=2),
    )

    result = agent.run("recover")

    assert result.status == "success"
    assert result.blocks[0].success is False
    assert "ValueError" in result.blocks[0].stdout
    assert model.messages_seen[1][-1]["role"] == "tool"
    assert model.messages_seen[1][-1]["content"].startswith("PTC_ERROR ")


def test_one_ptc_program_can_call_multiple_runtime_tools() -> None:
    tools = FakeSearchTools(calls=[])
    code = (
        "a = search_web(query='alpha')\n"
        "b = search_web(query='beta')\n"
        "print(a[0]['title'], b[0]['title'])"
    )
    model = ScriptedModel(
        [_tool_turn("tool-1", code), _answer_turn("alpha and beta")]
    )
    agent = OriginalPTCAgent(
        model=model,
        search_tools=tools,  # type: ignore[arg-type]
        runtime=RuntimeConfig(max_turns=3, max_ptc_blocks=2),
    )

    result = agent.run("research both in one program")

    assert result.status == "success"
    assert result.ptc_blocks == 1
    assert result.blocks[0].stdout.strip() == "alpha beta"
    assert [call["query"] for call in tools.calls] == ["alpha", "beta"]


def test_agent_can_recover_from_positional_runtime_tool_call() -> None:
    tools = FakeSearchTools(calls=[])
    model = ScriptedModel(
        [
            _tool_turn("tool-1", "print(search_web('alpha'))"),
            _tool_turn(
                "tool-2",
                "print(search_web(query='alpha')[0]['title'])",
            ),
            _answer_turn("alpha"),
        ]
    )
    agent = OriginalPTCAgent(
        model=model,
        search_tools=tools,  # type: ignore[arg-type]
        runtime=RuntimeConfig(max_turns=4, max_ptc_blocks=3),
    )

    result = agent.run("recover from invalid call syntax")

    assert result.status == "success"
    assert [block.success for block in result.blocks] == [False, True]
    assert "positional" in result.blocks[0].stdout.lower()


def test_agent_rejects_truncated_final_answer() -> None:
    tools = FakeSearchTools(calls=[])
    model = ScriptedModel(
        [
            _tool_turn("tool-1", "print('evidence')"),
            _answer_turn("partial answer", stop_reason="content_filter"),
        ]
    )
    agent = OriginalPTCAgent(
        model=model,
        search_tools=tools,  # type: ignore[arg-type]
        runtime=RuntimeConfig(max_turns=3, max_ptc_blocks=2),
    )

    result = agent.run("research")

    assert result.status == "failed"
    assert result.finish_reason == "content_filter"
    assert "content_filter" in (result.error or "")


def test_agent_continues_after_length_checkpoint() -> None:
    tools = FakeSearchTools(calls=[])
    model = ScriptedModel(
        [
            _answer_turn("partial", stop_reason="length"),
            _answer_turn("<result>complete</result>"),
        ]
    )
    agent = OriginalPTCAgent(
        model=model,
        search_tools=tools,  # type: ignore[arg-type]
        runtime=RuntimeConfig(max_turns=3, max_ptc_blocks=2),
    )

    result = agent.run("research")

    assert result.status == "success"
    assert result.answer == "<result>complete</result>"
    assert result.model_requests == 2


def test_agent_reserves_last_model_turn_for_finalization() -> None:
    tools = FakeSearchTools(calls=[])
    model = ScriptedModel(
        [
            _tool_turn("tool-1", "print('evidence')"),
            _answer_turn("<result>answer</result>"),
        ]
    )
    agent = OriginalPTCAgent(
        model=model,
        search_tools=tools,  # type: ignore[arg-type]
        runtime=RuntimeConfig(max_turns=2, max_ptc_blocks=10),
    )

    result = agent.run("research")

    assert result.status == "success"
    assert model.requests_seen[0]["tools"]
    assert model.requests_seen[1]["tools"] == []
    assert model.requests_seen[0]["system"] == model.requests_seen[1]["system"]
    assert "research tools are now unavailable" in (
        model.messages_seen[1][-1]["content"].lower()
    )
    assert model.requests_seen[1]["max_completion_tokens"] == 4096
    assert model.requests_seen[1]["thinking"] == "disabled"


def test_agent_removes_ptc_tool_after_block_budget_is_used() -> None:
    tools = FakeSearchTools(calls=[])
    model = ScriptedModel(
        [
            _tool_turn("tool-1", "print('enough evidence')"),
            _answer_turn("final answer"),
        ]
    )
    agent = OriginalPTCAgent(
        model=model,
        search_tools=tools,  # type: ignore[arg-type]
        runtime=RuntimeConfig(max_turns=3, max_ptc_blocks=1),
    )

    result = agent.run("research")

    assert result.status == "success"
    assert model.requests_seen[0]["tools"]
    assert model.requests_seen[1]["tools"] == []


def test_agent_does_not_repeat_finalization_after_textual_tool_call() -> None:
    tools = FakeSearchTools(calls=[])
    textual_call = (
        "<tool_call><function=programmatic_tool_call>"
        "<parameter=code>print('more')</parameter></function></tool_call>"
    )
    model = ScriptedModel(
        [
            _tool_turn("tool-1", "print('evidence')"),
            _answer_turn(textual_call),
            _answer_turn("final answer"),
        ]
    )
    agent = OriginalPTCAgent(
        model=model,
        search_tools=tools,  # type: ignore[arg-type]
        runtime=RuntimeConfig(max_turns=3, max_ptc_blocks=1),
    )

    result = agent.run("research")

    assert result.status == "failed"
    assert result.answer == textual_call
    assert result.model_requests == 2
    assert "textual tool-call" in (result.error or "")


def test_agent_allows_answer_without_any_ptc_block() -> None:
    tools = FakeSearchTools(calls=[])
    model = ScriptedModel([_answer_turn("<result>direct answer</result>")])
    agent = OriginalPTCAgent(
        model=model,
        search_tools=tools,  # type: ignore[arg-type]
        runtime=RuntimeConfig(max_turns=2, max_ptc_blocks=2),
    )

    result = agent.run("research first")

    assert result.status == "success"
    assert result.ptc_blocks == 0
    assert result.error is None
    assert "<question>research first</question>" in model.messages_seen[0][0]["content"]


def test_agent_places_demonstrations_before_the_current_task() -> None:
    tools = FakeSearchTools(calls=[])
    model = ScriptedModel([_answer_turn("<result>direct answer</result>")])
    demonstrations = (
        {"role": "user", "content": "demo question"},
        {"role": "assistant", "content": "<result>demo answer</result>"},
    )
    agent = OriginalPTCAgent(
        model=model,
        search_tools=tools,  # type: ignore[arg-type]
        runtime=RuntimeConfig(max_turns=2, max_ptc_blocks=1),
        demonstration_messages=demonstrations,
    )

    result = agent.run("current question")

    assert result.status == "success"
    assert model.messages_seen[0][:2] == list(demonstrations)
    assert "<question>current question</question>" in model.messages_seen[0][2]["content"]


def test_agent_recovers_empty_repetition_truncation_without_more_tools() -> None:
    tools = FakeSearchTools(calls=[])
    model = ScriptedModel(
        [
            _tool_turn("tool-1", "print('evidence')"),
            _answer_turn("", stop_reason="repetition_truncation"),
            _answer_turn("<result>final answer</result>"),
        ]
    )
    agent = OriginalPTCAgent(
        model=model,
        search_tools=tools,  # type: ignore[arg-type]
        runtime=RuntimeConfig(max_turns=4, max_ptc_blocks=3),
    )

    result = agent.run("research")

    assert result.status == "success"
    assert result.answer == "<result>final answer</result>"
    assert model.requests_seen[2]["tools"] == []
    assert "tools are now unavailable" in model.messages_seen[2][-1]["content"]


def test_agent_allows_only_one_tools_disabled_finalization_after_length() -> None:
    tools = FakeSearchTools(calls=[])
    model = ScriptedModel(
        [
            _answer_turn("partial", stop_reason="length"),
            _answer_turn("still partial", stop_reason="length"),
        ]
    )
    agent = OriginalPTCAgent(
        model=model,
        search_tools=tools,  # type: ignore[arg-type]
        runtime=RuntimeConfig(max_turns=5, max_ptc_blocks=3),
    )

    result = agent.run("research")

    assert result.status == "failed"
    assert result.model_requests == 2
    assert model.requests_seen[1]["tools"] == []
    assert result.finish_reason == "length"


def test_output_budget_reserves_one_finalization_request() -> None:
    tools = FakeSearchTools(calls=[])
    model = ScriptedModel(
        [
            _tool_turn("tool-1", "print('evidence')"),
            _answer_turn("<result>answer</result>"),
        ]
    )
    agent = OriginalPTCAgent(
        model=model,
        search_tools=tools,  # type: ignore[arg-type]
        runtime=RuntimeConfig(
            max_turns=5,
            max_ptc_blocks=4,
            max_total_output_tokens=5,
        ),
    )

    result = agent.run("research")

    assert result.status == "success"
    assert result.budget_trigger == "total_output_tokens"
    assert model.requests_seen[1]["tools"] == []


def test_agent_compacts_only_after_complete_tool_group() -> None:
    tools = FakeSearchTools(calls=[])
    tool_turn = _tool_turn("tool-1", "print('very old evidence')")
    tool_turn = ModelTurn(
        assistant_message=tool_turn.assistant_message,
        text=tool_turn.text,
        tool_calls=tool_turn.tool_calls,
        usage=TokenUsage(input_tokens=120, output_tokens=5),
        stop_reason=tool_turn.stop_reason,
    )
    model = ScriptedModel(
        [
            tool_turn,
            _answer_turn("<compacted_state>doc A supports X</compacted_state>"),
            _answer_turn("<result>X</result>"),
        ]
    )
    agent = OriginalPTCAgent(
        model=model,
        search_tools=tools,  # type: ignore[arg-type]
        runtime=RuntimeConfig(
            max_turns=4,
            max_ptc_blocks=3,
            compaction_trigger_input_tokens=100,
        ),
    )

    result = agent.run("research")

    assert result.status == "success"
    assert result.compaction_requests == 1
    assert len(result.compactions) == 1
    assert result.compactions[0].success is True
    assert model.requests_seen[1]["tools"] == []
    compacted_messages = model.requests_seen[2]["messages"]
    assert compacted_messages[0]["role"] == "user"
    assert "doc A supports X" in compacted_messages[1]["content"]
    assert "very old evidence" not in str(compacted_messages)


def test_failed_compaction_keeps_original_history() -> None:
    tools = FakeSearchTools(calls=[])
    tool_turn = _tool_turn("tool-1", "print('preserved evidence')")
    tool_turn = ModelTurn(
        assistant_message=tool_turn.assistant_message,
        text=tool_turn.text,
        tool_calls=tool_turn.tool_calls,
        usage=TokenUsage(input_tokens=120, output_tokens=5),
        stop_reason=tool_turn.stop_reason,
    )
    model = ScriptedModel(
        [
            tool_turn,
            _answer_turn(""),
            _answer_turn("<result>X</result>"),
        ]
    )
    agent = OriginalPTCAgent(
        model=model,
        search_tools=tools,  # type: ignore[arg-type]
        runtime=RuntimeConfig(
            max_turns=4,
            max_ptc_blocks=3,
            compaction_trigger_input_tokens=100,
        ),
    )

    result = agent.run("research")

    assert result.status == "success"
    assert result.compactions[0].success is False
    assert "preserved evidence" in str(model.requests_seen[2]["messages"])


def test_agent_accepts_benchmark_specific_user_prompt() -> None:
    tools = FakeSearchTools(calls=[])
    model = ScriptedModel([_answer_turn("<result>answer</result>")])
    agent = OriginalPTCAgent(
        model=model,
        search_tools=tools,  # type: ignore[arg-type]
        runtime=RuntimeConfig(),
        user_prompt_template="Local question: {question}",
    )

    agent.run("test")

    assert model.messages_seen[0][0]["content"] == "Local question: test"


def test_extract_result_tag_uses_last_complete_non_empty_tag() -> None:
    text = "<result>draft</result> notes <RESULT> final answer </RESULT>"

    assert extract_result_tag(text) == "final answer"
    assert extract_result_tag("no tagged answer") is None
    assert extract_result_tag("<result>   </result>") is None


def test_agent_stops_before_request_when_task_timeout_is_exhausted() -> None:
    tools = FakeSearchTools(calls=[])
    model = ScriptedModel([_answer_turn("<result>too late</result>")])
    agent = OriginalPTCAgent(
        model=model,
        search_tools=tools,  # type: ignore[arg-type]
        runtime=RuntimeConfig(task_timeout_seconds=0),
    )

    result = agent.run("research")

    assert result.status == "failed"
    assert result.finish_reason == "task_timeout"
    assert result.model_requests == 0
    assert "wall-clock budget" in (result.error or "")


def test_agent_passes_remaining_task_deadline_to_model(
    monkeypatch: Any,
) -> None:
    ticks = iter([100.0, 101.0, 102.0, 103.0, 104.0, 105.0])
    monkeypatch.setattr("graphptc.ptc.time.perf_counter", lambda: next(ticks))
    tools = FakeSearchTools(calls=[])
    model = ScriptedModel([_answer_turn("<result>answer</result>")])
    agent = OriginalPTCAgent(
        model=model,
        search_tools=tools,  # type: ignore[arg-type]
        runtime=RuntimeConfig(task_timeout_seconds=10),
    )

    result = agent.run("research")

    assert result.status == "success"
    assert model.requests_seen[0]["timeout_seconds"] == 9.0
