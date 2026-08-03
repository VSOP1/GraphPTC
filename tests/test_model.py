from __future__ import annotations

from types import SimpleNamespace
from typing import Any

from graphptc.config import ModelConfig
from graphptc.model import OpenAIChatModel


class FakeCompletions:
    def __init__(self, response: Any) -> None:
        self.response = response
        self.request: dict[str, Any] | None = None

    def create(self, **kwargs: Any) -> Any:
        self.request = kwargs
        return self.response


class FakeClient:
    def __init__(self, completions: FakeCompletions) -> None:
        self.chat = SimpleNamespace(completions=completions)
        self.options: dict[str, Any] | None = None

    def with_options(self, **kwargs: Any) -> FakeClient:
        self.options = kwargs
        return self


def test_mimo_turn_preserves_reasoning_content_and_tool_call() -> None:
    function = SimpleNamespace(
        name="programmatic_tool_call",
        arguments='{"code":"print(1)"}',
    )
    message = SimpleNamespace(
        content=None,
        tool_calls=[SimpleNamespace(id="call-1", function=function)],
        model_extra={"reasoning_content": "private chain state"},
    )
    response = SimpleNamespace(
        choices=[SimpleNamespace(message=message, finish_reason="tool_calls")],
        usage=SimpleNamespace(
            prompt_tokens=12,
            completion_tokens=7,
            prompt_tokens_details=SimpleNamespace(cached_tokens=3),
            completion_tokens_details=SimpleNamespace(reasoning_tokens=4),
        ),
    )
    completions = FakeCompletions(response)
    model = object.__new__(OpenAIChatModel)
    model.config = ModelConfig(
        model="mimo-v2.5",
        base_url="https://api.xiaomimimo.com/v1",
        thinking="enabled",
        temperature=0.3,
        top_p=0.8,
    )
    client = FakeClient(completions)
    model._client = client

    turn = model.create_turn(
        system="system",
        messages=[{"role": "user", "content": "task"}],
        tools=[{"type": "function"}],
        timeout_seconds=42.0,
    )

    assert turn.assistant_message["reasoning_content"] == "private chain state"
    assert turn.tool_calls[0].input == {"code": "print(1)"}
    assert turn.usage.cached_input_tokens == 3
    assert turn.usage.reasoning_tokens == 4
    assert completions.request is not None
    assert completions.request["tool_choice"] == "auto"
    assert completions.request["extra_body"] == {
        "thinking": {"type": "enabled"}
    }
    assert completions.request["temperature"] == 0.3
    assert completions.request["top_p"] == 0.8
    assert client.options == {"timeout": 42.0, "max_retries": 0}

    model.create_turn(
        system="system",
        messages=[{"role": "user", "content": "finish"}],
        tools=[],
        max_completion_tokens=4096,
        thinking="disabled",
    )
    assert completions.request is not None
    assert "tools" not in completions.request
    assert "tool_choice" not in completions.request
    assert completions.request["max_completion_tokens"] == 4096
    assert completions.request["extra_body"] == {
        "thinking": {"type": "disabled"}
    }


def test_empty_system_prompt_is_not_injected() -> None:
    response = SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(content="Answer: 1", tool_calls=[]),
                finish_reason="stop",
            )
        ],
        usage=SimpleNamespace(prompt_tokens=1, completion_tokens=1),
    )
    completions = FakeCompletions(response)
    model = object.__new__(OpenAIChatModel)
    model.config = ModelConfig(model="mimo-v2.5", thinking="disabled")
    model._client = FakeClient(completions)

    model.create_turn(
        system="",
        messages=[{"role": "user", "content": "task"}],
        tools=[],
    )

    assert completions.request is not None
    assert completions.request["messages"] == [{"role": "user", "content": "task"}]
