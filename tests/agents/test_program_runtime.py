from __future__ import annotations

from typing import Any, Callable

from codecell import BaseRuntime, CodeResult
from codecell.python import PythonValidator

from graphptc.config import RuntimeConfig
from graphptc.model import ModelTurn, TokenUsage, ToolCall
from graphptc.agents.original_ptc import OriginalPTCAgent


class EmptyTools:
    calls: list[dict[str, Any]] = []


class ScriptedModel:
    def __init__(self, turns: list[ModelTurn]) -> None:
        self._turns = iter(turns)
        self.calls = 0

    def create_turn(self, **_: Any) -> ModelTurn:
        self.calls += 1
        return next(self._turns)


class StatefulRuntime(BaseRuntime):
    def __init__(self) -> None:
        super().__init__(PythonValidator())
        self.globals: dict[str, object] = {}
        self.codes: list[str] = []
        self.completed = False
        self.last_execution_trace: dict[str, Any] = {}

    def execute(
        self,
        code: str,
        *,
        namespace: dict[str, Callable[..., Any]] | None = None,
        timeout: float | None = None,
    ) -> CodeResult:
        self.codes.append(code)
        if code == "x = 1":
            self.globals["x"] = 1
            return CodeResult(stdout="Execution successful.")
        if code == "print(x)":
            return CodeResult(stdout=f"{self.globals['x']}\n")
        if code == "long_output":
            return CodeResult(stdout="x" * 9_000)
        if code == "fail()":
            return CodeResult(
                stderr="Execution failed. Traceback:\nValueError: bad call",
                return_code=1,
            )
        if code == "apis.supervisor.complete_task()":
            self.completed = True
            return CodeResult(stdout="Execution successful.")
        raise AssertionError(code)

    @property
    def task_completed(self) -> bool:
        return self.completed


def tool_turn(call_id: str, code: str) -> ModelTurn:
    return ModelTurn(
        assistant_message={
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": call_id,
                    "type": "function",
                    "function": {"name": "programmatic_tool_call", "arguments": "{}"},
                }
            ],
        },
        text="",
        tool_calls=[ToolCall(id=call_id, name="programmatic_tool_call", input={"code": code})],
        usage=TokenUsage(),
        stop_reason="tool_calls",
    )


def answer_turn() -> ModelTurn:
    return ModelTurn(
        assistant_message={"role": "assistant", "content": "done"},
        text="done",
        tool_calls=[],
        usage=TokenUsage(),
        stop_reason="stop",
    )


def make_agent(model: ScriptedModel, runtime: StatefulRuntime, **runtime_kwargs: Any) -> OriginalPTCAgent:
    return OriginalPTCAgent(
        model=model,
        search_tools=EmptyTools(),  # type: ignore[arg-type]
        runtime=RuntimeConfig(max_turns=4, max_ptc_blocks=3, **runtime_kwargs),
        runtime_functions=(),
        program_runtime=runtime,
    )


def test_ptc_code_enters_injected_runtime_directly_and_state_persists() -> None:
    runtime = StatefulRuntime()
    model = ScriptedModel([tool_turn("one", "x = 1"), tool_turn("two", "print(x)"), answer_turn()])

    result = make_agent(model, runtime).run("task")

    assert runtime.codes == ["x = 1", "print(x)"]
    assert result.blocks[1].stdout == "1\n"
    assert result.status == "success"


def test_injected_runtime_output_uses_the_existing_8k_truncation() -> None:
    runtime = StatefulRuntime()
    model = ScriptedModel([tool_turn("one", "long_output"), answer_turn()])

    result = make_agent(model, runtime, max_stdout_chars=8_000).run("task")

    assert result.blocks[0].stdout_truncated is True
    assert len(result.blocks[0].stdout) == 8_000
    assert result.blocks[0].stdout.startswith("PTC_STDOUT_TRUNCATED ")


def test_injected_runtime_failure_is_recorded_on_the_block() -> None:
    runtime = StatefulRuntime()
    model = ScriptedModel([tool_turn("one", "fail()"), answer_turn()])

    result = make_agent(model, runtime).run("task")

    assert result.blocks[0].success is False
    assert result.blocks[0].error_type == "ValueError"
    assert "bad call" in (result.blocks[0].error_message or "")


def test_agent_stops_immediately_after_runtime_reports_task_completed() -> None:
    runtime = StatefulRuntime()
    model = ScriptedModel([tool_turn("one", "apis.supervisor.complete_task()")])

    result = make_agent(model, runtime).run("task")

    assert model.calls == 1
    assert result.status == "success"
    assert result.finish_reason == "task_completed"
