from __future__ import annotations

import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import pytest

from graphptc.appworld_runtime import AppWorldProgramRuntime
from graphptc.config import RuntimeConfig
from graphptc.goal_adaptation import GoalGraphAdaptation
from graphptc.graph_agent import GraphAgentHooks
from graphptc.model import ModelTurn, TokenUsage, ToolCall
from graphptc.ptc import OriginalPTCAgent


WORKER = Path(__file__).parents[1] / "fixtures" / "fake_appworld_worker.py"


def runtime(task_id: str) -> AppWorldProgramRuntime:
    return AppWorldProgramRuntime(
        worker_command=(sys.executable, str(WORKER)),
        root="/fake-root",
        task_id=task_id,
        experiment_name="graphptc",
    )


def test_appworld_runtime_persists_state_and_reports_effects() -> None:
    world = runtime("task-one")
    try:
        world.execute("counter += 1")
        result = world.execute("print(counter)")

        assert result.stdout == "1\n"
        assert world.metadata["instruction"] == "instruction for task-one"
        assert world.last_execution_trace["external_actions"][0]["effect"] == "write"
    finally:
        world.close()


def test_appworld_runtime_records_failure_and_complete_task() -> None:
    world = runtime("task-two")
    try:
        failed = world.execute("fail()")
        assert failed.return_code == 1
        assert "ValueError" in failed.stderr
        assert world.last_execution_trace["external_actions"][0]["success"] is None
        assert world.last_execution_trace["external_actions"][0]["outcome_unknown"] is True

        completed = world.execute("apis.supervisor.complete_task()")
        assert completed.return_code == 0
        assert world.task_completed is True
    finally:
        world.close()


def test_appworld_runtime_redacts_secrets_from_graph_trace() -> None:
    world = runtime("task-secrets")
    try:
        result = world.execute("secret()")
        serialized = json.dumps(world.last_execution_trace)

        assert result.return_code == 0
        assert "plain-secret" not in serialized
        assert "token-secret" not in serialized
        assert world.last_execution_trace["api_calls"][0]["data"] == {
            "password": "<redacted>",
            "access_token": "<redacted>",
            "profile": {"name": "visible"},
        }
    finally:
        world.close()


def test_appworld_runtime_isolates_tasks_and_closes_workers() -> None:
    first = runtime("task-a")
    first.execute("counter += 1")
    first.close()

    second = runtime("task-b")
    try:
        assert second.execute("print(counter)").stdout == "0\n"
    finally:
        second.close()

    assert first.telemetry()["closed"] is True
    assert second.telemetry()["closed"] is True
    assert first.telemetry()["termination_confirmed"] is True
    assert second.telemetry()["termination_confirmed"] is True
    with pytest.raises(RuntimeError, match="closed"):
        first.execute("print(counter)")


def test_two_appworld_workers_are_isolated_while_running_concurrently() -> None:
    def run_task(task_id: str, increments: int) -> tuple[str, bool]:
        world = runtime(task_id)
        try:
            for _ in range(increments):
                world.execute("counter += 1")
            return world.execute("print(counter)").stdout, world.task_completed
        finally:
            world.close()

    with ThreadPoolExecutor(max_workers=2) as executor:
        first = executor.submit(run_task, "concurrent-a", 1)
        second = executor.submit(run_task, "concurrent-b", 2)

    assert first.result() == ("1\n", False)
    assert second.result() == ("2\n", False)


def test_appworld_timeout_clears_trace_and_prevents_silent_world_restart() -> None:
    world = runtime("task-timeout")
    try:
        world.execute("counter += 1")
        assert world.last_execution_trace["external_actions"]

        timed_out = world.execute("hang()", timeout=0.01)

        assert timed_out.timed_out is True
        assert world.last_execution_trace["external_actions"] == []
        assert world.last_execution_trace["failure"]["type"] == "timeout"
        assert world.last_execution_trace["effects_unknown"] is True
        assert world.last_execution_trace["api_calls_complete"] is False
        assert world.telemetry()["broken"] is True
        assert world.metadata["task_id"] == "task-timeout"
        with pytest.raises(RuntimeError, match="unusable"):
            world.execute("print(counter)")
        assert world.last_execution_trace["failure"]["type"] == "broken_runtime"
    finally:
        world.close()


def test_appworld_close_is_best_effort_when_worker_exits_without_reply() -> None:
    world = runtime("close-crash")
    world.execute("counter += 1")

    world.close()

    assert world.telemetry()["closed"] is True
    assert world.telemetry()["termination_confirmed"] is True
    assert world.telemetry()["close_error"] is not None


class _EmptyTools:
    calls: list[dict[str, Any]] = []


class _ScriptedModel:
    def __init__(self, codes: list[str]) -> None:
        self._codes = iter(codes)
        self.calls = 0

    def create_turn(self, **_: Any) -> ModelTurn:
        self.calls += 1
        code = next(self._codes)
        call_id = f"call-{self.calls}"
        return ModelTurn(
            assistant_message={
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": call_id,
                        "type": "function",
                        "function": {
                            "name": "programmatic_tool_call",
                            "arguments": "{}",
                        },
                    }
                ],
            },
            text="",
            tool_calls=[
                ToolCall(
                    id=call_id,
                    name="programmatic_tool_call",
                    input={
                        "code": code,
                        "action": "CONTINUE",
                        "target": "task",
                        "expected_change": f"execute {code}",
                    },
                )
            ],
            usage=TokenUsage(),
            stop_reason="tool_calls",
        )


def test_agent_appworld_runtime_evaluate_then_close_contract() -> None:
    world = runtime("task-agent-contract")
    model = _ScriptedModel(
        ["counter += 1", "print(counter)", "apis.supervisor.complete_task()"]
    )
    controller = GoalGraphAdaptation({}, {}, task="increment and finish", expose_graph_api=False)
    hooks = GraphAgentHooks.from_controller(controller)
    hook_kwargs = hooks.agent_kwargs()
    hook_kwargs["runtime_functions"] = ()
    agent = OriginalPTCAgent(
        model=model,
        search_tools=_EmptyTools(),  # type: ignore[arg-type]
        runtime=RuntimeConfig(max_turns=4, max_ptc_blocks=4),
        program_runtime=world,
        **hook_kwargs,
    )

    result = agent.run("increment and finish")

    assert model.calls == 3
    assert [block.code for block in result.blocks] == [
        "counter += 1",
        "print(counter)",
        "apis.supervisor.complete_task()",
    ]
    assert result.blocks[1].stdout == "1\n"
    assert result.blocks[0].runtime_trace["external_actions"][0]["effect"] == "write"
    assert result.finish_reason == "task_completed"
    assert world.evaluate() == {"success": True}

    world.close()
    with pytest.raises(RuntimeError, match="closed"):
        world.execute("print(counter)")


def test_agent_stops_after_fatal_appworld_worker_timeout() -> None:
    world = runtime("task-agent-timeout")
    model = _ScriptedModel(["hang()", "print(counter)"])
    agent = OriginalPTCAgent(
        model=model,
        search_tools=_EmptyTools(),  # type: ignore[arg-type]
        runtime=RuntimeConfig(
            max_turns=4,
            max_ptc_blocks=4,
            code_timeout_seconds=0.01,
        ),
        program_runtime=world,
        runtime_functions=(),
    )

    result = agent.run("time out")

    assert model.calls == 1
    assert len(result.blocks) == 1
    assert result.blocks[0].runtime_trace["failure"]["type"] == "timeout"
    assert result.finish_reason == "runtime_failure"
    assert result.status == "failed"
    world.close()
