from __future__ import annotations

import sys
import inspect
from pathlib import Path

from graphptc.benchmarks.toolhop.runtime import ToolHopProgramRuntime
from toolregistry import ToolRegistry


WORKER = (
    Path(__file__).parents[3]
    / "src"
    / "graphptc"
    / "benchmarks"
    / "toolhop"
    / "official_worker.py"
)


def _task(task_id: int) -> dict:
    return {
        "id": task_id,
        "functions": [
            "def add_values(left, right=4):\n    return left + right",
            "def fail_value(value):\n    raise ValueError(value)",
        ],
        "tools": {
            "add": {
                "name": "add_values",
                "description": "Add values.",
                "parameters": {
                    "type": "object",
                    "properties": {"left": {"type": "integer"}, "right": {"type": "integer"}},
                    "required": ["left"],
                },
            },
            "fail": {
                "name": "fail_value",
                "description": "Fail.",
                "parameters": {
                    "type": "object",
                    "properties": {"value": {"type": "string"}},
                    "required": ["value"],
                },
            },
        },
    }


def test_toolhop_runtime_calls_official_functions_and_persists_ptc_state() -> None:
    runtime = ToolHopProgramRuntime(
        worker_command=(sys.executable, str(WORKER)), task=_task(1), timeout_seconds=10
    )
    functions = {function.__name__: function for function in runtime.functions}
    assert inspect.signature(functions["add_values"]).parameters["right"].default == 4
    try:
        first = runtime.execute("saved = add_values(left=2, right=3)", namespace=functions)
        second = runtime.execute("print(saved * 2)")
        failed = runtime.execute("fail_value(value='expected')")
    finally:
        runtime.close()

    assert first.return_code == 0
    assert second.stdout.strip() == "10"
    assert failed.return_code == 1
    assert "expected" in failed.stderr
    assert runtime.telemetry()["tool_calls"] == 2
    assert runtime.telemetry()["failed_tool_calls"] == 1
    assert runtime.telemetry()["closed"] is True


def test_toolhop_runtime_isolates_task_workers() -> None:
    one = ToolHopProgramRuntime(
        worker_command=(sys.executable, str(WORKER)), task=_task(1), timeout_seconds=10
    )
    two = ToolHopProgramRuntime(
        worker_command=(sys.executable, str(WORKER)), task=_task(2), timeout_seconds=10
    )
    try:
        one.execute("marker = 7")
        result = two.execute("print(marker)")
    finally:
        one.close()
        two.close()
    assert result.return_code == 1
    assert "NameError" in result.stderr


def test_toolhop_registry_preserves_omitted_official_parameters() -> None:
    task = _task(3)
    task["functions"][0] = "def add_values(left, right):\n    return left + right"
    runtime = ToolHopProgramRuntime(
        worker_command=(sys.executable, str(WORKER)), task=task, timeout_seconds=10
    )
    registry = ToolRegistry()
    registry.register(runtime.functions[0])
    registry.ptc.enable(runtime=runtime)
    try:
        omitted = registry.invoke(
            "programmatic_tool_call", {"code": "print(add_values(left=2))"}
        )
        explicit_null = registry.invoke(
            "programmatic_tool_call",
            {"code": "print(add_values(left=2, right=None))"},
        )
    finally:
        runtime.close()
    assert "missing 1 required positional argument: 'right'" in omitted
    assert "unsupported operand" in explicit_null
    assert runtime.calls[0]["arguments"] == {"left": 2}
    assert runtime.calls[1]["arguments"] == {"left": 2, "right": None}
