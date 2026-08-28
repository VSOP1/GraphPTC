from __future__ import annotations

import sys
from pathlib import Path

from graphptc.apiflow_runtime import APIFlowProgramRuntime


WORKER = Path(__file__).parents[1] / "fixtures" / "fake_apiflow_worker.py"


def _runtime(task_id: str = "fake-task") -> APIFlowProgramRuntime:
    return APIFlowProgramRuntime(
        worker_command=(sys.executable, str(WORKER)),
        root="/fake-apiflow",
        task_id=task_id,
    )


def test_apiflow_runtime_calls_official_tools_and_persists_python_state() -> None:
    runtime = _runtime()
    try:
        namespace = {function.__name__: function for function in runtime.functions}
        first = runtime.execute(
            "saved = search(query='', kind=None)\nprint(saved['counter'])",
            namespace=namespace,
        )
        first_trace = runtime.last_execution_trace
        second = runtime.execute("print(saved['name'])")

        assert first.stdout == "1\n"
        assert second.stdout == "search\n"
        assert first_trace["external_actions"][0]["name"] == "search"
        assert runtime.telemetry()["tool_calls"] == 1
    finally:
        runtime.close()


def test_apiflow_runtime_evaluates_and_isolates_tasks() -> None:
    first = _runtime("one")
    try:
        first.execute("search(query='x')", namespace={f.__name__: f for f in first.functions})
        assert first.evaluate("done")["passed"] is True
    finally:
        first.close()

    second = _runtime("two")
    try:
        result = second.execute("print(saved)")
        assert result.return_code == 1
        assert "NameError" in result.stderr
    finally:
        second.close()


def test_apiflow_runtime_enforces_block_timeout_on_windows() -> None:
    runtime = _runtime()
    try:
        result = runtime.execute("while True:\n    pass", timeout=0.02)

        assert result.timed_out is True
        assert result.return_code == -1
        assert "PTC block timed out" in result.stderr
    finally:
        runtime.close()
