from __future__ import annotations

import sys
from pathlib import Path

import pytest

from graphptc.appworld_runtime import AppWorldProgramRuntime


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

        completed = world.execute("apis.supervisor.complete_task()")
        assert completed.return_code == 0
        assert world.task_completed is True
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
    with pytest.raises(RuntimeError, match="closed"):
        first.execute("print(counter)")
