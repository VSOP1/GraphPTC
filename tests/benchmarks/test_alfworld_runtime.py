from __future__ import annotations

import sys
from pathlib import Path

from graphptc.alfworld_runtime import AlfWorldProgramRuntime

WORKER = Path(__file__).parents[1] / "fixtures" / "fake_alfworld_worker.py"


def _runtime() -> AlfWorldProgramRuntime:
    return AlfWorldProgramRuntime(
        worker_command=(sys.executable, str(WORKER)),
        data_root="/fake",
        official_config_path="/fake/config.yaml",
        split="eval_in_distribution",
        task_id="fake/task",
        seed=42,
        max_steps=3,
    )


def test_alfworld_runtime_persists_python_and_projects_environment_actions() -> None:
    runtime = _runtime()
    try:
        first = runtime.execute('counter += 1\nprint(act("look")["observation"])')
        second = runtime.execute('print(counter)\nact("finish")')

        assert first.stdout == "observed look\n"
        assert second.stdout == "1\n"
        assert runtime.task_completed is True
        assert runtime.last_execution_trace["won"] is True
        assert runtime.last_execution_trace["external_actions"][0]["name"] == "finish"
        assert runtime.evaluate()["success"] is True
    finally:
        runtime.close()

    assert runtime.telemetry()["runtime"] == "alfworld"
    assert runtime.telemetry()["termination_confirmed"] is True
