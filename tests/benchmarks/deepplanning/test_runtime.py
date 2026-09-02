from __future__ import annotations

import sys
from pathlib import Path

from graphptc.benchmarks.deepplanning.runtime import DeepPlanningProgramRuntime


FIXTURE = Path(__file__).parents[2] / "fixtures" / "fake_deepplanning_worker.py"


def test_official_worker_runtime_records_actions_and_effects() -> None:
    runtime = DeepPlanningProgramRuntime(
        worker_command=(sys.executable, str(FIXTURE)),
        request={"domain": "shopping", "sample_id": "1"},
        timeout_seconds=5,
    )
    try:
        assert runtime.metadata["tool_names"] == ["fake_tool"]
        result = runtime.execute("print('x')")
        assert result.return_code == 0
        assert result.stdout == "ok\n"
        assert runtime.last_execution_trace["external_actions"][0]["tool"] == "fake_tool"
        assert runtime.last_execution_trace["state_effects"][0]["artifact"] == "cart.json"
        assert runtime.telemetry()["tool_calls"] == 1
    finally:
        runtime.close()


def test_official_worker_failure_is_projected() -> None:
    runtime = DeepPlanningProgramRuntime(
        worker_command=(sys.executable, str(FIXTURE)),
        request={"domain": "travel", "sample_id": "0"},
        timeout_seconds=5,
    )
    try:
        result = runtime.execute("fail()")
        assert result.return_code == 1
        assert runtime.last_execution_trace["failure"]["type"] == "execution_error"
    finally:
        runtime.close()
