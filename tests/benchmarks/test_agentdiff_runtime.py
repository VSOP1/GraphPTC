from __future__ import annotations

import sys
from pathlib import Path

import pytest

from graphptc.agentdiff_runtime import AgentDiffProgramRuntime


WORKER = Path(__file__).parents[1] / "fixtures" / "fake_agentdiff_worker.py"
TASK = {
    "test_id": "box_001",
    "question": "Create one item.",
    "answer": {"assertions": [{"diff_type": "added", "entity": "items"}]},
    "service": "box",
    "info": {
        "service": "box",
        "seed_template": "box_default",
        "impersonate_user_id": "user",
    },
}


def runtime(trial: int = 0) -> AgentDiffProgramRuntime:
    return AgentDiffProgramRuntime(
        worker_command=(sys.executable, str(WORKER)),
        task=TASK,
        trial=trial,
        official_commit="official",
    )


def test_ptc_code_enters_official_executor_and_service_state_persists() -> None:
    world = runtime()
    try:
        first = world.execute("write_service_state()")
        second = world.execute("print_service_state()")

        assert first.return_code == 0
        assert second.stdout == "1\n"
        assert world.last_execution_trace["state_effects"] == []
        assert world.metadata["python_state_persistent"] is False
    finally:
        world.close()


def test_runtime_records_api_effect_and_failure_without_gold_assertions() -> None:
    world = runtime()
    try:
        world.execute("write_service_state()")
        trace = world.last_execution_trace
        assert trace["external_actions"][0]["effect_basis"] == "official_state_diff"
        assert trace["state_effects"] == [
            {"diff_type": "added", "entity": "items", "count": 1}
        ]
        assert "assertions" not in str(trace)

        failed = world.execute("fail()")
        assert failed.return_code == 1
        assert "ValueError" in failed.stderr
        assert world.last_execution_trace["failure"]["type"] == "execution_error"
    finally:
        world.close()


def test_runtime_evaluates_then_deletes_isolated_environment() -> None:
    first = runtime(0)
    first.execute("write_service_state()")
    assert first.evaluate()["passed"] is True
    first.close()
    assert first.telemetry()["environment_deleted"] is True
    assert first.telemetry()["termination_confirmed"] is True
    with pytest.raises(RuntimeError, match="closed"):
        first.execute("print_service_state()")

    second = runtime(1)
    try:
        assert second.execute("print_service_state()").stdout == "0\n"
    finally:
        second.close()
