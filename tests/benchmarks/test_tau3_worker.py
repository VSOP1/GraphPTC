from __future__ import annotations

from types import SimpleNamespace

from graphptc.tau3_worker import (
    _ptc_call_error,
    _stamp_trial,
    _valid_ptc_call_count,
    _validated_ptc_calls,
)


def test_multiple_programmatic_tool_calls_are_kept_in_model_order() -> None:
    calls = [
        SimpleNamespace(name="programmatic_tool_call", input={"code": "print(1)"}),
        SimpleNamespace(name="programmatic_tool_call", input={"code": "print(2)"}),
    ]
    assert _validated_ptc_calls(calls) == calls


def test_direct_environment_tool_call_becomes_repairable_ptc_observation() -> None:
    call = SimpleNamespace(name="get_order_details", input={"order_id": "1"})
    assert _ptc_call_error(call) == "Error: unknown tool: get_order_details"


def test_only_valid_ptc_calls_consume_block_budget() -> None:
    calls = [
        SimpleNamespace(name="get_order_details", input={"order_id": "1"}),
        SimpleNamespace(name="programmatic_tool_call", input={"code": "print(1)"}),
        SimpleNamespace(name="programmatic_tool_call", input={"code": ""}),
    ]
    assert _valid_ptc_call_count(calls) == 1


def test_single_task_simulation_is_stamped_with_official_trial_index() -> None:
    simulation = SimpleNamespace(trial=None)
    assert _stamp_trial(simulation, 3) is simulation
    assert simulation.trial == 3
