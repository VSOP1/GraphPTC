from __future__ import annotations

import json
from types import SimpleNamespace

from graphptc.goal_adaptation import GoalGraphAdaptation
from graphptc.graph_agent import GraphAgentHooks
from graphptc.tool_effects import ToolEffectContract


def _trace(
    *,
    success: bool = True,
    error_type: str | None = None,
    error_message: str | None = None,
) -> SimpleNamespace:
    return SimpleNamespace(
        success=success,
        code="pass",
        stdout="ok",
        runtime_calls=0,
        program_analysis={},
        runtime_trace={},
        error_type=error_type,
        error_message=error_message,
    )


def test_goal_graph_controller_runs_a_non_retrieval_dependency_workflow() -> None:
    def lookup_rows(*, table: str) -> list[int]:
        return [2, 3, 5] if table == "orders" else []

    def aggregate_values(*, values: list[int]) -> int:
        return sum(values)

    controller = GoalGraphAdaptation(
        {"lookup_rows": lookup_rows, "aggregate_values": aggregate_values},
        {
            "lookup_rows": ToolEffectContract(
                name="lookup_rows", deterministic=True, cacheable=True
            ),
            "aggregate_values": ToolEffectContract(
                name="aggregate_values",
                effect="pure",
                deterministic=True,
                cacheable=True,
            ),
        },
        task="Compute the order total",
    )
    hooks = GraphAgentHooks.from_controller(controller)
    functions = {function.__name__: function for function in hooks.runtime_functions}

    hooks.ptc_call_metadata_callback(
        {"action": "CONTINUE", "target": "task", "expected_change": "declare goal"}
    )
    functions["graph_declare_goal"](
        goal_id="total", description="compute the order total", depends_on=[]
    )
    hooks.block_observation_factory(_trace())

    hooks.ptc_call_metadata_callback(
        {
            "action": "CONTINUE",
            "target": "goal:total",
            "expected_change": "produce total and complete goal",
        }
    )
    rows = functions["lookup_rows"](table="orders")
    total = functions["aggregate_values"](values=rows)
    functions["graph_complete_goal"](goal_id="total")
    observation = hooks.block_observation_factory(_trace())
    payload = json.loads(observation.removeprefix("GRAPH_DELTA "))

    assert total == 10
    assert payload["action_verification"]["realized"] is True
    assert payload["next_action_contract"]["available_actions"][-1] == "ANSWER"
    assert controller.telemetry()["goal_states"] == {"COMPLETE": 1}


def test_generic_controller_exposes_branch_frontier_after_equivalent_effects() -> None:
    controller = GoalGraphAdaptation(
        {"read_value": lambda *, key: {"key": key, "value": "same"}},
        {"read_value": ToolEffectContract(name="read_value")},
        task="Find a value",
    )
    hooks = GraphAgentHooks.from_controller(controller)
    read_value = {
        function.__name__: function for function in hooks.runtime_functions
    }["read_value"]

    observation = ""
    for index in range(3):
        hooks.ptc_call_metadata_callback(
            {
                "action": "CONTINUE" if index < 2 else "REPLAN",
                "target": "task",
                "expected_change": f"path {index}",
            }
        )
        read_value(key="x")
        observation = hooks.block_observation_factory(_trace())

    payload = json.loads(observation.removeprefix("GRAPH_DELTA "))
    contract = payload["next_action_contract"]
    assert contract["action_opportunities"][0]["action"] == "REPLAN"
    assert contract["branch_frontier"]["productive_paths"]
    assert contract["branch_frontier"]["exhausted_paths"]


def test_generic_controller_routes_execution_failure_to_patch() -> None:
    controller = GoalGraphAdaptation({}, {}, task="Compute a value")
    hooks = GraphAgentHooks.from_controller(controller)

    hooks.ptc_call_metadata_callback(
        {
            "action": "CONTINUE",
            "target": "task",
            "expected_change": "compute the value",
        }
    )
    failed = hooks.block_observation_factory(
        _trace(success=False, error_type="NameError", error_message="missing name")
    )
    failure_contract = json.loads(failed.removeprefix("GRAPH_DELTA "))[
        "next_action_contract"
    ]

    assert failure_contract["action_opportunities"][0]["action"] == "PATCH"
    assert failure_contract["last_failure"]["error_type"] == "NameError"

    hooks.ptc_call_metadata_callback(
        {
            "action": "PATCH",
            "target": "task",
            "expected_change": "re-execute the corrected computation",
        }
    )
    repaired = json.loads(
        hooks.block_observation_factory(_trace()).removeprefix("GRAPH_DELTA ")
    )
    assert repaired["action_verification"]["realized"] is True
