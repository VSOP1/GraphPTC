from __future__ import annotations

import json
import hashlib
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


def test_generic_controller_projects_external_api_actions_and_state_effects() -> None:
    controller = GoalGraphAdaptation({}, {}, task="Update state", expose_graph_api=False)
    hooks = GraphAgentHooks.from_controller(controller)
    hooks.ptc_call_metadata_callback(
        {"action": "CONTINUE", "target": "task", "expected_change": "update task state"}
    )
    trace = _trace()
    trace.runtime_trace = {
        "external_actions": [
            {
                "name": "POST /example/items",
                "arguments": {"method": "post", "url": "/example/items", "data": {}},
                "effect": "write",
                "success": True,
            }
        ]
    }

    payload = json.loads(hooks.block_observation_factory(trace).removeprefix("GRAPH_DELTA "))
    graph = controller.graph_artifact()

    assert payload["actual_effect"]["state_changes"]
    assert any(node["kind"] == "API_ACTION" for node in graph["nodes"])
    assert any(edge["type"] == "mutates" for edge in graph["edges"])


def test_failed_block_keeps_prior_api_outcome_unknown_without_inventing_state_effect() -> None:
    controller = GoalGraphAdaptation({}, {}, task="Update state", expose_graph_api=False)
    hooks = GraphAgentHooks.from_controller(controller)
    hooks.ptc_call_metadata_callback(
        {"action": "PATCH", "target": "task", "expected_change": "attempt update"}
    )
    trace = _trace(success=False, error_type="ValueError", error_message="later failure")
    trace.runtime_trace = {
        "external_actions": [
            {
                "name": "POST /example/items",
                "arguments": {"method": "post", "url": "/example/items", "data": {}},
                "effect": "write",
                "success": None,
                "outcome_unknown": True,
            }
        ]
    }

    hooks.block_observation_factory(trace)
    graph = controller.graph_artifact()
    action = next(node for node in graph["nodes"] if node["kind"] == "API_ACTION")

    assert action["data"]["success"] is None
    assert action["data"]["outcome_unknown"] is True
    assert not any(node["kind"] == "STATE_EFFECT" for node in graph["nodes"])


def test_declared_write_without_official_state_delta_does_not_invent_state_effect() -> None:
    controller = GoalGraphAdaptation({}, {}, task="Idempotent update", expose_graph_api=False)
    hooks = GraphAgentHooks.from_controller(controller)
    hooks.ptc_call_metadata_callback(
        {"action": "CONTINUE", "target": "task", "expected_change": "ensure state"}
    )
    trace = _trace()
    trace.runtime_trace = {
        "external_actions": [
            {
                "name": "ensure_enabled",
                "arguments": {},
                "effect": "write",
                "success": True,
                "state_changed": False,
                "effect_basis": "official_tool_metadata_and_db_hash",
            }
        ]
    }

    hooks.block_observation_factory(trace)
    graph = controller.graph_artifact()
    action = next(node for node in graph["nodes"] if node["kind"] == "API_ACTION")

    assert action["data"]["effect"] == "write"
    assert action["data"]["state_changed"] is False
    assert not any(node["kind"] == "STATE_EFFECT" for node in graph["nodes"])


def test_disabled_host_inspection_is_not_offered() -> None:
    controller = GoalGraphAdaptation(
        {}, {}, task="Inspect dependencies", expose_graph_api=False
    )

    assessment = json.loads(controller.initial_observation().removeprefix("GRAPH_ASSESSMENT "))

    assert "INSPECT" not in assessment["available_actions"]
    assert not any(
        item["action"] == "INSPECT" for item in assessment["action_opportunities"]
    )


def test_host_trace_inspection_runs_after_current_block_projection() -> None:
    controller = GoalGraphAdaptation(
        {},
        {},
        task="Inspect dependencies",
        expose_graph_api=False,
        host_inspection_enabled=True,
    )
    hooks = GraphAgentHooks.from_controller(controller)
    hooks.ptc_call_metadata_callback(
        {
            "action": "INSPECT",
            "target": "task",
            "expected_change": "inspect the current block",
            "inspection": {"view": "trace", "node_id": "block:1"},
        }
    )

    payload = json.loads(hooks.block_observation_factory(_trace()).removeprefix("GRAPH_DELTA "))

    assert payload["inspection_result"]["status"] == "ok"
    assert payload["inspection_result"]["request"] == {
        "view": "trace",
        "node_id": "block:1",
    }
    assert payload["inspection_result"]["result"]["node"]["id"] == "block:1"
    assert payload["inspection_result"]["returned"] is True
    assert payload["action_verification"]["realized"] is True
    telemetry = controller.telemetry()["inspection"]
    assert telemetry == {
        "declared": 1,
        "well_formed": 1,
        "query_attempts": 1,
        "succeeded": 1,
        "failed": 0,
        "responses_emitted": 1,
        "results_returned": 1,
    }


def test_host_frontier_returns_projected_artifacts_without_runtime_graph_api() -> None:
    controller = GoalGraphAdaptation(
        {},
        {},
        task="Inspect dependencies",
        expose_graph_api=False,
        host_inspection_enabled=True,
    )
    hooks = GraphAgentHooks.from_controller(controller)
    hooks.ptc_call_metadata_callback(
        {
            "action": "INSPECT",
            "target": "task",
            "expected_change": "inspect the projected frontier",
            "inspection": {"view": "frontier"},
        }
    )

    payload = json.loads(hooks.block_observation_factory(_trace()).removeprefix("GRAPH_DELTA "))

    artifacts = payload["inspection_result"]["result"]["reusable_artifacts"]
    assert artifacts
    assert any(value.startswith("artifact:block:1:") for value in artifacts)


def test_host_inspection_is_bounded_stable_and_read_only() -> None:
    controller = GoalGraphAdaptation(
        {},
        {},
        task="Inspect dependencies",
        expose_graph_api=False,
        host_inspection_enabled=True,
    )
    hooks = GraphAgentHooks.from_controller(controller)
    hooks.ptc_call_metadata_callback(
        {"action": "CONTINUE", "target": "task", "expected_change": "create block"}
    )
    hooks.block_observation_factory(_trace())
    before = json.dumps(controller.graph_artifact(), sort_keys=True)

    first = controller.graph_trace(node_id="block:1")
    second = controller.graph_trace(node_id="block:1")
    first["node"]["data"]["success"] = "mutated"
    after = json.dumps(controller.graph_artifact(), sort_keys=True)

    assert second["node"]["data"]["success"] is True
    assert first["edges"] == second["edges"]
    assert hashlib.sha256(before.encode()).digest() == hashlib.sha256(after.encode()).digest()
    assert len(second["edges"]) <= 12


def test_invalid_host_inspection_is_structured_and_unrealized() -> None:
    controller = GoalGraphAdaptation(
        {},
        {},
        task="Inspect dependencies",
        expose_graph_api=False,
        host_inspection_enabled=True,
    )
    hooks = GraphAgentHooks.from_controller(controller)
    hooks.ptc_call_metadata_callback(
        {
            "action": "INSPECT",
            "target": "task",
            "expected_change": "inspect a missing node",
            "inspection": {"view": "trace", "node_id": "missing"},
        }
    )

    payload = json.loads(hooks.block_observation_factory(_trace()).removeprefix("GRAPH_DELTA "))

    assert payload["inspection_result"]["status"] == "error"
    assert "unknown graph node" in payload["inspection_result"]["error"]
    assert payload["inspection_result"]["returned"] is False
    assert payload["action_verification"]["realized"] is False
    telemetry = controller.telemetry()["inspection"]
    assert telemetry["well_formed"] == 1
    assert telemetry["query_attempts"] == 1
    assert telemetry["succeeded"] == 0
    assert telemetry["failed"] == 1
    assert telemetry["responses_emitted"] == 1
    assert telemetry["results_returned"] == 0


def test_host_inspection_render_is_bounded_and_keeps_useful_result_summary() -> None:
    controller = GoalGraphAdaptation(
        {},
        {},
        task="Inspect dependencies",
        max_observation_chars=800,
        expose_graph_api=False,
        host_inspection_enabled=True,
    )
    hooks = GraphAgentHooks.from_controller(controller)
    hooks.ptc_call_metadata_callback(
        {
            "action": "INSPECT",
            "target": "missing-" + "x" * 10_000,
            "expected_change": "inspect the current block",
            "inspection": {"view": "trace", "node_id": "block:1"},
        }
    )

    observation = hooks.block_observation_factory(_trace())
    payload = json.loads(observation.removeprefix("GRAPH_DELTA "))

    assert len(observation) <= 800
    assert payload["inspection_result"]["status"] == "ok"
    assert payload["inspection_result"]["returned"] is True
    assert payload["inspection_result"]["result_summary"]["node"] == {
        "id": "block:1",
        "kind": "BLOCK",
    }
    assert payload["action_verification"]["realized"] is True


def test_truncated_stdout_and_graph_delta_remain_separately_bounded_json() -> None:
    controller = GoalGraphAdaptation(
        {},
        {},
        task="Inspect a large block",
        max_observation_chars=3_200,
        expose_graph_api=False,
        host_inspection_enabled=True,
    )
    hooks = GraphAgentHooks.from_controller(controller)
    hooks.ptc_call_metadata_callback(
        {
            "action": "INSPECT",
            "target": "task",
            "expected_change": "inspect the large projected block",
            "inspection": {"view": "frontier"},
        }
    )
    trace = _trace()
    trace.stdout = "PTC_STDOUT_TRUNCATED " + "x" * (8_000 - 21)

    graph_delta = hooks.block_observation_factory(trace)
    combined = f"{trace.stdout}\n\n{graph_delta}"
    parsed = json.loads(combined.split("GRAPH_DELTA ", 1)[1])

    assert len(trace.stdout) == 8_000
    assert len(graph_delta) <= 3_200
    assert parsed["inspection_result"]["status"] == "ok"
    assert parsed["inspection_result"]["result"]["reusable_artifacts"]
