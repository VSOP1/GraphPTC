from __future__ import annotations

import json
import copy
import hashlib
from collections import Counter
from functools import wraps
from typing import Any, Callable, Mapping

from .episode_graph import EpisodeGraph
from .execution_projection import PTCExecutionProjection
from .graph_agent import GraphProgressTracker
from .tool_effects import ToolEffectContract, ToolGraphRuntime


class GoalGraphAdaptation:
    """Domain-neutral online graph control for arbitrary tool-using agents."""

    def __init__(
        self,
        tools: Mapping[str, Callable[..., Any]],
        contracts: Mapping[str, ToolEffectContract],
        *,
        task: str,
        max_observation_chars: int = 3_200,
        expose_graph_api: bool = True,
        host_inspection_enabled: bool = False,
    ) -> None:
        self._graph = EpisodeGraph(task=task)
        self._runtime = ToolGraphRuntime(self._graph)
        self._execution = PTCExecutionProjection(self._graph)
        self._progress = GraphProgressTracker(self._graph)
        if max_observation_chars < 512:
            raise ValueError("max_observation_chars must be at least 512")
        self._max_observation_chars = max_observation_chars
        self._expose_graph_api = expose_graph_api
        self._host_inspection_enabled = host_inspection_enabled
        self._tool_functions: list[Callable[..., Any]] = []
        for name, function in tools.items():
            contract = contracts[name]
            if contract.name != name:
                raise ValueError(f"tool contract name mismatch for {name!r}")
            self._runtime.register(function, contract)
            self._tool_functions.append(self._tool_wrapper(name, function))
        self._goals: list[str] = []
        self._goal_dependencies: dict[str, list[str]] = {}
        self._actions: Counter[str] = Counter()
        self._current_action: dict[str, Any] | None = None
        self._current_target = "task"
        self._declared_inputs: tuple[str, ...] = ()
        self._inspection_count = 0
        self._artifact_loads = 0
        self._observation_calls = 0
        self._action_history: list[dict[str, Any]] = []
        self._invalid_action_targets = 0
        self._realized_actions = 0
        self._missed_actions = 0
        self._inspection_well_formed = 0
        self._inspection_query_attempts = 0
        self._inspection_succeeded = 0
        self._inspection_failed = 0
        self._inspection_responses_emitted = 0
        self._inspection_results_returned = 0
        self._pending_inspection_request: Any = None
        self._current_inspection_result: dict[str, Any] | None = None

    def runtime_functions(self) -> tuple[Callable[..., Any], ...]:
        if not self._expose_graph_api:
            return tuple(self._tool_functions)
        return (
            *self._tool_functions,
            self.graph_declare_goal,
            self.graph_complete_goal,
            self.graph_frontier,
            self.graph_trace,
            self.graph_load_artifact,
        )

    def graph_declare_goal(
        self,
        *,
        goal_id: str,
        description: str,
        depends_on: list[str] | None = None,
    ) -> dict[str, Any]:
        node_id = _goal_id(goal_id)
        dependencies = [_goal_id(value) for value in (depends_on or [])]
        unknown = [value for value in dependencies if value not in self._graph.nodes]
        if unknown:
            raise ValueError(f"unknown goal dependencies: {unknown}")
        self._graph.add_node(
            node_id,
            "GOAL",
            {"description": str(description)[:500], "status": "PENDING"},
        )
        if node_id not in self._goals:
            self._goals.append(node_id)
        self._goal_dependencies[node_id] = dependencies
        self._graph.add_edge("requires", "task", node_id)
        for dependency in dependencies:
            self._graph.add_edge("depends_on", node_id, dependency)
        return {"goal_id": node_id, "status": "PENDING"}

    def graph_complete_goal(
        self,
        *,
        goal_id: str,
        artifact_ids: list[str] | None = None,
    ) -> dict[str, Any]:
        node_id = _goal_id(goal_id)
        goal = self._graph.nodes.get(node_id)
        if goal is None or goal["kind"] != "GOAL":
            raise ValueError(f"unknown goal {node_id!r}")
        dependencies = self._goal_dependencies.get(node_id, [])
        if any(self._goal_status(value) != "COMPLETE" for value in dependencies):
            raise ValueError("goal dependencies are not complete")
        artifacts = list(artifact_ids or self._artifacts_for_target(node_id))
        for artifact_id in artifacts:
            if artifact_id not in self._graph.artifacts:
                raise ValueError(f"unknown graph artifact {artifact_id!r}")
            self._graph.add_edge("satisfies", artifact_id, node_id)
        goal["data"]["status"] = "COMPLETE"
        goal["data"]["artifact_count"] = len(artifacts)
        return {"goal_id": node_id, "status": "COMPLETE", "artifact_ids": artifacts}

    def graph_frontier(self) -> dict[str, Any]:
        self._inspection_count += 1
        return self._frontier(include_reusable=True)

    def graph_trace(self, *, node_id: str) -> dict[str, Any]:
        if node_id not in self._graph.nodes:
            raise ValueError(f"unknown graph node {node_id!r}")
        self._inspection_count += 1
        edges = [
            edge
            for edge in self._graph.edges
            if edge["source"] == node_id or edge["target"] == node_id
        ][-12:]
        neighbors = sorted({
            endpoint
            for edge in edges
            for endpoint in (edge["source"], edge["target"])
            if endpoint != node_id
        })
        return {
            "node": copy.deepcopy(self._graph.nodes[node_id]),
            "neighbors": [copy.deepcopy(self._graph.nodes[value]) for value in neighbors],
            "edges": copy.deepcopy(edges),
        }

    def graph_load_artifact(self, *, artifact_id: str) -> Any:
        self._artifact_loads += 1
        value = self._graph.load_artifact(artifact_id)
        if self._current_action is not None:
            self._graph.add_edge("consumes", artifact_id, self._current_action["id"])
        return value

    def initial_observation(self) -> str:
        return self._render("GRAPH_ASSESSMENT ", self._control_view())

    def prepare_program_action(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        action = str(payload.get("action", "CONTINUE")).upper()
        requested_target = str(payload.get("target", "task"))[:500]
        target_valid = requested_target in self._graph.nodes
        target = requested_target if target_valid else "task"
        if not target_valid:
            self._invalid_action_targets += 1
        inputs = tuple(
            str(value)
            for value in payload.get("input_artifacts", ())
            if str(value) in self._graph.artifacts
        )
        intent_id = f"intent:{sum(self._actions.values()) + 1}"
        self._graph.add_node(
            intent_id,
            "ACTION_INTENT",
            {
                "action": action,
                "expected_change": str(payload.get("expected_change", ""))[:500],
            },
        )
        self._graph.add_edge("targets", intent_id, target)
        for artifact_id in inputs:
            self._graph.add_edge("consumes", artifact_id, intent_id)
        self._actions[action] += 1
        self._current_target = target
        self._declared_inputs = inputs
        self._current_action = {
            "id": intent_id,
            "action": action,
            "target": target,
            "requested_target": requested_target,
            "target_valid": target_valid,
            "expected_change": str(payload.get("expected_change", ""))[:500],
            "before": self._snapshot(target),
            "inspection_count": self._inspection_count,
            "artifact_loads": self._artifact_loads,
        }
        self._pending_inspection_request = (
            copy.deepcopy(payload.get("inspection")) if action == "INSPECT" else None
        )
        self._current_inspection_result = None
        self._action_history.append(self._current_action)
        return dict(payload)

    def observe(self, trace: Any) -> str:
        self._observation_calls += 1
        block_id = self._execution.observe(trace)
        if self._current_action is not None:
            self._graph.add_edge("implemented_by", self._current_action["id"], block_id)
        effect = self._progress.observe(
            block_id,
            target=(self._current_action or {}).get("target", "task"),
        )
        failure = self._failure_view(trace, block_id)
        inspection_result = self._execute_pending_inspection()
        verification = self._verify_current_action(trace, effect)
        if self._current_action is not None:
            self._current_action["realized"] = bool(verification["realized"])
            self._current_action["effect"] = {
                key: effect.get(key)
                for key in (
                    "progressed",
                    "novel_artifacts",
                    "equivalent_artifacts",
                    "state_changes",
                    "stagnant_streak",
                )
            }
            if inspection_result is not None:
                self._current_action["inspection"] = {
                    key: copy.deepcopy(inspection_result.get(key))
                    for key in ("status", "request", "returned", "result_sha256", "error")
                    if key in inspection_result
                }
        if verification["realized"]:
            self._realized_actions += 1
        else:
            self._missed_actions += 1
        payload = {
            "schema_version": 1,
            "control_contract": "generic-goal-graph-v1",
            "block": self._observation_calls,
            "declared_action": _visible_action(self._current_action),
            "action_verification": verification,
            "actual_effect": effect,
            "next_action_contract": self._control_view(effect, failure=failure),
        }
        if inspection_result is not None:
            payload["inspection_result"] = inspection_result
        return self._render("GRAPH_DELTA ", payload)

    def finish(self, *, answered: bool) -> None:
        if not answered:
            return
        self._graph.nodes["task"]["data"]["status"] = "COMPLETE"
        self._actions["ANSWER"] += 1
        answer_id = f"intent:{sum(self._actions.values())}"
        self._graph.add_node(answer_id, "ACTION_INTENT", {"action": "ANSWER"})
        self._graph.add_edge("targets", answer_id, "task")

    def telemetry(self) -> dict[str, Any]:
        graph = self._graph.telemetry()
        graph.update(
            {
                "artifact_reuse_hits": self._runtime.reuse_hits + self._artifact_loads,
                "interface_calls": {
                    "graph_declare_goal": len(self._goals),
                    "graph_load_artifact": self._artifact_loads,
                },
                "requirement_states": dict(
                    Counter(self._goal_status(value) for value in self._goals)
                ),
                "task_graph_initialized": bool(self._goals),
            }
        )
        return {
            "mode": "generic_online",
            "control_contract": "generic-goal-graph-v1",
            "observation_calls": self._observation_calls,
            "action_distribution": dict(self._actions),
            "action_history": [dict(value) for value in self._action_history],
            "goal_states": dict(Counter(self._goal_status(value) for value in self._goals)),
            "tool_reuse_hits": self._runtime.reuse_hits,
            "artifact_loads": self._artifact_loads,
            "invalid_action_targets": self._invalid_action_targets,
            "realized_graph_deltas": self._realized_actions,
            "missed_graph_deltas": self._missed_actions,
            "aligned_actions": self._realized_actions,
            "misaligned_actions": self._missed_actions,
            "inspection": {
                "declared": self._actions["INSPECT"],
                "well_formed": self._inspection_well_formed,
                "query_attempts": self._inspection_query_attempts,
                "succeeded": self._inspection_succeeded,
                "failed": self._inspection_failed,
                "responses_emitted": self._inspection_responses_emitted,
                "results_returned": self._inspection_results_returned,
            },
            "research_graph": graph,
        }

    def graph_artifact(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "nodes": [
                copy.deepcopy(self._graph.nodes[node_id])
                for node_id in self._graph.node_order
            ],
            "edges": copy.deepcopy(self._graph.edges),
            "artifacts": copy.deepcopy(self._graph.artifacts),
        }

    def _tool_wrapper(
        self,
        name: str,
        function: Callable[..., Any],
    ) -> Callable[..., Any]:
        @wraps(function)
        def invoke(**kwargs: Any) -> Any:
            result = self._runtime.invoke(
                name,
                graph_target=self._current_target,
                consumes=self._declared_inputs,
                **kwargs,
            )
            return result.value

        invoke.__name__ = name
        return invoke

    def _frontier(self, *, include_reusable: bool | None = None) -> dict[str, Any]:
        if include_reusable is None:
            include_reusable = self._expose_graph_api
        ready = []
        blocked = []
        for goal_id in self._goals:
            if self._goal_status(goal_id) == "COMPLETE":
                continue
            item = {
                "id": goal_id,
                "description": self._graph.nodes[goal_id]["data"]["description"],
                "depends_on": self._goal_dependencies.get(goal_id, []),
            }
            if all(
                self._goal_status(value) == "COMPLETE"
                for value in self._goal_dependencies.get(goal_id, [])
            ):
                ready.append(item)
            else:
                blocked.append(item)
        consumed = {
            edge["source"]
            for edge in self._graph.edges
            if edge["type"] in {"consumes", "reuses", "satisfies"}
            and edge["source"] in self._graph.artifacts
        }
        return {
            "ready_goals": ready,
            "blocked_goals": blocked,
            "complete_goals": [
                value for value in self._goals if self._goal_status(value) == "COMPLETE"
            ],
            "reusable_artifacts": [
                artifact_id
                for artifact_id in self._graph.artifacts
                if artifact_id not in consumed
            ][-4:]
            if include_reusable
            else [],
        }

    def _control_view(
        self,
        effect: Mapping[str, Any] | None = None,
        *,
        failure: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        frontier = self._frontier()
        ready = [item["id"] for item in frontier["ready_goals"]]
        active_target = str((effect or {}).get("target") or self._current_target or "task")
        opportunities = [
            {
                "action": "CONTINUE",
                "target": target,
                "reason": (
                    "continue the task using a new dependency effect"
                    if target == "task"
                    else "all declared dependencies for this goal are complete"
                ),
            }
            for target in ("task", *ready)
        ]
        if failure:
            opportunities.insert(
                0,
                {
                    "action": "PATCH",
                    "target": active_target,
                    "reason": "correct the failed program or dependency assumption, then re-execute it",
                },
            )
        if int((effect or {}).get("stagnant_streak", 0)) >= 2:
            opportunities.insert(
                1 if failure else 0,
                {
                    "action": "REPLAN",
                    "target": active_target,
                    "reason": "recent actions only reproduced existing effects; choose a dependency path not listed as exhausted",
                },
            )
        if self._host_inspection_enabled and (self._goals or self._graph.artifacts):
            opportunities.append(
                {
                    "action": "INSPECT",
                    "target": ready[0] if ready else "task",
                    "reason": "inspect the current dependency frontier or a prior artifact",
                }
            )
        opportunities.append(
            {
                "action": "ANSWER",
                "target": "task",
                "reason": "finish only when the available artifacts satisfy the task",
            }
        )
        return {
            "available_actions": list(dict.fromkeys(item["action"] for item in opportunities)),
            "action_opportunities": opportunities,
            "frontier": frontier,
            "last_effect": dict(effect or {}),
            "last_failure": dict(failure or {}),
            "branch_frontier": self._branch_frontier(active_target, effect),
        }

    def _failure_view(self, trace: Any, block_id: str) -> dict[str, Any] | None:
        if bool(getattr(trace, "success", False)):
            return None
        return {
            "id": f"failure:{block_id}",
            "target": (self._current_action or {}).get("target", "task"),
            "failed_action": (self._current_action or {}).get("action"),
            "error_type": getattr(trace, "error_type", None),
            "error_message": str(getattr(trace, "error_message", ""))[:500],
        }

    def _branch_frontier(
        self,
        target: str,
        effect: Mapping[str, Any] | None,
    ) -> dict[str, Any]:
        if int((effect or {}).get("stagnant_streak", 0)) < 1:
            return {}
        productive: list[dict[str, str]] = []
        exhausted: list[dict[str, str]] = []
        seen: set[tuple[str, str]] = set()
        for action in reversed(self._action_history):
            if action.get("target") != target or "realized" not in action:
                continue
            expected = str(action.get("expected_change", "")).strip()
            if not expected:
                continue
            outcome = "productive" if action.get("realized") else "exhausted"
            key = (expected.casefold(), outcome)
            if key in seen:
                continue
            seen.add(key)
            item = {"expected_change": expected, "outcome": outcome}
            (productive if outcome == "productive" else exhausted).append(item)
            if len(productive) >= 3 and len(exhausted) >= 3:
                break
        return {
            "target": target,
            "productive_paths": list(reversed(productive[:3])),
            "exhausted_paths": list(reversed(exhausted[:3])),
            "shared_artifacts": self._frontier()["reusable_artifacts"],
        }

    def _verify_current_action(
        self,
        trace: Any,
        effect: Mapping[str, Any],
    ) -> dict[str, Any]:
        action = self._current_action
        if action is None:
            return {"realized": False, "reason": "no declared action"}
        after = self._snapshot(action["target"])
        selected = action["action"]
        if selected == "PATCH":
            realized = bool(getattr(trace, "success", False))
        elif selected == "INSPECT":
            realized = bool(
                self._current_inspection_result
                and self._current_inspection_result.get("status") == "ok"
                and self._current_inspection_result.get("returned")
            )
        else:
            realized = bool(effect.get("progressed")) or any(
                after[key] > action["before"][key]
                for key in ("complete_goals",)
            )
        return {
            "action": selected,
            "target": action["target"],
            "realized": realized,
            "before": action["before"],
            "after": after,
        }

    def _snapshot(self, target: str) -> dict[str, int]:
        return {
            "artifacts": len(self._graph.artifacts),
            "state_versions": sum(
                node["kind"] == "STATE_VERSION" for node in self._graph.nodes.values()
            ),
            "complete_goals": sum(
                self._goal_status(value) == "COMPLETE" for value in self._goals
            ),
            "target_complete": int(
                target in self._graph.nodes
                and self._graph.nodes[target]["kind"] == "GOAL"
                and self._goal_status(target) == "COMPLETE"
            ),
        }

    def _goal_status(self, goal_id: str) -> str:
        return str(self._graph.nodes[goal_id]["data"].get("status", "PENDING"))

    def _artifacts_for_target(self, target: str) -> set[str]:
        action_ids = {
            edge["source"]
            for edge in self._graph.edges
            if edge["type"] == "targets" and edge["target"] == target
            and self._graph.nodes.get(edge["source"], {}).get("kind") == "TOOL_ACTION"
        }
        return {
            edge["target"]
            for edge in self._graph.edges
            if edge["type"] == "produces" and edge["source"] in action_ids
        }

    def _render(self, prefix: str, payload: dict[str, Any]) -> str:
        def encode(value: Mapping[str, Any]) -> str:
            return prefix + json.dumps(value, ensure_ascii=False, separators=(",", ":"))

        rendered = encode(payload)
        if len(rendered) <= self._max_observation_chars:
            return rendered
        compact = dict(payload)
        contract = compact.get("next_action_contract")
        if isinstance(contract, dict):
            contract = dict(contract)
            contract["action_opportunities"] = contract.get("action_opportunities", [])[:4]
            frontier = dict(contract.get("frontier") or {})
            for key in frontier:
                if isinstance(frontier[key], list):
                    frontier[key] = frontier[key][-4:]
            contract["frontier"] = frontier
            compact["next_action_contract"] = contract
        rendered = encode(compact)
        if len(rendered) <= self._max_observation_chars:
            return rendered
        digest = hashlib.sha256(
            json.dumps(payload, ensure_ascii=False, sort_keys=True, default=repr).encode("utf-8")
        ).hexdigest()
        minimal = {
            "schema_version": payload.get("schema_version"),
            "control_contract": payload.get("control_contract"),
            "block": payload.get("block"),
            "declared_action": _action_receipt(payload.get("declared_action")),
            "action_verification": _verification_receipt(
                payload.get("action_verification")
            ),
            "inspection_result": _inspection_receipt(payload.get("inspection_result")),
            "next_action_contract": {
                "available_actions": (payload.get("next_action_contract") or {}).get(
                    "available_actions", []
                ),
                "last_failure": _failure_receipt(
                    (payload.get("next_action_contract") or {}).get("last_failure")
                ),
            },
            "truncated": True,
            "full_payload_sha256": digest,
        }
        rendered = encode(minimal)
        if len(rendered) <= self._max_observation_chars:
            return rendered
        emergency = {
            "schema_version": payload.get("schema_version"),
            "block": payload.get("block"),
            "action_verification": _verification_receipt(
                payload.get("action_verification")
            ),
            "inspection_result": _inspection_receipt(payload.get("inspection_result")),
            "truncated": True,
            "full_payload_sha256": digest,
        }
        rendered = encode(emergency)
        if len(rendered) <= self._max_observation_chars:
            return rendered
        fallback = {
            "inspection_result": _inspection_receipt(payload.get("inspection_result")),
            "truncated": True,
        }
        rendered = encode(fallback)
        if len(rendered) <= self._max_observation_chars:
            return rendered
        return encode({"truncated": True})

    def _execute_pending_inspection(self) -> dict[str, Any] | None:
        action = self._current_action
        if action is None or action.get("action") != "INSPECT":
            return None
        request = self._pending_inspection_request
        if not self._host_inspection_enabled:
            return self._inspection_error(request, "host graph inspection is disabled")
        if not isinstance(request, Mapping):
            return self._inspection_error(request, "inspection must be an object")
        view = str(request.get("view", ""))[:64]
        normalized: dict[str, str] = {"view": view}
        if view not in {"frontier", "trace"}:
            return self._inspection_error(normalized, f"unsupported inspection view {view!r}")
        if view == "trace":
            node_id = str(request.get("node_id", ""))[:500]
            normalized["node_id"] = node_id
            if not node_id:
                return self._inspection_error(normalized, "trace inspection requires node_id")
        self._inspection_well_formed += 1
        self._inspection_query_attempts += 1
        try:
            result = (
                self.graph_frontier()
                if view == "frontier"
                else self.graph_trace(node_id=normalized["node_id"])
            )
        except ValueError as exc:
            return self._inspection_error(normalized, str(exc))
        self._inspection_succeeded += 1
        full_json = json.dumps(
            result, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=repr
        )
        max_chars = max(512, min(1_600, self._max_observation_chars // 2))
        bounded = _bounded_inspection_value(result)
        bounded_json = json.dumps(
            bounded, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=repr
        )
        truncated = len(full_json) > max_chars or len(bounded_json) > max_chars
        if len(bounded_json) > max_chars:
            bounded = _inspection_summary(result)
        response = {
            "status": "ok",
            "request": normalized,
            "result": bounded,
            "result_summary": _inspection_summary(result),
            "truncated": truncated,
            "result_sha256": hashlib.sha256(full_json.encode("utf-8")).hexdigest(),
            "returned": True,
            "response_emitted": True,
        }
        self._inspection_responses_emitted += 1
        self._inspection_results_returned += 1
        self._current_inspection_result = response
        return response

    def _inspection_error(self, request: Any, message: str) -> dict[str, Any]:
        self._inspection_failed += 1
        self._inspection_responses_emitted += 1
        response = {
            "status": "error",
            "request": _bounded_inspection_value(request),
            "error": str(message)[:500],
            "returned": False,
            "response_emitted": True,
        }
        self._current_inspection_result = response
        return response


def _bounded_inspection_value(value: Any, *, depth: int = 0) -> Any:
    if depth >= 4:
        return str(value)[:240]
    if isinstance(value, Mapping):
        return {
            str(key): _bounded_inspection_value(item, depth=depth + 1)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))[:24]
        }
    if isinstance(value, (list, tuple)):
        return [_bounded_inspection_value(item, depth=depth + 1) for item in value[:12]]
    if isinstance(value, str):
        return value[:240]
    return copy.deepcopy(value)


def _inspection_summary(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        return {"type": type(value).__name__}
    node = value.get("node")
    node_summary = None
    if isinstance(node, Mapping):
        node_summary = {key: copy.deepcopy(node.get(key)) for key in ("id", "kind")}
    return {
        "top_level_keys": sorted(str(key) for key in value),
        "node": node_summary,
        "list_counts": {
            str(key): len(item)
            for key, item in value.items()
            if isinstance(item, (list, tuple))
        },
    }


def _inspection_receipt(value: Any) -> dict[str, Any] | None:
    if not isinstance(value, Mapping):
        return None
    return {
        key: copy.deepcopy(value.get(key))
        for key in (
            "status",
            "request",
            "result_summary",
            "truncated",
            "result_sha256",
            "returned",
            "response_emitted",
            "error",
        )
        if key in value
    }


def _action_receipt(value: Any) -> dict[str, Any] | None:
    if not isinstance(value, Mapping):
        return None
    return {
        key: _bounded_inspection_value(value.get(key))
        for key in ("action", "target", "target_valid", "expected_change")
        if key in value
    }


def _verification_receipt(value: Any) -> dict[str, Any] | None:
    if not isinstance(value, Mapping):
        return None
    return {
        key: _bounded_inspection_value(value.get(key))
        for key in ("action", "target", "realized", "reason")
        if key in value
    }


def _failure_receipt(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        return {}
    return {
        key: _bounded_inspection_value(value.get(key))
        for key in ("id", "target", "failed_action", "error_type", "error_message")
        if key in value
    }


def _goal_id(value: str) -> str:
    key = str(value).strip()
    if not key:
        raise ValueError("goal id must not be empty")
    return key if key.startswith("goal:") else f"goal:{key}"


def _visible_action(value: Mapping[str, Any] | None) -> dict[str, Any] | None:
    if value is None:
        return None
    return {
        key: value.get(key)
        for key in ("action", "target", "requested_target", "target_valid")
    }

