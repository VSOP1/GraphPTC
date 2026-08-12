from __future__ import annotations

import json
from collections import Counter
from functools import wraps
from typing import Any, Callable, Mapping

from .episode_graph import EpisodeGraph
from .execution_projection import PTCExecutionProjection
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
    ) -> None:
        self._graph = EpisodeGraph(task=task)
        self._runtime = ToolGraphRuntime(self._graph)
        self._execution = PTCExecutionProjection(self._graph)
        self._max_observation_chars = max_observation_chars
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

    def runtime_functions(self) -> tuple[Callable[..., Any], ...]:
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
        return self._frontier()

    def graph_trace(self, *, node_id: str) -> dict[str, Any]:
        self._inspection_count += 1
        if node_id not in self._graph.nodes:
            raise ValueError(f"unknown graph node {node_id!r}")
        edges = [
            edge
            for edge in self._graph.edges
            if edge["source"] == node_id or edge["target"] == node_id
        ][-12:]
        neighbors = {
            endpoint
            for edge in edges
            for endpoint in (edge["source"], edge["target"])
            if endpoint != node_id
        }
        return {
            "node": self._graph.nodes[node_id],
            "neighbors": [self._graph.nodes[value] for value in neighbors],
            "edges": edges,
        }

    def graph_load_artifact(self, *, artifact_id: str) -> Any:
        self._artifact_loads += 1
        value = self._graph.load_artifact(artifact_id)
        reuse_id = f"reuse:{self._artifact_loads}"
        self._graph.add_node(reuse_id, "REUSE", {"artifact_id": artifact_id})
        self._graph.add_edge("reuses", artifact_id, reuse_id)
        if self._current_target in self._graph.nodes:
            self._graph.add_edge("contributes_to", reuse_id, self._current_target)
        return value

    def initial_observation(self) -> str:
        return self._render("GRAPH_ASSESSMENT ", self._control_view())

    def prepare_program_action(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        action = str(payload.get("action", "CONTINUE")).upper()
        target = str(payload.get("target", "task"))
        if target not in self._graph.nodes:
            target = "task"
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
            "before": self._snapshot(target),
            "inspection_count": self._inspection_count,
            "artifact_loads": self._artifact_loads,
        }
        return dict(payload)

    def observe(self, trace: Any) -> str:
        block_id = self._execution.observe(trace)
        if self._current_action is not None:
            self._graph.add_edge("implemented_by", self._current_action["id"], block_id)
        verification = self._verify_current_action(trace)
        delta = self._graph.delta()
        payload = {
            "schema_version": 1,
            "control_contract": "generic-goal-graph-v1",
            "action_verification": verification,
            "graph_delta": {
                "new_nodes": list(delta.nodes),
                "new_edges": list(delta.edges),
            },
            "next_action_contract": self._control_view(),
        }
        return self._render("GRAPH_DELTA ", payload)

    def finish(self, *, answered: bool) -> None:
        if answered:
            self._graph.nodes["task"]["data"]["status"] = "COMPLETE"

    def telemetry(self) -> dict[str, Any]:
        return {
            "mode": "generic_online",
            "action_distribution": dict(self._actions),
            "goal_states": dict(Counter(self._goal_status(value) for value in self._goals)),
            "tool_reuse_hits": self._runtime.reuse_hits,
            "artifact_loads": self._artifact_loads,
            "graph": self._graph.telemetry(),
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
                target=self._current_target,
                consumes=self._declared_inputs,
                **kwargs,
            )
            return result.value

        invoke.__name__ = name
        return invoke

    def _frontier(self) -> dict[str, Any]:
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
        return {
            "ready_goals": ready,
            "blocked_goals": blocked,
            "complete_goals": [
                value for value in self._goals if self._goal_status(value) == "COMPLETE"
            ],
            "reusable_artifacts": list(self._graph.artifacts)[-8:],
        }

    def _control_view(self) -> dict[str, Any]:
        frontier = self._frontier()
        ready = [item["id"] for item in frontier["ready_goals"]]
        if not self._goals:
            opportunities = [
                {
                    "action": "CONTINUE",
                    "target": "task",
                    "reason": "declare stable goals and execute the first required step",
                }
            ]
        elif ready:
            opportunities = [
                {
                    "action": "CONTINUE",
                    "target": goal_id,
                    "reason": "all declared dependencies are complete",
                }
                for goal_id in ready
            ]
        else:
            opportunities = []
        if frontier["reusable_artifacts"]:
            opportunities.append(
                {
                    "action": "REUSE_REPLAY",
                    "target": frontier["reusable_artifacts"][-1],
                    "reason": "a prior artifact can be loaded without repeating its tool action",
                }
            )
        if any(self._goal_status(value) != "COMPLETE" for value in self._goals):
            opportunities.append(
                {
                    "action": "INSPECT",
                    "target": ready[0] if ready else "task",
                    "reason": "inspect dependencies or prior artifacts before selecting work",
                }
            )
        if self._goals and all(self._goal_status(value) == "COMPLETE" for value in self._goals):
            opportunities.append(
                {"action": "ANSWER", "target": "task", "reason": "all goals are complete"}
            )
        return {
            "available_actions": list(dict.fromkeys(item["action"] for item in opportunities)),
            "action_opportunities": opportunities,
            "frontier": frontier,
        }

    def _verify_current_action(self, trace: Any) -> dict[str, Any]:
        action = self._current_action
        if action is None:
            return {"realized": False, "reason": "no declared action"}
        after = self._snapshot(action["target"])
        selected = action["action"]
        if selected == "PATCH":
            realized = bool(getattr(trace, "success", False))
        elif selected == "INSPECT":
            realized = self._inspection_count > action["inspection_count"]
        elif selected == "REUSE_REPLAY":
            realized = self._artifact_loads > action["artifact_loads"]
        else:
            realized = any(
                after[key] > action["before"][key]
                for key in ("artifacts", "state_versions", "complete_goals")
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
        rendered = prefix + json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
        if len(rendered) <= self._max_observation_chars:
            return rendered
        compact = dict(payload)
        if "graph_delta" in compact:
            compact["graph_delta"] = {
                "new_nodes": compact["graph_delta"]["new_nodes"][-4:],
                "new_edges": compact["graph_delta"]["new_edges"][-6:],
            }
        rendered = prefix + json.dumps(compact, ensure_ascii=False, separators=(",", ":"))
        if len(rendered) > self._max_observation_chars:
            compact.pop("graph_delta", None)
            rendered = prefix + json.dumps(compact, ensure_ascii=False, separators=(",", ":"))
        return rendered[: self._max_observation_chars]


def _goal_id(value: str) -> str:
    key = str(value).strip()
    if not key:
        raise ValueError("goal id must not be empty")
    return key if key.startswith("goal:") else f"goal:{key}"

