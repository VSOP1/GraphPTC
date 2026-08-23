from __future__ import annotations

import copy
import json
from dataclasses import dataclass
from typing import Any, Callable, Mapping, Protocol

from .episode_graph import EpisodeGraph


class GraphController(Protocol):
    """Hooks required to attach graph control to an arbitrary PTC agent."""

    def runtime_functions(self) -> tuple[Callable[..., Any], ...]: ...

    def initial_observation(self) -> str: ...

    def prepare_program_action(self, payload: Mapping[str, Any]) -> dict[str, Any]: ...

    def observe(self, trace: Any) -> str: ...


@dataclass(frozen=True)
class GraphAgentHooks:
    runtime_functions: tuple[Callable[..., Any], ...]
    adaptation_initial_observation: Callable[[], str]
    ptc_call_metadata_callback: Callable[[Mapping[str, Any]], dict[str, Any]]
    block_observation_factory: Callable[[Any], str]
    message_projection_callback: Callable[[list[dict[str, Any]]], None] | None = None

    @classmethod
    def from_controller(cls, controller: GraphController) -> "GraphAgentHooks":
        projector = getattr(controller, "project_messages", None)
        return cls(
            runtime_functions=controller.runtime_functions(),
            adaptation_initial_observation=controller.initial_observation,
            ptc_call_metadata_callback=controller.prepare_program_action,
            block_observation_factory=controller.observe,
            message_projection_callback=projector if callable(projector) else None,
        )

    def agent_kwargs(self) -> dict[str, Any]:
        return {
            "runtime_functions": self.runtime_functions,
            "adaptation_initial_observation": self.adaptation_initial_observation,
            "ptc_call_metadata_callback": self.ptc_call_metadata_callback,
            "block_observation_factory": self.block_observation_factory,
            "message_projection_callback": self.message_projection_callback,
        }


class GraphContextProjector:
    """Keep model context aligned with the active dependency subgraph.

    Full tool observations remain graph artifacts. Old inactive observations are
    represented by reloadable references, while recent and dependency-relevant
    observations remain inline. The operation only changes model context; it
    never deletes episode data.
    """

    def __init__(
        self,
        graph: EpisodeGraph,
        *,
        active_nodes: Callable[[], tuple[str, ...]] = tuple,
        retain_recent: int = 6,
        retain_relevant: int = 4,
        relevance_depth: int = 6,
    ) -> None:
        self._graph = graph
        self._active_nodes = active_nodes
        self._retain_recent = retain_recent
        self._retain_relevant = retain_relevant
        self._relevance_depth = relevance_depth
        self._message_ids: dict[str, str] = {}
        self._contents: dict[str, Any] = {}

    def project(self, messages: list[dict[str, Any]], *, block_id: str) -> None:
        message = next(
            (
                item
                for item in reversed(messages)
                if item.get("role") == "tool"
                and str(item.get("tool_call_id", "")) not in self._message_ids.values()
            ),
            None,
        )
        if message is None:
            return
        call_id = str(message.get("tool_call_id", ""))
        self._message_ids[block_id] = call_id
        self._contents[block_id] = copy.deepcopy(message.get("content", ""))

        ordered = [
            node_id
            for node_id in self._graph.node_order
            if node_id in self._message_ids
        ]
        keep = set(ordered[-self._retain_recent :])
        relevant = self._relevant_blocks()
        keep.update([node_id for node_id in ordered if node_id in relevant][-self._retain_relevant :])
        by_call_id = {
            str(item.get("tool_call_id", "")): item
            for item in messages
            if item.get("role") == "tool"
        }
        for known_block, known_call_id in self._message_ids.items():
            item = by_call_id.get(known_call_id)
            if item is None:
                continue
            if known_block in keep:
                item["content"] = copy.deepcopy(self._contents[known_block])
                continue
            artifact_id = f"artifact:{known_block}:stdout"
            item["content"] = "GRAPH_MEMORY_REF " + json.dumps(
                {
                    "block_id": known_block,
                    "artifact_id": artifact_id,
                    "status": "archived_from_context",
                    "reload": f"graph_load_artifact(artifact_id='{artifact_id}')",
                },
                ensure_ascii=False,
                separators=(",", ":"),
            )

    def _relevant_blocks(self) -> set[str]:
        frontier = {
            node_id
            for node_id in self._active_nodes()
            if node_id in self._graph.nodes and node_id != "task"
        }
        seen = set(frontier)
        blocks: set[str] = set()
        for _ in range(self._relevance_depth):
            if not frontier:
                break
            next_frontier: set[str] = set()
            for node_id in frontier:
                if self._graph.nodes[node_id]["kind"] == "BLOCK":
                    blocks.add(node_id)
                for edge in self._graph.edges:
                    if edge["source"] == node_id:
                        neighbor = edge["target"]
                    elif edge["target"] == node_id:
                        neighbor = edge["source"]
                    else:
                        continue
                    if neighbor in seen or self._graph.nodes[neighbor]["kind"] == "TASK":
                        continue
                    seen.add(neighbor)
                    next_frontier.add(neighbor)
            frontier = next_frontier
        blocks.update(
            node_id
            for node_id in frontier
            if self._graph.nodes[node_id]["kind"] == "BLOCK"
        )
        return blocks


class GraphProgressTracker:
    """Detect target-local loops from graph effects rather than tool names."""

    def __init__(self, graph: EpisodeGraph) -> None:
        self._graph = graph
        self._streaks: dict[str, int] = {}

    def observe(self, block_id: str, *, target: str) -> dict[str, Any]:
        actions = set(self._graph.successors(block_id, edge_type="executes"))
        produced = {
            edge["target"]
            for edge in self._graph.edges
            if edge["type"] == "produces" and edge["source"] in actions
        }
        equivalent = {
            edge["target"]
            for edge in self._graph.edges
            if edge["type"] == "equivalent_to" and edge["target"] in produced
        }
        state_changes = {
            edge["target"]
            for edge in self._graph.edges
            if edge["type"] in {"mutates", "writes"} and edge["source"] in actions | {block_id}
        }
        novel = produced - equivalent
        progressed = bool(novel or state_changes)
        self._streaks[target] = 0 if progressed else self._streaks.get(target, 0) + 1
        return {
            "target": target,
            "progressed": progressed,
            "novel_artifacts": sorted(novel)[:4],
            "equivalent_artifacts": sorted(equivalent)[:4],
            "state_changes": sorted(state_changes)[:4],
            "stagnant_streak": self._streaks[target],
        }


class PlanRevisionLedger:
    """Persist explicit plan changes and their supersession chain."""

    def __init__(self, graph: EpisodeGraph) -> None:
        self._graph = graph
        self._count = 0
        self._latest_by_target: dict[str, str] = {}

    def record(self, *, target: str, approach: str, action_id: str) -> str:
        text = str(approach).strip()
        if not text:
            raise ValueError("REPLAN requires a non-empty plan_revision.approach")
        self._count += 1
        plan_id = f"plan_revision:{self._count}"
        self._graph.add_node(plan_id, "PLAN_REVISION", {"approach": text[:500]})
        if target in self._graph.nodes:
            self._graph.add_edge("targets", plan_id, target)
        previous = self._latest_by_target.get(target)
        if previous is not None:
            self._graph.add_edge("supersedes", previous, plan_id)
        if action_id in self._graph.nodes:
            self._graph.add_edge("declares", action_id, plan_id)
        self._latest_by_target[target] = plan_id
        return plan_id

    def latest(self, target: str) -> dict[str, Any] | None:
        plan_id = self._latest_by_target.get(target)
        if plan_id is None:
            return None
        return copy.deepcopy(self._graph.nodes[plan_id])


def extend_ptc_spec_with_graph_control(
    base_spec: dict[str, Any],
    *,
    extra_properties: Mapping[str, Any] | None = None,
    include_target: bool = True,
    include_input_artifacts: bool = True,
    include_inspection: bool = False,
    action_description: str = "The explicit graph-control action implemented by this block.",
    target_description: str = "An existing goal, action, state, or artifact node id.",
    expected_change_description: str = (
        "The observable graph or task-state change intended by this block."
    ),
) -> dict[str, Any]:
    """Add domain-neutral action intent to any PTC tool schema."""
    spec = copy.deepcopy(base_spec)
    original = spec["function"]["parameters"]["properties"]
    actions = ["CONTINUE", "PATCH", "REPLAN"]
    if include_inspection:
        actions.insert(1, "INSPECT")
    controls: dict[str, Any] = {
        "action": {
            "type": "string",
            "enum": actions,
            "description": action_description,
        },
        "expected_change": {
            "type": "string",
            "minLength": 1,
            "description": expected_change_description,
        },
    }
    if include_target:
        controls["target"] = {
            "type": "string",
            "minLength": 1,
            "description": target_description,
        }
    if include_input_artifacts:
        controls["input_artifacts"] = {
            "type": "array",
            "items": {"type": "string", "minLength": 1},
            "description": "Existing graph artifacts intentionally consumed by this block.",
        }
    if include_inspection:
        controls["inspection"] = {
            "type": "object",
            "properties": {
                "view": {
                    "type": "string",
                    "enum": ["frontier", "trace"],
                    "description": "The bounded read-only graph view to return after this block.",
                },
                "node_id": {
                    "type": "string",
                    "minLength": 1,
                    "description": "Existing graph node id required by the trace view.",
                },
            },
            "required": ["view"],
            "additionalProperties": False,
            "description": (
                "Host-side graph query executed after this block is projected. "
                "Use only when action is INSPECT; its result is available next turn."
            ),
        }
    properties: dict[str, Any] = {}
    for name, value in original.items():
        properties[name] = value
        if name == "code":
            properties.update(controls)
    properties.update(copy.deepcopy(dict(extra_properties or {})))
    spec["function"]["parameters"]["properties"] = properties
    required = list(spec["function"]["parameters"].get("required", ()))
    required_controls = ["code", "action"]
    if include_target:
        required_controls.append("target")
    required_controls.append("expected_change")
    for name in required_controls:
        if name not in required:
            required.append(name)
    spec["function"]["parameters"]["required"] = required
    return spec
