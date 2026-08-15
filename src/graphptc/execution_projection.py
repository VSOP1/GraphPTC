from __future__ import annotations

from typing import Any, Mapping

from .episode_graph import EpisodeGraph


class PTCExecutionProjection:
    """Project block execution and persistent Python state into EpisodeGraph."""

    def __init__(self, graph: EpisodeGraph) -> None:
        self._graph = graph
        self._block_count = 0
        self._known_tool_actions: set[str] = set()
        self._state_versions: dict[str, int] = {}
        self._current_states: dict[str, str] = {}

    def observe(self, trace: Any) -> str:
        self._block_count += 1
        block_id = f"block:{self._block_count}"
        runtime_trace = getattr(trace, "runtime_trace", {}) or {}
        program_analysis = getattr(trace, "program_analysis", {}) or {}
        self._graph.add_node(
            block_id,
            "BLOCK",
            {
                "index": self._block_count,
                "success": bool(getattr(trace, "success", False)),
                "error_type": getattr(trace, "error_type", None),
                "runtime_calls": int(getattr(trace, "runtime_calls", 0)),
                "program_analysis": _compact_program_analysis(program_analysis),
            },
        )
        self._graph.add_edge("contains", "task", block_id)

        code_artifact = f"artifact:block:{self._block_count}:code"
        self._graph.put_artifact(
            code_artifact,
            str(getattr(trace, "code", "")),
            kind="program",
            data={"block": block_id},
        )
        self._graph.add_edge("executes", code_artifact, block_id)
        stdout_artifact = f"artifact:block:{self._block_count}:stdout"
        self._graph.put_artifact(
            stdout_artifact,
            str(getattr(trace, "stdout", "")),
            kind="observation",
            data={"block": block_id},
        )
        self._graph.add_edge("produces", block_id, stdout_artifact)

        tool_actions = {
            node_id
            for node_id, node in self._graph.nodes.items()
            if node["kind"] == "TOOL_ACTION"
        }
        for action_id in sorted(tool_actions - self._known_tool_actions):
            self._graph.add_edge("executes", block_id, action_id)
        self._known_tool_actions = tool_actions

        if isinstance(runtime_trace, Mapping):
            self._project_state(block_id, runtime_trace)
        if not bool(getattr(trace, "success", False)):
            failure_id = f"failure:block:{self._block_count}"
            self._graph.add_node(
                failure_id,
                "FAILURE",
                {
                    "error_type": getattr(trace, "error_type", None),
                    "error_message": str(getattr(trace, "error_message", ""))[:500],
                    "location": runtime_trace.get("error_location")
                    if isinstance(runtime_trace, Mapping)
                    else None,
                },
            )
            self._graph.add_edge("fails_at", block_id, failure_id)
        return block_id

    def _project_state(self, block_id: str, runtime_trace: Mapping[str, Any]) -> None:
        state_before = runtime_trace.get("state_before")
        state_after = runtime_trace.get("state_after")
        before = state_before if isinstance(state_before, Mapping) else {}
        after = state_after if isinstance(state_after, Mapping) else {}
        loaded = {str(value) for value in runtime_trace.get("loaded_names", ())}
        stored = {str(value) for value in runtime_trace.get("stored_names", ())}
        block_actions = set(self._graph.successors(block_id, edge_type="executes"))
        block_artifacts = {
            edge["target"]
            for edge in self._graph.edges
            if edge["type"] == "produces" and edge["source"] in block_actions
        }

        for name in sorted(loaded):
            state_id = self._current_states.get(name)
            if state_id is None and name in before:
                state_id = self._new_state(name, before[name], version=0)
            if state_id is not None:
                self._graph.add_edge("reads", state_id, block_id)

        for name in sorted(stored):
            if name not in after:
                continue
            previous = self._current_states.get(name)
            version = self._state_versions.get(name, 0) + 1
            state_id = self._new_state(name, after[name], version=version)
            self._graph.add_edge("writes", block_id, state_id)
            for artifact_id in block_artifacts:
                self._graph.add_edge("derives", artifact_id, state_id)
            if previous is not None:
                self._graph.add_edge("supersedes", previous, state_id)

    def _new_state(self, name: str, value_type: Any, *, version: int) -> str:
        state_id = f"state:python:{name}:{version}"
        self._graph.add_node(
            state_id,
            "STATE_VERSION",
            {"name": name, "version": version, "value_type": str(value_type)},
        )
        self._state_versions[name] = version
        self._current_states[name] = state_id
        return state_id


def _compact_program_analysis(value: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: value[key]
        for key in (
            "tool_call_count",
            "transform_count",
            "control_dependency_count",
            "syntax_error",
        )
        if key in value
    }

