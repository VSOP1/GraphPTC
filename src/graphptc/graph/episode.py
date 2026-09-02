from __future__ import annotations

import copy
from collections import Counter
from dataclasses import dataclass
from typing import Any, Mapping


@dataclass(frozen=True)
class EpisodeGraphDelta:
    nodes: tuple[dict[str, Any], ...]
    edges: tuple[dict[str, Any], ...]


class EpisodeGraph:
    """Mutable, domain-neutral dependency graph for one agent episode.

    The graph owns execution facts and artifact values. Domain projections may
    add their own node kinds and relations, but tool execution and projection
    always share this store.
    """

    def __init__(self, *, task: str = "") -> None:
        self.nodes: dict[str, dict[str, Any]] = {}
        self.edges: list[dict[str, Any]] = []
        self.node_order: list[str] = []
        self.artifacts: dict[str, Any] = {}
        self._node_cursor = 0
        self._edge_cursor = 0
        self._kind_counts: Counter[str] = Counter()
        self.add_node("task", "TASK", {"description": str(task)[:1_000]})

    def add_node(
        self,
        node_id: str,
        kind: str,
        data: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        existing = self.nodes.get(node_id)
        if existing is None:
            node = {"id": node_id, "kind": kind, "data": dict(data or {})}
            self.nodes[node_id] = node
            self.node_order.append(node_id)
            self._kind_counts[kind] += 1
            return node
        if existing["kind"] != kind:
            raise ValueError(f"graph node {node_id!r} already has another kind")
        existing["data"].update(
            {key: value for key, value in (data or {}).items() if value is not None}
        )
        return existing

    def add_edge(
        self,
        edge_type: str,
        source: str,
        target: str,
        data: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        if source not in self.nodes or target not in self.nodes:
            raise ValueError(f"graph edge {source!r} -> {target!r} has an unknown endpoint")
        edge = {
            "type": edge_type,
            "source": source,
            "target": target,
            **({"data": dict(data)} if data else {}),
        }
        if edge not in self.edges:
            self.edges.append(edge)
        return edge

    def put_artifact(
        self,
        artifact_id: str,
        value: Any,
        *,
        kind: str = "tool_result",
        data: Mapping[str, Any] | None = None,
    ) -> str:
        self.artifacts[artifact_id] = copy.deepcopy(value)
        self.add_node(
            artifact_id,
            "ARTIFACT",
            {"artifact_kind": kind, **dict(data or {})},
        )
        return artifact_id

    def load_artifact(self, artifact_id: str) -> Any:
        try:
            value = self.artifacts[artifact_id]
        except KeyError as exc:
            raise ValueError(f"unknown graph artifact {artifact_id!r}") from exc
        return copy.deepcopy(value)

    def predecessors(self, node_id: str, *, edge_type: str | None = None) -> list[str]:
        if node_id not in self.nodes:
            raise ValueError(f"unknown graph node {node_id!r}")
        return [
            edge["source"]
            for edge in self.edges
            if edge["target"] == node_id
            and (edge_type is None or edge["type"] == edge_type)
        ]

    def successors(self, node_id: str, *, edge_type: str | None = None) -> list[str]:
        if node_id not in self.nodes:
            raise ValueError(f"unknown graph node {node_id!r}")
        return [
            edge["target"]
            for edge in self.edges
            if edge["source"] == node_id
            and (edge_type is None or edge["type"] == edge_type)
        ]

    def delta(self) -> EpisodeGraphDelta:
        node_ids = self.node_order[self._node_cursor :]
        edges = self.edges[self._edge_cursor :]
        self._node_cursor = len(self.node_order)
        self._edge_cursor = len(self.edges)
        return EpisodeGraphDelta(
            nodes=tuple(copy.deepcopy(self.nodes[node_id]) for node_id in node_ids),
            edges=tuple(copy.deepcopy(edges)),
        )

    def telemetry(self) -> dict[str, Any]:
        return {
            "node_count": len(self.nodes),
            "edge_count": len(self.edges),
            "node_kinds": dict(self._kind_counts),
            "artifact_count": len(self.artifacts),
        }

