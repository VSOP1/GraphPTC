from __future__ import annotations

import ast
import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable


EVENT_TYPES = {
    "episode.started",
    "block.started",
    "tool.called",
    "block.finished",
    "repair.finished",
    "episode.finished",
}


@dataclass(frozen=True)
class GraphArtifact:
    id: str
    kind: str
    sha256: str
    chars: int
    value: Any


@dataclass(frozen=True)
class GraphNode:
    id: str
    type: str
    episode_id: str
    block_id: str | None
    data: dict[str, Any] = field(default_factory=dict)
    artifact_ids: tuple[str, ...] = ()


@dataclass(frozen=True)
class GraphEdge:
    id: str
    type: str
    source: str
    target: str
    data: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class DependencyGraph:
    episode_id: str
    task_id: str
    source_event_count: int
    source_events_sha256: str
    nodes: tuple[GraphNode, ...]
    edges: tuple[GraphEdge, ...]
    artifacts: tuple[GraphArtifact, ...]

    @property
    def node_ids(self) -> frozenset[str]:
        return frozenset(node.id for node in self.nodes)

    def node(self, node_id: str) -> GraphNode:
        for node in self.nodes:
            if node.id == node_id:
                return node
        raise KeyError(node_id)

    def artifact(self, artifact_id: str) -> GraphArtifact:
        for artifact in self.artifacts:
            if artifact.id == artifact_id:
                return artifact
        raise KeyError(artifact_id)

    def artifacts_for_node(self, node_id: str) -> tuple[GraphArtifact, ...]:
        node = self.node(node_id)
        return tuple(self.artifact(artifact_id) for artifact_id in node.artifact_ids)

    def state_dependencies(self, block_node_id: str) -> tuple[GraphNode, ...]:
        block = self.node(block_node_id)
        if block.type != "BLOCK":
            raise ValueError("state_dependencies requires a BLOCK node")
        return self.predecessors(block_node_id, edge_type="STATE")

    def predecessors(
        self, node_id: str, *, edge_type: str | None = None
    ) -> tuple[GraphNode, ...]:
        self.node(node_id)
        return tuple(
            self.node(edge.source)
            for edge in self.edges
            if edge.target == node_id
            and (edge_type is None or edge.type == edge_type)
        )

    def successors(
        self, node_id: str, *, edge_type: str | None = None
    ) -> tuple[GraphNode, ...]:
        self.node(node_id)
        return tuple(
            self.node(edge.target)
            for edge in self.edges
            if edge.source == node_id
            and (edge_type is None or edge.type == edge_type)
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": 3,
            "episode_id": self.episode_id,
            "task_id": self.task_id,
            "source_event_count": self.source_event_count,
            "source_events_sha256": self.source_events_sha256,
            "nodes": [asdict(node) for node in self.nodes],
            "edges": [asdict(edge) for edge in self.edges],
            "artifacts": [asdict(artifact) for artifact in self.artifacts],
        }


def load_execution_events(path: str | Path) -> tuple[dict[str, Any], ...]:
    events: list[dict[str, Any]] = []
    source = Path(path)
    with source.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                event = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid event JSON on line {line_number}: {exc.msg}"
                ) from exc
            if not isinstance(event, dict):
                raise ValueError(f"Event on line {line_number} must be an object")
            events.append(event)
    _validate_event_stream(events)
    return tuple(events)


def write_dependency_graph_report(
    events_path: str | Path,
    output_path: str | Path,
) -> dict[str, Any]:
    graphs = build_dependency_graphs(load_execution_events(events_path))
    report = {
        "schema_version": 3,
        "graph_count": len(graphs),
        "graphs": [graph.to_dict() for graph in graphs],
    }
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return report


def write_dependency_graph_bundle(
    events_path: str | Path,
    manifest_path: str | Path,
    artifacts_path: str | Path | None = None,
) -> dict[str, Any]:
    graphs = build_dependency_graphs(load_execution_events(events_path))
    destination = Path(manifest_path)
    artifact_destination = (
        Path(artifacts_path)
        if artifacts_path is not None
        else destination.with_name(f"{destination.stem}.artifacts.jsonl")
    )
    if artifact_destination.resolve().parent != destination.resolve().parent:
        raise ValueError("Artifact sidecar must be in the manifest directory")

    artifact_rows = [
        {
            "schema_version": 1,
            "episode_id": graph.episode_id,
            "artifact": asdict(artifact),
        }
        for graph in graphs
        for artifact in graph.artifacts
    ]
    artifact_bytes = (
        "".join(
            json.dumps(
                row,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            )
            + "\n"
            for row in artifact_rows
        )
    ).encode("utf-8")
    report = {
        "schema_version": 4,
        "graph_schema_version": 3,
        "graph_count": len(graphs),
        "artifact_store": {
            "schema_version": 1,
            "path": artifact_destination.name,
            "sha256": hashlib.sha256(artifact_bytes).hexdigest(),
            "count": len(artifact_rows),
        },
        "graphs": [_graph_manifest(graph) for graph in graphs],
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    artifact_destination.write_bytes(artifact_bytes)
    destination.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return report


def load_dependency_graph_report(
    manifest_path: str | Path,
) -> tuple[DependencyGraph, ...]:
    source = Path(manifest_path)
    try:
        report = json.loads(source.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid graph report JSON: {exc.msg}") from exc
    if not isinstance(report, dict):
        raise ValueError("Graph report must be an object")
    schema_version = report.get("schema_version")
    if schema_version == 3:
        return _load_embedded_graphs(report)
    if schema_version == 4:
        return _load_graph_bundle(source, report)
    raise ValueError(f"Unsupported graph report schema_version: {schema_version}")


def _graph_manifest(graph: DependencyGraph) -> dict[str, Any]:
    value = graph.to_dict()
    value["artifacts"] = [
        {
            "id": artifact.id,
            "kind": artifact.kind,
            "sha256": artifact.sha256,
            "chars": artifact.chars,
        }
        for artifact in graph.artifacts
    ]
    return value


def _load_embedded_graphs(report: dict[str, Any]) -> tuple[DependencyGraph, ...]:
    graph_values = report.get("graphs")
    if not isinstance(graph_values, list):
        raise ValueError("Graph report requires a graphs list")
    if report.get("graph_count") != len(graph_values):
        raise ValueError("Graph report graph_count does not match graphs")
    return tuple(_dependency_graph_from_dict(value) for value in graph_values)


def _load_graph_bundle(
    manifest_path: Path,
    report: dict[str, Any],
) -> tuple[DependencyGraph, ...]:
    if report.get("graph_schema_version") != 3:
        raise ValueError("Graph bundle has unsupported graph_schema_version")
    store = report.get("artifact_store")
    if not isinstance(store, dict) or store.get("schema_version") != 1:
        raise ValueError("Graph bundle has invalid artifact_store metadata")
    relative_path = store.get("path")
    if (
        not isinstance(relative_path, str)
        or not relative_path
        or Path(relative_path).name != relative_path
    ):
        raise ValueError("Graph bundle artifact path must be a sibling filename")
    artifact_path = manifest_path.parent / relative_path
    artifact_bytes = artifact_path.read_bytes()
    if hashlib.sha256(artifact_bytes).hexdigest() != store.get("sha256"):
        raise ValueError("Graph bundle artifact store hash mismatch")

    artifacts: dict[str, tuple[str, GraphArtifact]] = {}
    for line_number, line in enumerate(artifact_bytes.decode("utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"Invalid artifact JSON on line {line_number}: {exc.msg}"
            ) from exc
        if not isinstance(row, dict) or row.get("schema_version") != 1:
            raise ValueError(f"Invalid artifact record on line {line_number}")
        episode_id = row.get("episode_id")
        if not isinstance(episode_id, str) or not episode_id:
            raise ValueError(f"Artifact record {line_number} requires episode_id")
        artifact = _artifact_from_dict(row.get("artifact"))
        if artifact.id in artifacts:
            raise ValueError(f"Duplicate artifact ID: {artifact.id}")
        artifacts[artifact.id] = (episode_id, artifact)
    if store.get("count") != len(artifacts):
        raise ValueError("Graph bundle artifact count does not match sidecar")

    graph_values = report.get("graphs")
    if not isinstance(graph_values, list):
        raise ValueError("Graph bundle requires a graphs list")
    if report.get("graph_count") != len(graph_values):
        raise ValueError("Graph bundle graph_count does not match graphs")
    used_artifacts: set[str] = set()
    graphs: list[DependencyGraph] = []
    for graph_value in graph_values:
        if not isinstance(graph_value, dict):
            raise ValueError("Graph bundle entries must be objects")
        episode_id = graph_value.get("episode_id")
        metadata_values = graph_value.get("artifacts")
        if not isinstance(metadata_values, list):
            raise ValueError("Graph bundle entry requires artifact metadata")
        full_artifacts: list[dict[str, Any]] = []
        for metadata in metadata_values:
            if not isinstance(metadata, dict):
                raise ValueError("Artifact metadata must be an object")
            artifact_id = metadata.get("id")
            stored = artifacts.get(artifact_id)
            if stored is None or stored[0] != episode_id:
                raise ValueError(f"Missing artifact sidecar value: {artifact_id}")
            artifact = stored[1]
            expected_metadata = {
                "id": artifact.id,
                "kind": artifact.kind,
                "sha256": artifact.sha256,
                "chars": artifact.chars,
            }
            if metadata != expected_metadata:
                raise ValueError(f"Artifact metadata mismatch: {artifact_id}")
            full_artifacts.append(asdict(artifact))
            used_artifacts.add(artifact.id)
        full_graph_value = dict(graph_value)
        full_graph_value["artifacts"] = full_artifacts
        graphs.append(_dependency_graph_from_dict(full_graph_value))
    if used_artifacts != set(artifacts):
        raise ValueError("Graph bundle contains unreferenced artifact values")
    return tuple(graphs)


def _dependency_graph_from_dict(value: Any) -> DependencyGraph:
    if not isinstance(value, dict) or value.get("schema_version") != 3:
        raise ValueError("Graph entry has unsupported schema_version")
    try:
        graph = DependencyGraph(
            episode_id=value["episode_id"],
            task_id=value["task_id"],
            source_event_count=value["source_event_count"],
            source_events_sha256=value["source_events_sha256"],
            nodes=tuple(
                GraphNode(
                    id=node["id"],
                    type=node["type"],
                    episode_id=node["episode_id"],
                    block_id=node["block_id"],
                    data=dict(node.get("data", {})),
                    artifact_ids=tuple(node.get("artifact_ids", ())),
                )
                for node in value["nodes"]
            ),
            edges=tuple(
                GraphEdge(
                    id=edge["id"],
                    type=edge["type"],
                    source=edge["source"],
                    target=edge["target"],
                    data=dict(edge.get("data", {})),
                )
                for edge in value["edges"]
            ),
            artifacts=tuple(
                _artifact_from_dict(artifact) for artifact in value["artifacts"]
            ),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"Malformed graph entry: {exc}") from exc
    _validate_graph(graph)
    return graph


def _artifact_from_dict(value: Any) -> GraphArtifact:
    if not isinstance(value, dict):
        raise ValueError("Artifact must be an object")
    try:
        return GraphArtifact(
            id=value["id"],
            kind=value["kind"],
            sha256=value["sha256"],
            chars=value["chars"],
            value=value["value"],
        )
    except KeyError as exc:
        raise ValueError(f"Artifact is missing {exc.args[0]}") from exc


def build_dependency_graphs(
    events: Iterable[dict[str, Any]],
) -> tuple[DependencyGraph, ...]:
    values = list(events)
    _validate_event_stream(values)
    grouped: dict[str, list[dict[str, Any]]] = {}
    for event in values:
        grouped.setdefault(str(event["episode_id"]), []).append(event)
    return tuple(build_dependency_graph(group) for group in grouped.values())


def build_dependency_graph(
    events: Iterable[dict[str, Any]],
) -> DependencyGraph:
    values = list(events)
    _validate_event_stream(values)
    episode_ids = {str(event["episode_id"]) for event in values}
    if len(episode_ids) != 1:
        raise ValueError("build_dependency_graph requires events from one episode")
    if not values:
        raise ValueError("build_dependency_graph requires at least one event")

    episode_id = next(iter(episode_ids))
    task_id = str(values[0]["task_id"])
    started = [event for event in values if event["type"] == "episode.started"]
    finished = [event for event in values if event["type"] == "episode.finished"]
    if len(started) != 1 or started[0] is not values[0]:
        raise ValueError("Episode must begin with exactly one episode.started event")
    if len(finished) != 1 or finished[0] is not values[-1]:
        raise ValueError("Episode must end with exactly one episode.finished event")

    blocks = _collect_blocks(values)
    declared_blocks = finished[0]["data"].get("ptc_blocks")
    if isinstance(declared_blocks, int) and declared_blocks != len(blocks):
        raise ValueError(
            "episode.finished ptc_blocks does not match completed block count"
        )

    nodes: list[GraphNode] = []
    edges: list[GraphEdge] = []
    artifacts: list[GraphArtifact] = []
    episode_node_id = f"episode:{episode_id}"
    nodes.append(
        GraphNode(
            id=episode_node_id,
            type="EPISODE",
            episode_id=episode_id,
            block_id=None,
            data={
                "started_sequence": started[0]["sequence"],
                "finished_sequence": finished[0]["sequence"],
                "task": started[0]["data"].get("task"),
                "status": finished[0]["data"].get("status"),
                "error": finished[0]["data"].get("error"),
            },
        )
    )

    edge_count = 0

    def add_edge(
        edge_type: str,
        source: str,
        target: str,
        data: dict[str, Any] | None = None,
    ) -> None:
        nonlocal edge_count
        edge_count += 1
        edges.append(
            GraphEdge(
                id=f"edge:{edge_count:06d}",
                type=edge_type,
                source=source,
                target=target,
                data=data or {},
            )
        )

    previous_block_node_id: str | None = None
    state_producers: dict[str, str] = {}
    for block in blocks:
        block_id = str(block["started"]["block_id"])
        block_node_id = f"block:{block_id}"
        code = str(block["started"]["data"].get("code", ""))
        tool_names = {
            str(event["data"].get("tool", "")) for event in block["tools"]
        }
        source_sites, transform_sites, state_source_sites, source_error = _program_sites(
            code,
            tool_names,
            block_id,
        )
        output_data = block["finished"]["data"]
        runtime_trace = output_data.get("runtime_trace", {})
        nodes.append(
            GraphNode(
                id=block_node_id,
                type="BLOCK",
                episode_id=episode_id,
                block_id=block_id,
                data={
                    "started_sequence": block["started"]["sequence"],
                    "finished_sequence": block["finished"]["sequence"],
                    "turn": block["started"]["data"].get("turn"),
                    "tool_call_id": block["started"]["data"].get("tool_call_id"),
                    "code": code,
                    "source_sites": source_sites,
                    "transform_sites": transform_sites,
                    "state_source_sites": state_source_sites,
                    "source_error": source_error,
                    **{
                        key: block["started"]["data"][key]
                        for key in (
                            "execution_version_id",
                            "source_block_id",
                            "program_version_id",
                        )
                        if key in block["started"]["data"]
                    },
                },
            )
        )
        add_edge("CONTAINS", episode_node_id, block_node_id)
        if previous_block_node_id is not None:
            add_edge("PRECEDES", previous_block_node_id, block_node_id)
        previous_block_node_id = block_node_id

        state_before = runtime_trace.get("state_before", {})
        loaded_names = set(runtime_trace.get("loaded_names", ()))
        if isinstance(state_before, dict):
            for name in sorted(loaded_names & set(state_before)):
                state_node_id = state_producers.get(name)
                if state_node_id is not None:
                    add_edge(
                        "STATE",
                        state_node_id,
                        block_node_id,
                        {
                            "name": name,
                            "evidence": "runtime_load_from_persistent_state",
                        },
                    )

        previous_tool_node_id: str | None = None
        stdout_sources: list[tuple[str, dict[str, Any]]] = []
        lineage_nodes: dict[str, list[str]] = {}
        for ordinal, tool_event in enumerate(block["tools"], 1):
            tool_node_id = f"tool:{block_id}:{ordinal}"
            tool_name = str(tool_event["data"].get("tool", ""))
            matching_sites = [
                site for site in source_sites if site["tool"] == tool_name
            ]
            call_site = tool_event["data"].get("call_site")
            runtime_matches = [
                site
                for site in matching_sites
                if isinstance(call_site, dict)
                and site["line"] == call_site.get("line")
                and site["column"] == call_site.get("column")
            ]
            if isinstance(call_site, dict):
                candidates = [site["id"] for site in runtime_matches]
                alignment = "runtime_exact" if len(candidates) == 1 else "runtime_unmatched"
            else:
                candidates = [site["id"] for site in matching_sites]
                alignment = (
                    "exact_single_candidate"
                    if len(candidates) == 1
                    else "ambiguous" if candidates else "unmatched"
                )
            artifact_ids: tuple[str, ...] = ()
            if tool_event["data"].get("success") is True and "result" in tool_event["data"]:
                artifact = _artifact(
                    f"artifact:{tool_node_id}:result",
                    "TOOL_RESULT",
                    tool_event["data"]["result"],
                )
                artifacts.append(artifact)
                artifact_ids = (artifact.id,)
            previous_control_node_id: str | None = None
            controls = (
                runtime_matches[0].get("controls", ())
                if alignment == "runtime_exact"
                else ()
            )
            for control_ordinal, control in enumerate(controls, 1):
                control_node_id = (
                    f"transform:{block_id}:{ordinal}:control:{control_ordinal}"
                )
                nodes.append(
                    GraphNode(
                        id=control_node_id,
                        type="TRANSFORM",
                        episode_id=episode_id,
                        block_id=block_id,
                        data={
                            **control,
                            "kind": "CONTROL",
                            "tool_event_sequence": tool_event["sequence"],
                            "source_site_id": runtime_matches[0]["id"],
                            "evidence": "runtime_call_in_static_control_region",
                        },
                    )
                )
                add_edge("CONTAINS", block_node_id, control_node_id)
                if previous_control_node_id is not None:
                    add_edge(
                        "CONTROL",
                        previous_control_node_id,
                        control_node_id,
                        {"evidence": "nested_static_control_region"},
                    )
                previous_control_node_id = control_node_id
            nodes.append(
                GraphNode(
                    id=tool_node_id,
                    type="TOOL",
                    episode_id=episode_id,
                    block_id=block_id,
                    data={
                        "event_sequence": tool_event["sequence"],
                        "tool": tool_name,
                        "arguments": tool_event["data"].get("arguments", {}),
                        "success": tool_event["data"].get("success"),
                        "error": tool_event["data"].get("error"),
                        "call_site": call_site,
                        "source_alignment": alignment,
                        "source_site_ids": candidates,
                        **{
                            key: tool_event["data"][key]
                            for key in (
                                "replay_action",
                                "source_block_id",
                                "source_tool_node_id",
                                "source_artifact_id",
                            )
                            if key in tool_event["data"]
                        },
                    },
                    artifact_ids=artifact_ids,
                )
            )
            add_edge("CONTAINS", block_node_id, tool_node_id)
            if previous_control_node_id is not None:
                add_edge(
                    "CONTROL",
                    previous_control_node_id,
                    tool_node_id,
                    {"evidence": "runtime_call_in_static_control_region"},
                )
            if previous_tool_node_id is not None:
                add_edge("PRECEDES", previous_tool_node_id, tool_node_id)
            previous_tool_node_id = tool_node_id
            if alignment == "runtime_exact" and tool_event["data"].get("success") is True:
                lineage_nodes.setdefault(runtime_matches[0]["id"], []).append(
                    tool_node_id
                )
            if (
                alignment == "runtime_exact"
                and tool_event["data"].get("success") is True
                and runtime_matches[0].get("feeds_stdout") is True
            ):
                stdout_sources.append((tool_node_id, runtime_matches[0]))

        executed_spans = {
            (
                span.get("line"),
                span.get("column"),
                span.get("end_line"),
                span.get("end_column"),
            )
            for span in runtime_trace.get("executed_spans", ())
            if isinstance(span, dict)
        }
        pending_transforms = list(transform_sites)
        while pending_transforms:
            progressed = False
            for transform_site in list(pending_transforms):
                span = (
                    transform_site["line"],
                    transform_site["column"],
                    transform_site["end_line"],
                    transform_site["end_column"],
                )
                input_site_ids = transform_site["input_site_ids"]
                if span not in executed_spans or not input_site_ids:
                    pending_transforms.remove(transform_site)
                    continue
                if not all(site_id in lineage_nodes for site_id in input_site_ids):
                    continue
                transform_node_id = (
                    f"transform:{block_id}:data:{transform_site['ordinal']}"
                )
                nodes.append(
                    GraphNode(
                        id=transform_node_id,
                        type="TRANSFORM",
                        episode_id=episode_id,
                        block_id=block_id,
                        data={
                            **transform_site,
                            "kind": "DATA",
                            "evidence": "runtime_exact_span",
                        },
                    )
                )
                add_edge("CONTAINS", block_node_id, transform_node_id)
                for input_site_id in input_site_ids:
                    for input_node_id in lineage_nodes[input_site_id]:
                        add_edge(
                            "DATA",
                            input_node_id,
                            transform_node_id,
                            {
                                "source_site_id": input_site_id,
                                "evidence": "static_lineage_with_runtime_span",
                            },
                        )
                lineage_nodes[transform_site["id"]] = [transform_node_id]
                if transform_site["feeds_stdout"]:
                    stdout_sources.append((transform_node_id, transform_site))
                pending_transforms.remove(transform_site)
                progressed = True
            if not progressed:
                break

        output_node_id = f"output:{block_id}"
        stdout_artifact = _artifact(
            f"artifact:{output_node_id}:stdout",
            "BLOCK_STDOUT",
            output_data.get("stdout", ""),
        )
        artifacts.append(stdout_artifact)
        nodes.append(
            GraphNode(
                id=output_node_id,
                type="OUTPUT",
                episode_id=episode_id,
                block_id=block_id,
                data={
                    "scope": "block",
                    "event_sequence": block["finished"]["sequence"],
                    "success": output_data.get("success"),
                    "stdout_chars": output_data.get("stdout_chars"),
                    "stdout_truncated": output_data.get("stdout_truncated"),
                    "error_type": output_data.get("error_type"),
                    "error_message": output_data.get("error_message"),
                    "error_location": runtime_trace.get("error_location"),
                },
                artifact_ids=(stdout_artifact.id,),
            )
        )
        add_edge("CONTAINS", block_node_id, output_node_id)
        add_edge("RESULT_OF", block_node_id, output_node_id)
        if previous_tool_node_id is not None:
            add_edge("PRECEDES", previous_tool_node_id, output_node_id)
        for tool_node_id, source_site in stdout_sources:
            add_edge(
                "DATA",
                tool_node_id,
                output_node_id,
                {
                    "source_site_id": source_site["id"],
                    "evidence": (
                        "static_transform_to_stdout"
                        if source_site.get("site_type") == "TRANSFORM"
                        else "static_value_flow_to_stdout"
                    ),
                },
            )

        state_after = runtime_trace.get("state_after", {})
        stored_names = set(runtime_trace.get("stored_names", ()))
        if isinstance(state_after, dict):
            for name in sorted(stored_names & set(state_after)):
                state_node_id = f"state:{block_id}:{name}"
                nodes.append(
                    GraphNode(
                        id=state_node_id,
                        type="STATE",
                        episode_id=episode_id,
                        block_id=block_id,
                        data={
                            "name": name,
                            "value_type": state_after[name],
                            "evidence": "runtime_store_to_persistent_state",
                        },
                    )
                )
                add_edge("CONTAINS", block_node_id, state_node_id)
                for source_site_id in state_source_sites.get(name, ()):
                    for source_node_id in lineage_nodes.get(source_site_id, ()):
                        add_edge(
                            "DATA",
                            source_node_id,
                            state_node_id,
                            {
                                "name": name,
                                "source_site_id": source_site_id,
                                "evidence": "static_value_flow_to_state",
                            },
                        )
                state_producers[name] = state_node_id

    final_output_node_id = f"output:{episode_id}:final"
    answer_artifact = _artifact(
        f"artifact:{final_output_node_id}:answer",
        "FINAL_ANSWER",
        finished[0]["data"].get("answer", ""),
    )
    artifacts.append(answer_artifact)
    nodes.append(
        GraphNode(
            id=final_output_node_id,
            type="OUTPUT",
            episode_id=episode_id,
            block_id=None,
            data={
                "scope": "episode",
                "event_sequence": finished[0]["sequence"],
                "status": finished[0]["data"].get("status"),
                "error": finished[0]["data"].get("error"),
            },
            artifact_ids=(answer_artifact.id,),
        )
    )
    add_edge("CONTAINS", episode_node_id, final_output_node_id)
    add_edge("RESULT_OF", episode_node_id, final_output_node_id)

    graph = DependencyGraph(
        episode_id=episode_id,
        task_id=task_id,
        source_event_count=len(values),
        source_events_sha256=_sha256(values),
        nodes=tuple(nodes),
        edges=tuple(edges),
        artifacts=tuple(artifacts),
    )
    _validate_graph(graph)
    return graph


def _validate_event_stream(events: list[dict[str, Any]]) -> None:
    next_sequence: dict[str, int] = {}
    task_ids: dict[str, str] = {}
    for index, event in enumerate(events, 1):
        if not isinstance(event, dict):
            raise ValueError(f"Event {index} must be an object")
        if event.get("schema_version") != 1:
            raise ValueError(f"Event {index} has unsupported schema_version")
        event_type = event.get("type")
        if event_type not in EVENT_TYPES:
            raise ValueError(f"Event {index} has unsupported type: {event_type}")
        episode_id = event.get("episode_id")
        task_id = event.get("task_id")
        if not isinstance(episode_id, str) or not episode_id:
            raise ValueError(f"Event {index} requires a non-empty episode_id")
        if not isinstance(task_id, str) or not task_id:
            raise ValueError(f"Event {index} requires a non-empty task_id")
        if episode_id in task_ids and task_ids[episode_id] != task_id:
            raise ValueError(f"Episode {episode_id} has inconsistent task_id")
        task_ids[episode_id] = task_id
        sequence = event.get("sequence")
        expected = next_sequence.get(episode_id, 1)
        if sequence != expected:
            raise ValueError(
                f"Episode {episode_id} sequence must be {expected}, got {sequence}"
            )
        next_sequence[episode_id] = expected + 1
        if not isinstance(event.get("data"), dict):
            raise ValueError(f"Event {index} data must be an object")
        block_id = event.get("block_id")
        requires_block = event_type in {
            "block.started",
            "tool.called",
            "block.finished",
            "repair.finished",
        }
        if requires_block and (not isinstance(block_id, str) or not block_id):
            raise ValueError(f"Event {index} requires a non-empty block_id")
        if not requires_block and block_id is not None:
            raise ValueError(f"Event {index} must not have a block_id")


def _collect_blocks(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    blocks: list[dict[str, Any]] = []
    active: dict[str, Any] | None = None
    seen: set[str] = set()
    for event in events[1:-1]:
        event_type = event["type"]
        block_id = str(event.get("block_id", ""))
        if event_type == "block.started":
            if active is not None:
                raise ValueError("A block started before the active block finished")
            if block_id in seen:
                raise ValueError(f"Duplicate block_id: {block_id}")
            active = {"started": event, "tools": [], "finished": None}
            seen.add(block_id)
        elif event_type == "tool.called":
            if active is None or active["started"]["block_id"] != block_id:
                raise ValueError(f"Tool event is outside its active block: {block_id}")
            active["tools"].append(event)
        elif event_type == "block.finished":
            if active is None or active["started"]["block_id"] != block_id:
                raise ValueError(f"Block finished without matching start: {block_id}")
            active["finished"] = event
            blocks.append(active)
            active = None
        elif event_type == "repair.finished":
            if active is not None or block_id not in seen:
                raise ValueError(
                    f"Repair event is outside its completed block: {block_id}"
                )
        else:
            raise ValueError(f"Unexpected episode event inside lifecycle: {event_type}")
    if active is not None:
        raise ValueError("Episode ended with an unfinished block")
    return blocks


def _program_sites(
    code: str,
    tool_names: set[str],
    block_id: str,
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    dict[str, list[str]],
    str | None,
]:
    try:
        tree = ast.parse(code)
    except SyntaxError as exc:
        return [], [], {}, f"{exc.msg} (line {exc.lineno})"
    collector = _ControlContextCollector(tool_names)
    collector.visit(tree)
    calls = list(collector.calls)
    calls.sort(key=lambda node: (node.lineno, node.col_offset))
    transforms = [
        node for node in ast.walk(tree) if _transform_type(node) is not None
    ]
    transforms.sort(
        key=lambda node: (
            node.lineno,
            node.col_offset,
            -(node.end_lineno or node.lineno),
            -(node.end_col_offset or node.col_offset),
        )
    )
    analyzer = _ValueFlowAnalyzer(set(calls), set(transforms))
    analyzer.process(tree.body)
    tool_site_ids = {
        node: f"site:{block_id}:{index}" for index, node in enumerate(calls, 1)
    }
    transform_site_ids = {
        node: f"transform-site:{block_id}:{index}"
        for index, node in enumerate(transforms, 1)
    }
    dependency_site_ids = {**tool_site_ids, **transform_site_ids}
    sites = [
        {
            "id": tool_site_ids[node],
            "tool": node.func.id,
            "line": node.lineno,
            "column": node.col_offset,
            "end_line": node.end_lineno,
            "end_column": node.end_col_offset,
            "feeds_stdout": node in analyzer.stdout_dependencies,
            "controls": collector.controls_by_call[node],
        }
        for node in calls
    ]
    transform_sites = [
        {
            "id": transform_site_ids[node],
            "site_type": "TRANSFORM",
            "ordinal": index,
            "transform_type": _transform_type(node),
            "line": node.lineno,
            "column": node.col_offset,
            "end_line": node.end_lineno,
            "end_column": node.end_col_offset,
            "input_site_ids": sorted(
                dependency_site_ids[dependency]
                for dependency in analyzer.transform_inputs.get(node, ())
                if dependency in dependency_site_ids
            ),
            "feeds_stdout": node in analyzer.stdout_dependencies,
        }
        for index, node in enumerate(transforms, 1)
    ]
    state_source_sites = {
        name: sorted(
            dependency_site_ids[dependency]
            for dependency in dependencies
            if dependency in dependency_site_ids
        )
        for name, dependencies in sorted(analyzer.environment.items())
        if any(dependency in dependency_site_ids for dependency in dependencies)
    }
    return sites, transform_sites, state_source_sites, None


def _transform_type(node: ast.AST) -> str | None:
    if isinstance(node, (ast.ListComp, ast.DictComp, ast.GeneratorExp)) and any(
        generator.ifs for generator in node.generators
    ):
        return "FILTER"
    if isinstance(node, ast.SetComp):
        return "DEDUP"
    if not isinstance(node, ast.Call):
        return None
    if isinstance(node.func, ast.Name):
        if node.func.id == "filter":
            return "FILTER"
        if node.func.id == "set":
            return "DEDUP"
        if node.func.id in {"all", "any", "len", "max", "min", "sum"}:
            return "AGGREGATE"
    if (
        isinstance(node.func, ast.Attribute)
        and node.func.attr == "fromkeys"
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "dict"
    ):
        return "DEDUP"
    return None


class _ControlContextCollector(ast.NodeVisitor):
    def __init__(self, tool_names: set[str]) -> None:
        self.tool_names = tool_names
        self.calls: list[ast.Call] = []
        self.controls_by_call: dict[ast.Call, list[dict[str, Any]]] = {}
        self._controls: list[dict[str, Any]] = []

    def visit_Call(self, node: ast.Call) -> None:
        if isinstance(node.func, ast.Name) and node.func.id in self.tool_names:
            self.calls.append(node)
            self.controls_by_call[node] = [dict(item) for item in self._controls]
        self.generic_visit(node)

    def visit_If(self, node: ast.If) -> None:
        self.visit(node.test)
        self._visit_controlled(node, "body", node.body)
        self._visit_controlled(node, "orelse", node.orelse)

    def visit_For(self, node: ast.For) -> None:
        self.visit(node.target)
        self.visit(node.iter)
        self._visit_controlled(node, "body", node.body)
        self._visit_controlled(node, "orelse", node.orelse)

    visit_AsyncFor = visit_For

    def visit_While(self, node: ast.While) -> None:
        self.visit(node.test)
        self._visit_controlled(node, "body", node.body)
        self._visit_controlled(node, "orelse", node.orelse)

    def visit_Try(self, node: ast.Try) -> None:
        self._visit_statements(node.body)
        for index, handler in enumerate(node.handlers):
            details: dict[str, Any] = {"handler_index": index}
            if handler.type is not None:
                details["exception_type"] = ast.unparse(handler.type)
            self._visit_controlled(node, "handler", handler.body, details)
        self._visit_controlled(node, "orelse", node.orelse)
        self._visit_statements(node.finalbody)

    visit_TryStar = visit_Try

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self._visit_function(node)

    visit_AsyncFunctionDef = visit_FunctionDef

    def visit_Lambda(self, node: ast.Lambda) -> None:
        controls = self._controls
        self._controls = []
        try:
            self.visit(node.body)
        finally:
            self._controls = controls

    def _visit_function(self, node: ast.FunctionDef | ast.AsyncFunctionDef) -> None:
        for decorator in node.decorator_list:
            self.visit(decorator)
        for default in [*node.args.defaults, *node.args.kw_defaults]:
            if default is not None:
                self.visit(default)
        controls = self._controls
        self._controls = []
        try:
            self._visit_statements(node.body)
        finally:
            self._controls = controls

    def _visit_controlled(
        self,
        node: ast.AST,
        branch: str,
        statements: list[ast.stmt],
        details: dict[str, Any] | None = None,
    ) -> None:
        if not statements:
            return
        context = {
            "control_type": type(node).__name__,
            "branch": branch,
            "line": node.lineno,
            "column": node.col_offset,
            "end_line": node.end_lineno,
            "end_column": node.end_col_offset,
            **(details or {}),
        }
        self._controls.append(context)
        try:
            self._visit_statements(statements)
        finally:
            self._controls.pop()

    def _visit_statements(self, statements: list[ast.stmt]) -> None:
        for statement in statements:
            self.visit(statement)


class _ValueFlowAnalyzer:
    _MUTATING_METHODS = {"add", "append", "extend", "insert", "setdefault", "update"}

    def __init__(
        self,
        tool_calls: set[ast.Call],
        transform_nodes: set[ast.AST] | None = None,
    ) -> None:
        self.tool_calls = tool_calls
        self.transform_nodes = transform_nodes or set()
        self.transform_inputs: dict[ast.AST, set[ast.AST]] = {}
        self.environment: dict[str, set[ast.AST]] = {}
        self.stdout_dependencies: set[ast.AST] = set()

    def process(self, statements: list[ast.stmt]) -> None:
        for statement in statements:
            self._statement(statement)

    def _statement(self, statement: ast.stmt) -> None:
        if isinstance(statement, ast.Assign):
            dependencies = self._dependencies(statement.value)
            for target in statement.targets:
                self._assign(target, dependencies)
            return
        if isinstance(statement, ast.AnnAssign):
            dependencies = (
                self._dependencies(statement.value)
                if statement.value is not None
                else set()
            )
            self._assign(statement.target, dependencies)
            return
        if isinstance(statement, ast.AugAssign):
            dependencies = self._dependencies(statement.target) | self._dependencies(
                statement.value
            )
            self._assign(statement.target, dependencies)
            return
        if isinstance(statement, ast.Expr) and isinstance(statement.value, ast.Call):
            call = statement.value
            if isinstance(call.func, ast.Name) and call.func.id == "print":
                self.stdout_dependencies.update(
                    dependency
                    for value in [*call.args, *(item.value for item in call.keywords)]
                    for dependency in self._dependencies(value)
                )
                return
            if (
                isinstance(call.func, ast.Attribute)
                and call.func.attr in self._MUTATING_METHODS
            ):
                dependencies = set().union(
                    *(self._dependencies(value) for value in call.args),
                    *(self._dependencies(item.value) for item in call.keywords),
                )
                self._mutate(call.func.value, dependencies)
            return
        if isinstance(statement, (ast.For, ast.AsyncFor)):
            self._assign(statement.target, self._dependencies(statement.iter))
            self.process(statement.body)
            self.process(statement.orelse)
            return
        if isinstance(statement, ast.If):
            self._merge_branches(statement.body, statement.orelse)
            return
        if isinstance(statement, ast.While):
            before = self._copy_environment()
            self.process(statement.body)
            body = self._copy_environment()
            self.environment = _merge_environments(before, body)
            self.process(statement.orelse)
            return
        if isinstance(statement, ast.Try):
            branches = [statement.body, *(handler.body for handler in statement.handlers)]
            self._merge_many(branches)
            self.process(statement.orelse)
            self.process(statement.finalbody)
            return
        if isinstance(statement, (ast.With, ast.AsyncWith)):
            self.process(statement.body)
            return
        if isinstance(statement, ast.Delete):
            for target in statement.targets:
                if isinstance(target, ast.Name):
                    self.environment.pop(target.id, None)

    def _dependencies(self, node: ast.AST | None) -> set[ast.AST]:
        if node is None:
            return set()
        if node in self.tool_calls:
            return {node}
        if node in self.transform_nodes:
            inputs = set().union(
                *(self._dependencies(child) for child in ast.iter_child_nodes(node))
            )
            self.transform_inputs.setdefault(node, set()).update(inputs)
            return {node}
        if isinstance(node, ast.Name):
            if isinstance(node.ctx, ast.Load):
                return set(self.environment.get(node.id, ()))
            return set()
        return set().union(
            *(self._dependencies(child) for child in ast.iter_child_nodes(node))
        )

    def _assign(self, target: ast.AST, dependencies: set[ast.AST]) -> None:
        if isinstance(target, ast.Name):
            self.environment[target.id] = set(dependencies)
        elif isinstance(target, (ast.Tuple, ast.List)):
            for item in target.elts:
                self._assign(item, dependencies)
        elif isinstance(target, ast.Starred):
            self._assign(target.value, dependencies)
        elif isinstance(target, (ast.Subscript, ast.Attribute)):
            self._mutate(target.value, dependencies)

    def _mutate(self, target: ast.AST, dependencies: set[ast.AST]) -> None:
        while isinstance(target, (ast.Subscript, ast.Attribute)):
            target = target.value
        if isinstance(target, ast.Name):
            self.environment.setdefault(target.id, set()).update(dependencies)

    def _merge_branches(
        self,
        body: list[ast.stmt],
        orelse: list[ast.stmt],
    ) -> None:
        self._merge_many([body, orelse])

    def _merge_many(self, branches: list[list[ast.stmt]]) -> None:
        before = self._copy_environment()
        outcomes: list[dict[str, set[ast.Call]]] = []
        for branch in branches:
            self.environment = {
                name: set(dependencies) for name, dependencies in before.items()
            }
            self.process(branch)
            outcomes.append(self._copy_environment())
        self.environment = _merge_environments(before, *outcomes)

    def _copy_environment(self) -> dict[str, set[ast.AST]]:
        return {
            name: set(dependencies)
            for name, dependencies in self.environment.items()
        }


def _merge_environments(
    *environments: dict[str, set[ast.AST]],
) -> dict[str, set[ast.AST]]:
    names = set().union(*(set(environment) for environment in environments))
    return {
        name: set().union(
            *(environment.get(name, set()) for environment in environments)
        )
        for name in names
    }


def _artifact(artifact_id: str, kind: str, value: Any) -> GraphArtifact:
    serialized = _canonical_json(value)
    chars = len(value) if isinstance(value, str) else len(serialized.decode("utf-8"))
    return GraphArtifact(
        id=artifact_id,
        kind=kind,
        sha256=hashlib.sha256(serialized).hexdigest(),
        chars=chars,
        value=value,
    )


def _validate_graph(graph: DependencyGraph) -> None:
    node_ids = [node.id for node in graph.nodes]
    if len(node_ids) != len(set(node_ids)):
        raise ValueError("Graph contains duplicate node IDs")
    artifact_ids = [artifact.id for artifact in graph.artifacts]
    if len(artifact_ids) != len(set(artifact_ids)):
        raise ValueError("Graph contains duplicate artifact IDs")
    known_nodes = set(node_ids)
    known_artifacts = set(artifact_ids)
    if any(
        edge.source not in known_nodes or edge.target not in known_nodes
        for edge in graph.edges
    ):
        raise ValueError("Graph contains a dangling edge")
    if any(
        artifact_id not in known_artifacts
        for node in graph.nodes
        for artifact_id in node.artifact_ids
    ):
        raise ValueError("Graph contains a dangling artifact reference")
    for artifact in graph.artifacts:
        expected = _artifact(artifact.id, artifact.kind, artifact.value)
        if artifact.sha256 != expected.sha256 or artifact.chars != expected.chars:
            raise ValueError(f"Graph artifact integrity check failed: {artifact.id}")


def _sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value)).hexdigest()


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
