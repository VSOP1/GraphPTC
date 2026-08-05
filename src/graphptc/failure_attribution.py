from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .stage2_graph import (
    DependencyGraph,
    GraphArtifact,
    GraphEdge,
    GraphNode,
    load_dependency_graph_report,
)


CAUSAL_EDGE_TYPES = frozenset({"CONTROL", "DATA", "RESULT_OF", "STATE"})
_ERROR_PATTERN = re.compile(
    r"^([A-Za-z_][A-Za-z0-9_]*(?:Error|Exception))\s*:\s*(.*)$",
    re.DOTALL,
)
_BLOCK_CONTEXT_OMISSIONS = {
    "code",
    "source_sites",
    "transform_sites",
    "state_source_sites",
}


@dataclass(frozen=True)
class FailureAnchor:
    kind: str
    node_id: str
    episode_id: str
    block_id: str | None
    error_type: str
    message: str
    location: dict[str, int] | None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ContextNode:
    id: str
    type: str
    block_id: str | None
    data: dict[str, Any]
    artifact_ids: tuple[str, ...]


@dataclass(frozen=True)
class CodeRegion:
    block_id: str
    start_line: int
    end_line: int
    focus_lines: tuple[int, ...]
    code: str


@dataclass(frozen=True)
class ArtifactSummary:
    id: str
    kind: str
    sha256: str
    chars: int
    preview: str
    preview_truncated: bool


@dataclass(frozen=True)
class FailureContext:
    anchor: FailureAnchor
    nodes: tuple[ContextNode, ...]
    edges: tuple[GraphEdge, ...]
    code_regions: tuple[CodeRegion, ...]
    artifacts: tuple[ArtifactSummary, ...]
    truncated: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class FailureExpansionOptions:
    boundary_node_ids: tuple[str, ...]
    artifact_ids: tuple[str, ...]
    code_block_ids: tuple[str, ...]


@dataclass(frozen=True)
class ExpandedArtifact:
    id: str
    kind: str
    sha256: str
    chars: int
    value: Any


@dataclass(frozen=True)
class FullCodeBlock:
    block_id: str
    code: str


@dataclass(frozen=True)
class FailureContextExpansion:
    context: FailureContext
    artifacts: tuple[ExpandedArtifact, ...]
    code_blocks: tuple[FullCodeBlock, ...]
    options: FailureExpansionOptions

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def find_failure_anchors(graph: DependencyGraph) -> tuple[FailureAnchor, ...]:
    failed_tools = [
        node
        for node in graph.nodes
        if node.type == "TOOL" and node.data.get("success") is False
    ]
    anchors = tuple(_tool_anchor(node) for node in failed_tools)
    failed_tool_blocks = {node.block_id for node in failed_tools}

    runtime_anchors = tuple(
        _runtime_anchor(node)
        for node in graph.nodes
        if node.type == "OUTPUT"
        and node.data.get("scope") == "block"
        and node.data.get("success") is False
        and node.block_id not in failed_tool_blocks
    )
    episode_output = next(
        (
            node
            for node in graph.nodes
            if node.type == "OUTPUT"
            and node.data.get("scope") == "episode"
            and node.data.get("status") == "failed"
        ),
        None,
    )
    specific = (*anchors, *runtime_anchors)
    failed_blocks = {
        node.block_id
        for node in graph.nodes
        if node.type == "OUTPUT"
        and node.data.get("scope") == "block"
        and node.data.get("success") is False
    }
    has_unrecovered_specific = any(
        anchor.block_id in failed_blocks for anchor in specific
    )
    if episode_output is not None and (not specific or not has_unrecovered_specific):
        return (*specific, _episode_anchor(episode_output))
    return specific


def build_failure_contexts(
    graph: DependencyGraph,
    *,
    max_nodes: int = 64,
    code_radius: int = 2,
    preview_chars: int = 160,
) -> tuple[FailureContext, ...]:
    return tuple(
        build_failure_context(
            graph,
            anchor,
            max_nodes=max_nodes,
            code_radius=code_radius,
            preview_chars=preview_chars,
        )
        for anchor in find_failure_anchors(graph)
    )


def build_failure_context(
    graph: DependencyGraph,
    anchor: FailureAnchor,
    *,
    max_nodes: int = 64,
    code_radius: int = 2,
    preview_chars: int = 160,
) -> FailureContext:
    if max_nodes < 1:
        raise ValueError("max_nodes must be positive")
    if code_radius < 0:
        raise ValueError("code_radius must be non-negative")
    if preview_chars < 0:
        raise ValueError("preview_chars must be non-negative")
    graph.node(anchor.node_id)

    selected = {anchor.node_id}
    queue = [anchor.node_id]
    truncated = False
    while queue:
        target = queue.pop(0)
        for edge in graph.edges:
            if edge.target != target or edge.type not in CAUSAL_EDGE_TYPES:
                continue
            if edge.source in selected:
                continue
            if len(selected) >= max_nodes:
                truncated = True
                continue
            selected.add(edge.source)
            queue.append(edge.source)

    raw_nodes = tuple(node for node in graph.nodes if node.id in selected)
    edges = tuple(
        edge
        for edge in graph.edges
        if edge.type in CAUSAL_EDGE_TYPES
        and edge.source in selected
        and edge.target in selected
    )
    return FailureContext(
        anchor=anchor,
        nodes=tuple(_context_node(node) for node in raw_nodes),
        edges=edges,
        code_regions=_code_regions(graph, raw_nodes, anchor, code_radius),
        artifacts=_artifact_summaries(graph, raw_nodes, preview_chars),
        truncated=truncated,
    )


def failure_expansion_options(
    graph: DependencyGraph,
    context: FailureContext,
) -> FailureExpansionOptions:
    graph.node(context.anchor.node_id)
    selected = {node.id for node in context.nodes}
    boundary = {
        edge.source
        for edge in graph.edges
        if edge.type in CAUSAL_EDGE_TYPES
        and edge.target in selected
        and edge.source not in selected
    }
    artifact_ids = {artifact.id for artifact in context.artifacts}
    block_ids = {
        node.block_id for node in context.nodes if node.block_id is not None
    }
    if context.anchor.block_id is not None:
        block_ids.add(context.anchor.block_id)
    return FailureExpansionOptions(
        boundary_node_ids=tuple(
            node.id for node in graph.nodes if node.id in boundary
        ),
        artifact_ids=tuple(
            artifact.id for artifact in graph.artifacts if artifact.id in artifact_ids
        ),
        code_block_ids=tuple(
            node.block_id
            for node in graph.nodes
            if node.type == "BLOCK" and node.block_id in block_ids
        ),
    )


def expand_failure_context(
    graph: DependencyGraph,
    context: FailureContext,
    *,
    max_nodes: int | None = None,
    artifact_ids: tuple[str, ...] = (),
    code_block_ids: tuple[str, ...] = (),
    code_radius: int = 2,
    preview_chars: int = 160,
) -> FailureContextExpansion:
    node_budget = len(context.nodes) if max_nodes is None else max_nodes
    if node_budget < len(context.nodes):
        raise ValueError("max_nodes cannot shrink the current context")
    expanded_context = build_failure_context(
        graph,
        context.anchor,
        max_nodes=node_budget,
        code_radius=code_radius,
        preview_chars=preview_chars,
    )
    options = failure_expansion_options(graph, expanded_context)
    allowed_artifacts = set(options.artifact_ids)
    for artifact_id in artifact_ids:
        if artifact_id not in allowed_artifacts:
            raise ValueError(
                f"artifact is outside the expanded causal context: {artifact_id}"
            )
    allowed_blocks = set(options.code_block_ids)
    for block_id in code_block_ids:
        if block_id not in allowed_blocks:
            raise ValueError(
                f"code block is outside the expanded causal context: {block_id}"
            )

    requested_artifacts = set(artifact_ids)
    requested_blocks = set(code_block_ids)
    return FailureContextExpansion(
        context=expanded_context,
        artifacts=tuple(
            ExpandedArtifact(
                id=artifact.id,
                kind=artifact.kind,
                sha256=artifact.sha256,
                chars=artifact.chars,
                value=artifact.value,
            )
            for artifact in graph.artifacts
            if artifact.id in requested_artifacts
        ),
        code_blocks=tuple(
            FullCodeBlock(
                block_id=node.block_id,
                code=str(node.data.get("code", "")),
            )
            for node in graph.nodes
            if node.type == "BLOCK" and node.block_id in requested_blocks
        ),
        options=options,
    )


def write_failure_attribution_report(
    graph_path: str | Path,
    output_path: str | Path,
    *,
    max_nodes: int = 64,
    code_radius: int = 2,
    preview_chars: int = 160,
) -> dict[str, Any]:
    graphs = load_dependency_graph_report(graph_path)
    episodes = []
    failure_count = 0
    for graph in graphs:
        contexts = build_failure_contexts(
            graph,
            max_nodes=max_nodes,
            code_radius=code_radius,
            preview_chars=preview_chars,
        )
        if not contexts:
            continue
        failure_count += len(contexts)
        episodes.append(
            {
                "episode_id": graph.episode_id,
                "task_id": graph.task_id,
                "contexts": [context.to_dict() for context in contexts],
            }
        )
    report = {
        "schema_version": 1,
        "source_graph_count": len(graphs),
        "episode_count": len(episodes),
        "failure_count": failure_count,
        "episodes": episodes,
    }
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return report


def _tool_anchor(node: GraphNode) -> FailureAnchor:
    error_type, message = _error_parts(node.data.get("error"), "ToolError")
    return FailureAnchor(
        kind="TOOL_ERROR",
        node_id=node.id,
        episode_id=node.episode_id,
        block_id=node.block_id,
        error_type=error_type,
        message=message,
        location=_location(node.data.get("call_site")),
    )


def _runtime_anchor(node: GraphNode) -> FailureAnchor:
    error_type = str(node.data.get("error_type") or "ExecutionError")
    message = str(node.data.get("error_message") or "")
    return FailureAnchor(
        kind="RUNTIME_ERROR",
        node_id=node.id,
        episode_id=node.episode_id,
        block_id=node.block_id,
        error_type=error_type,
        message=message,
        location=_location(node.data.get("error_location")),
    )


def _episode_anchor(node: GraphNode) -> FailureAnchor:
    error_type, message = _error_parts(node.data.get("error"), "EpisodeError")
    return FailureAnchor(
        kind="EPISODE_ERROR",
        node_id=node.id,
        episode_id=node.episode_id,
        block_id=None,
        error_type=error_type,
        message=message,
        location=None,
    )


def _error_parts(value: Any, fallback: str) -> tuple[str, str]:
    text = str(value or "")
    match = _ERROR_PATTERN.match(text)
    if match is None:
        return fallback, text
    return match.group(1), match.group(2).strip()


def _location(value: Any) -> dict[str, int] | None:
    if not isinstance(value, dict):
        return None
    required = ("line", "column", "end_line", "end_column")
    if not all(isinstance(value.get(name), int) for name in required):
        return None
    return {name: value[name] for name in required}


def _context_node(node: GraphNode) -> ContextNode:
    data = {
        name: value
        for name, value in node.data.items()
        if name not in _BLOCK_CONTEXT_OMISSIONS
    }
    return ContextNode(
        id=node.id,
        type=node.type,
        block_id=node.block_id,
        data=data,
        artifact_ids=node.artifact_ids,
    )


def _code_regions(
    graph: DependencyGraph,
    nodes: tuple[GraphNode, ...],
    anchor: FailureAnchor,
    radius: int,
) -> tuple[CodeRegion, ...]:
    focus_by_block: dict[str, set[int]] = {}
    if anchor.block_id is not None and anchor.location is not None:
        focus_by_block.setdefault(anchor.block_id, set()).add(anchor.location["line"])
    for node in nodes:
        if node.block_id is None:
            continue
        location = node.data.get("call_site")
        line = location.get("line") if isinstance(location, dict) else node.data.get("line")
        if isinstance(line, int):
            focus_by_block.setdefault(node.block_id, set()).add(line)

    relevant_blocks = {
        node.block_id for node in nodes if node.block_id is not None
    }
    if anchor.block_id is not None:
        relevant_blocks.add(anchor.block_id)
    regions: list[CodeRegion] = []
    for block in graph.nodes:
        if block.type != "BLOCK" or block.block_id not in relevant_blocks:
            continue
        lines = str(block.data.get("code", "")).splitlines()
        if not lines:
            continue
        focus_lines = sorted(focus_by_block.get(block.block_id, ()))
        ranges = _line_ranges(focus_lines, len(lines), radius)
        if not ranges:
            ranges = [(1, min(len(lines), 2 * radius + 1))]
        for start, end in ranges:
            region_focus = tuple(line for line in focus_lines if start <= line <= end)
            regions.append(
                CodeRegion(
                    block_id=block.block_id,
                    start_line=start,
                    end_line=end,
                    focus_lines=region_focus,
                    code="\n".join(lines[start - 1 : end]),
                )
            )
    return tuple(regions)


def _line_ranges(
    focus_lines: list[int],
    line_count: int,
    radius: int,
) -> list[tuple[int, int]]:
    ranges: list[tuple[int, int]] = []
    for line in focus_lines:
        start = max(1, line - radius)
        end = min(line_count, line + radius)
        if ranges and start <= ranges[-1][1] + 1:
            ranges[-1] = (ranges[-1][0], max(ranges[-1][1], end))
        else:
            ranges.append((start, end))
    return ranges


def _artifact_summaries(
    graph: DependencyGraph,
    nodes: tuple[GraphNode, ...],
    preview_chars: int,
) -> tuple[ArtifactSummary, ...]:
    selected_ids = {
        artifact_id for node in nodes for artifact_id in node.artifact_ids
    }
    return tuple(
        _artifact_summary(artifact, preview_chars)
        for artifact in graph.artifacts
        if artifact.id in selected_ids
    )


def _artifact_summary(
    artifact: GraphArtifact,
    preview_chars: int,
) -> ArtifactSummary:
    serialized = json.dumps(
        artifact.value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return ArtifactSummary(
        id=artifact.id,
        kind=artifact.kind,
        sha256=artifact.sha256,
        chars=artifact.chars,
        preview=serialized[:preview_chars],
        preview_truncated=len(serialized) > preview_chars,
    )
