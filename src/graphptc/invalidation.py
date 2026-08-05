from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .failure_attribution import CAUSAL_EDGE_TYPES, build_failure_contexts
from .patch_controller import (
    LocalPatchProposal,
    PatchApplication,
    apply_local_patch,
    build_repair_context,
)
from .stage2_graph import (
    DependencyGraph,
    GraphEdge,
    GraphNode,
    load_dependency_graph_report,
)


DEFAULT_READ_ONLY_TOOLS = frozenset({"search", "fetch"})


@dataclass(frozen=True)
class ToolReplayDecision:
    node_id: str
    tool: str
    computation_invalidated: bool
    action: str
    reason: str
    artifact_ids: tuple[str, ...]


@dataclass(frozen=True)
class InvalidationPlan:
    episode_id: str
    target_block_id: str
    original_version_id: str
    patched_version_id: str
    modified_start_line: int
    modified_end_line: int
    invalidated_node_ids: tuple[str, ...]
    reexecute_node_ids: tuple[str, ...]
    revalidate_node_ids: tuple[str, ...]
    reusable_node_ids: tuple[str, ...]
    invalidated_artifact_ids: tuple[str, ...]
    reusable_artifact_ids: tuple[str, ...]
    tool_decisions: tuple[ToolReplayDecision, ...]
    propagation_edges: tuple[GraphEdge, ...]
    invalidation_reasons: dict[str, tuple[str, ...]]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def analyze_invalidation(
    graph: DependencyGraph,
    application: PatchApplication,
    *,
    read_only_tool_names: frozenset[str] = DEFAULT_READ_ONLY_TOOLS,
) -> InvalidationPlan:
    if application.patched.episode_id != graph.episode_id:
        raise ValueError("patch application belongs to a different episode")
    if application.patched.parent_version_id != application.original.id:
        raise ValueError("patched program does not reference the original version")
    target_block_id = application.patched.block_id
    target_block = _block_node(graph, target_block_id)
    if str(target_block.data.get("code", "")) != application.original.code:
        raise ValueError("original program version does not match the dependency graph")

    blocks = [node for node in graph.nodes if node.type == "BLOCK"]
    block_ids = [node.block_id for node in blocks]
    if target_block_id not in block_ids:
        raise ValueError(f"unknown patch target block: {target_block_id}")
    target_index = block_ids.index(target_block_id)
    later_block_ids = set(block_ids[target_index + 1 :])
    start_line = application.proposal.start_line
    end_line = application.proposal.end_line

    reasons: dict[str, set[str]] = {}

    def invalidate(node_id: str, reason: str) -> None:
        reasons.setdefault(node_id, set()).add(reason)

    episode = next(node for node in graph.nodes if node.type == "EPISODE")
    invalidate(episode.id, "episode_requires_revalidation")
    invalidate(target_block.id, "program_version_changed")
    for node in graph.nodes:
        if node.type == "OUTPUT" and (
            node.block_id == target_block_id or node.data.get("scope") == "episode"
        ):
            invalidate(node.id, "output_requires_recomputation")
        if node.block_id in later_block_ids:
            invalidate(node.id, "later_block_depends_on_prior_agent_observation")
        if node.block_id != target_block_id or node.type not in {"TOOL", "TRANSFORM"}:
            continue
        span = _source_span(node)
        if span is None:
            continue
        if _overlaps(start_line, end_line, span[0], span[1]):
            invalidate(node.id, "source_region_modified")
        elif span[0] > end_line:
            invalidate(node.id, "later_execution_in_modified_block")

    propagation_edges: list[GraphEdge] = []
    queue = [node.id for node in graph.nodes if node.id in reasons]
    visited = set(queue)
    while queue:
        source = queue.pop(0)
        for edge in graph.edges:
            if edge.source != source or edge.type not in CAUSAL_EDGE_TYPES:
                continue
            invalidate(edge.target, f"downstream_of:{source}")
            if edge.target in visited:
                continue
            visited.add(edge.target)
            queue.append(edge.target)
            propagation_edges.append(edge)

    invalidated = set(reasons)
    invalidated_node_ids = tuple(
        node.id for node in graph.nodes if node.id in invalidated
    )
    reexecute_node_ids = tuple(
        node.id
        for node in graph.nodes
        if node.id in invalidated
        and (
            node.type in {"BLOCK", "TOOL", "TRANSFORM", "STATE"}
            or (node.type == "OUTPUT" and node.data.get("scope") == "block")
        )
    )
    revalidate_node_ids = tuple(
        node.id
        for node in graph.nodes
        if node.id in invalidated
        and (
            node.type == "EPISODE"
            or (node.type == "OUTPUT" and node.data.get("scope") == "episode")
        )
    )
    reusable_node_ids = tuple(
        node.id for node in graph.nodes if node.id not in invalidated
    )
    invalidated_artifacts = {
        artifact_id
        for node in graph.nodes
        if node.id in invalidated
        for artifact_id in node.artifact_ids
    }
    invalidated_artifact_ids = tuple(
        artifact.id
        for artifact in graph.artifacts
        if artifact.id in invalidated_artifacts
    )
    reusable_artifact_ids = tuple(
        artifact.id
        for artifact in graph.artifacts
        if artifact.id not in invalidated_artifacts
    )
    tool_decisions = tuple(
        _tool_decision(node, node.id in invalidated, read_only_tool_names)
        for node in graph.nodes
        if node.type == "TOOL"
    )
    return InvalidationPlan(
        episode_id=graph.episode_id,
        target_block_id=target_block_id,
        original_version_id=application.original.id,
        patched_version_id=application.patched.id,
        modified_start_line=start_line,
        modified_end_line=end_line,
        invalidated_node_ids=invalidated_node_ids,
        reexecute_node_ids=reexecute_node_ids,
        revalidate_node_ids=revalidate_node_ids,
        reusable_node_ids=reusable_node_ids,
        invalidated_artifact_ids=invalidated_artifact_ids,
        reusable_artifact_ids=reusable_artifact_ids,
        tool_decisions=tool_decisions,
        propagation_edges=tuple(propagation_edges),
        invalidation_reasons={
            node.id: tuple(sorted(reasons[node.id]))
            for node in graph.nodes
            if node.id in reasons
        },
    )


def write_invalidation_audit_report(
    graph_path: str | Path,
    expectations_path: str | Path,
    output_path: str | Path,
) -> dict[str, Any]:
    expectation_bytes = Path(expectations_path).read_bytes()
    expectations = json.loads(expectation_bytes)
    if not isinstance(expectations, dict) or expectations.get("schema_version") != 1:
        raise ValueError("Unsupported Stage 5 invalidation audit expectations")
    cases = expectations.get("cases")
    if not isinstance(cases, list) or not all(isinstance(case, dict) for case in cases):
        raise ValueError("Stage 5 invalidation audit requires a cases list")
    graphs = load_dependency_graph_report(graph_path)
    graphs_by_episode = {graph.episode_id: graph for graph in graphs}

    results = []
    exact_passed = 0
    exact_total = 0
    for case in cases:
        episode_id = str(case.get("episode_id") or "")
        if episode_id not in graphs_by_episode:
            raise ValueError(f"Unknown Stage 5 audit episode: {episode_id}")
        graph = graphs_by_episode[episode_id]
        anchor_node_id = str(case.get("anchor_node_id") or "")
        contexts = [
            context
            for context in build_failure_contexts(graph)
            if context.anchor.node_id == anchor_node_id
        ]
        if len(contexts) != 1:
            raise ValueError(f"Expected one Stage 5 audit anchor: {anchor_node_id}")
        proposal_value = case.get("proposal")
        if not isinstance(proposal_value, dict):
            raise ValueError("Stage 5 invalidation audit case requires a proposal")
        try:
            proposal = LocalPatchProposal(**proposal_value)
        except TypeError as exc:
            raise ValueError("Invalid Stage 5 audit proposal") from exc
        application = apply_local_patch(
            graph,
            build_repair_context(graph, contexts[0]),
            proposal,
        )
        plan = analyze_invalidation(graph, application)
        tool_actions = [
            {"node_id": decision.node_id, "action": decision.action}
            for decision in plan.tool_decisions
        ]
        checks = {
            "invalidated_node_ids": list(plan.invalidated_node_ids)
            == case.get("invalidated_node_ids"),
            "reexecute_node_ids": list(plan.reexecute_node_ids)
            == case.get("reexecute_node_ids"),
            "revalidate_node_ids": list(plan.revalidate_node_ids)
            == case.get("revalidate_node_ids"),
            "invalidated_artifact_ids": list(plan.invalidated_artifact_ids)
            == case.get("invalidated_artifact_ids"),
            "reusable_artifact_ids": list(plan.reusable_artifact_ids)
            == case.get("reusable_artifact_ids"),
            "tool_actions": tool_actions == case.get("tool_actions"),
            "node_partition": (
                set(plan.invalidated_node_ids).isdisjoint(plan.reusable_node_ids)
                and set(plan.invalidated_node_ids) | set(plan.reusable_node_ids)
                == set(graph.node_ids)
            ),
            "artifact_partition": (
                set(plan.invalidated_artifact_ids).isdisjoint(
                    plan.reusable_artifact_ids
                )
                and set(plan.invalidated_artifact_ids)
                | set(plan.reusable_artifact_ids)
                == {artifact.id for artifact in graph.artifacts}
            ),
        }
        exact_names = (
            "invalidated_node_ids",
            "reexecute_node_ids",
            "revalidate_node_ids",
            "invalidated_artifact_ids",
            "reusable_artifact_ids",
            "tool_actions",
        )
        exact_passed += sum(checks[name] for name in exact_names)
        exact_total += len(exact_names)
        results.append(
            {
                "case_id": case.get("case_id"),
                "episode_id": episode_id,
                "passed": all(checks.values()),
                "checks": checks,
                "plan": plan.to_dict(),
            }
        )
    exact_match_rate = exact_passed / exact_total if exact_total else 1.0
    report = {
        "schema_version": 1,
        "expectations_sha256": hashlib.sha256(expectation_bytes).hexdigest(),
        "case_count": len(results),
        "exact_match_rate": exact_match_rate,
        "passed": all(result["passed"] for result in results)
        and exact_match_rate == 1.0,
        "cases": results,
    }
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return report


def _tool_decision(
    node: GraphNode,
    invalidated: bool,
    read_only_tool_names: frozenset[str],
) -> ToolReplayDecision:
    tool = str(node.data.get("tool") or "")
    success = node.data.get("success") is True
    has_artifact = bool(node.artifact_ids)
    if invalidated or not success or not has_artifact:
        if tool in read_only_tool_names:
            action = "REEXECUTE"
            reason = "read_only_call_requires_fresh_result"
        else:
            action = "RESET_REQUIRED"
            reason = "tool_replay_safety_is_unknown"
    else:
        action = "REUSE_RESULT"
        reason = "successful_unaffected_read_only_result"
    return ToolReplayDecision(
        node_id=node.id,
        tool=tool,
        computation_invalidated=invalidated,
        action=action,
        reason=reason,
        artifact_ids=node.artifact_ids,
    )


def _source_span(node: GraphNode) -> tuple[int, int] | None:
    value = node.data.get("call_site") if node.type == "TOOL" else node.data
    if not isinstance(value, dict):
        return None
    line = value.get("line")
    end_line = value.get("end_line", line)
    if not isinstance(line, int) or not isinstance(end_line, int):
        return None
    return line, end_line


def _overlaps(
    first_start: int,
    first_end: int,
    second_start: int,
    second_end: int,
) -> bool:
    return first_start <= second_end and second_start <= first_end


def _block_node(graph: DependencyGraph, block_id: str) -> GraphNode:
    for node in graph.nodes:
        if node.type == "BLOCK" and node.block_id == block_id:
            return node
    raise ValueError(f"unknown block_id: {block_id}")
