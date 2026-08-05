from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Callable

from .failure_attribution import build_failure_contexts
from .invalidation import analyze_invalidation
from .patch_controller import (
    GRAPHPTC_REPAIR_PROMPT_VARIANT,
    LocalPatchProposal,
    apply_local_patch,
    build_repair_context,
)
from .replay_commit import ReplayCommitResult, commit_selective_replay
from .stage2_graph import DependencyGraph, load_dependency_graph_report


def write_stage5_commit_gate_report(
    graph_path: str | Path,
    expectations_path: str | Path,
    output_path: str | Path,
) -> dict[str, Any]:
    expectation_bytes = Path(expectations_path).read_bytes()
    specification = json.loads(expectation_bytes)
    if not isinstance(specification, dict) or specification.get("schema_version") != 1:
        raise ValueError("Unsupported Stage 5 commit gate specification")
    if specification.get("prompt_variant") != GRAPHPTC_REPAIR_PROMPT_VARIANT:
        raise ValueError("Stage 5 commit gate requires prompt_variant='fewshot-ptc-v1'")
    cases = specification.get("cases")
    if not isinstance(cases, list) or not all(isinstance(case, dict) for case in cases):
        raise ValueError("Stage 5 commit gate requires a cases list")
    graphs = load_dependency_graph_report(graph_path)
    graphs_by_episode = {graph.episode_id: graph for graph in graphs}

    results = []
    exact_passed = 0
    exact_total = 0
    for case in cases:
        episode_id = str(case.get("episode_id") or "")
        source_graph = graphs_by_episode.get(episode_id)
        if source_graph is None:
            raise ValueError(f"Unknown Stage 5 commit episode: {episode_id}")
        application = _application(source_graph, case)
        plan = analyze_invalidation(
            source_graph,
            application,
            read_only_tool_names=(
                frozenset()
                if case.get("force_reset") is True
                else frozenset({"search", "fetch"})
            ),
        )
        source_snapshot = source_graph.to_dict()
        live_calls: list[dict[str, Any]] = []
        commit = commit_selective_replay(
            source_graph,
            application,
            plan,
            live_tools=_live_tools(case, live_calls),
            timeout_seconds=5,
        )
        actual = _actual(
            source_graph,
            source_snapshot,
            application.patched.id,
            plan.invalidated_artifact_ids,
            commit,
            live_calls,
        )
        expected = case.get("expected")
        if not isinstance(expected, dict):
            raise ValueError("Stage 5 commit case requires expected values")
        checks = {name: actual.get(name) == value for name, value in expected.items()}
        exact_passed += sum(checks.values())
        exact_total += len(checks)
        results.append(
            {
                "case_id": case.get("case_id"),
                "episode_id": episode_id,
                "passed": all(checks.values()),
                "checks": checks,
                "actual": actual,
            }
        )
    exact_match_rate = exact_passed / exact_total if exact_total else 1.0
    report = {
        "schema_version": 1,
        "prompt_variant": GRAPHPTC_REPAIR_PROMPT_VARIANT,
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


def _application(source_graph: DependencyGraph, case: dict[str, Any]):  # type: ignore[no-untyped-def]
    anchor_node_id = str(case.get("anchor_node_id") or "")
    contexts = [
        context
        for context in build_failure_contexts(source_graph)
        if context.anchor.node_id == anchor_node_id
    ]
    if len(contexts) != 1:
        raise ValueError(f"Expected one Stage 5 commit anchor: {anchor_node_id}")
    proposal_value = case.get("proposal")
    if not isinstance(proposal_value, dict):
        raise ValueError("Stage 5 commit case requires a proposal")
    try:
        proposal = LocalPatchProposal(**proposal_value)
    except TypeError as exc:
        raise ValueError("Invalid Stage 5 commit proposal") from exc
    return apply_local_patch(
        source_graph,
        build_repair_context(source_graph, contexts[0]),
        proposal,
    )


def _actual(
    source_graph: DependencyGraph,
    source_snapshot: dict[str, Any],
    patched_program_version_id: str,
    invalidated_artifact_ids: tuple[str, ...],
    commit: ReplayCommitResult,
    live_calls: list[dict[str, Any]],
) -> dict[str, Any]:
    graph = commit.graph
    version = commit.execution_version
    tool_nodes = [] if graph is None else [node for node in graph.nodes if node.type == "TOOL"]
    graph_node_ids = set() if graph is None else set(graph.node_ids)
    graph_artifact_ids = (
        set() if graph is None else {artifact.id for artifact in graph.artifacts}
    )
    dangling_edges = (
        []
        if graph is None
        else [
            edge
            for edge in graph.edges
            if edge.source not in graph_node_ids or edge.target not in graph_node_ids
        ]
    )
    unknown_provenance = [
        node
        for node in tool_nodes
        if node.data.get("source_tool_node_id") not in source_graph.node_ids
        or (
            node.data.get("source_artifact_id") is not None
            and node.data.get("source_artifact_id")
            not in {artifact.id for artifact in source_graph.artifacts}
        )
    ]
    answer = None
    if graph is not None:
        output = next(
            node
            for node in graph.nodes
            if node.type == "OUTPUT" and node.data.get("scope") == "episode"
        )
        answer = graph.artifact(output.artifact_ids[0]).value
    return {
        "committed": commit.committed,
        "zero_commit": version is None and graph is None and commit.events == (),
        "source_unchanged": source_graph.to_dict() == source_snapshot,
        "execution_versioned": (
            version is not None
            and version.id.startswith("execution-version:")
            and version.parent_source_events_sha256 == source_graph.source_events_sha256
        ),
        "new_episode": graph is not None and graph.episode_id != source_graph.episode_id,
        "program_version_matched": (
            version is not None
            and version.program_version_id == patched_program_version_id
        ),
        "graph_event_count": 0 if graph is None else graph.source_event_count,
        "block_count": 0 if graph is None else sum(node.type == "BLOCK" for node in graph.nodes),
        "tool_actions": [node.data.get("replay_action") for node in tool_nodes],
        "reused_source_artifact_count": sum(
            node.data.get("source_artifact_id") is not None for node in tool_nodes
        ),
        "new_tool_artifact_count": sum(len(node.artifact_ids) for node in tool_nodes),
        "invalidated_artifact_leak_count": len(
            set(invalidated_artifact_ids) & graph_artifact_ids
        ),
        "dangling_edge_count": len(dangling_edges),
        "unknown_source_provenance_count": len(unknown_provenance),
        "answer": answer,
        "live_calls": live_calls,
        "execution_version_id": None if version is None else version.id,
        "new_episode_id": None if version is None else version.episode_id,
    }


def _live_tools(
    case: dict[str, Any],
    live_calls: list[dict[str, Any]],
) -> dict[str, Callable[..., Any]]:
    fixture = case.get("live_tool")
    if fixture is None:
        return {}
    if not isinstance(fixture, dict):
        raise ValueError("Stage 5 commit live_tool must be an object")
    name = str(fixture.get("name") or "")
    if not name or "result" not in fixture:
        raise ValueError("Stage 5 commit live_tool requires name and result")
    result = fixture["result"]

    def tool(**kwargs: Any) -> Any:
        live_calls.append({"tool": name, "arguments": dict(kwargs)})
        return result

    return {name: tool}
