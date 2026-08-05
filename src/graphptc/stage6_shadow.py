from __future__ import annotations

from dataclasses import asdict
from typing import Any, Callable, Iterable

from .browsecomp_plus_benchmark import BROWSECOMP_PLUS_RUNTIME_TOOL_MANIFEST
from .failure_attribution import build_failure_contexts
from .invalidation import analyze_invalidation
from .patch_controller import apply_local_patch, build_repair_context
from .replay_commit import commit_selective_replay
from .stage2_graph import build_dependency_graphs
from .stage4_repair import RepairModel, request_local_patch


def analyze_shadow_episode(
    events: Iterable[dict[str, Any]],
    *,
    repair_model: RepairModel | None,
    live_tools: dict[str, Callable[..., Any]],
    timeout_seconds: float,
) -> dict[str, Any]:
    if timeout_seconds <= 0:
        raise ValueError("shadow timeout_seconds must be positive")
    event_tuple = tuple(events)
    graphs = build_dependency_graphs(event_tuple)
    if len(graphs) != 1:
        raise ValueError("shadow analysis requires exactly one complete episode")
    graph = graphs[0]
    contexts = build_failure_contexts(graph)
    repairable = [
        context
        for context in contexts
        if context.anchor.block_id is not None and context.anchor.location is not None
    ]
    base = {
        "schema_version": 1,
        "source_episode_id": graph.episode_id,
        "source_events_sha256": graph.source_events_sha256,
        "source_event_count": graph.source_event_count,
        "failure_count": len(contexts),
        "repairable_failure_count": len(repairable),
        "selected_failure_index": None,
        "model_request_count": 0,
        "generated_patch": None,
        "application": None,
        "invalidation": None,
        "replay": None,
        "commit": None,
    }
    if not repairable:
        return {**base, "status": "no_repairable_failure"}
    if repair_model is None:
        return {**base, "status": "repair_model_unavailable"}

    failure = repairable[0]
    repair = build_repair_context(
        graph,
        failure,
        runtime_tool_manifest=BROWSECOMP_PLUS_RUNTIME_TOOL_MANIFEST,
    )
    generated = request_local_patch(
        repair_model,
        repair,
        timeout_seconds=timeout_seconds,
        max_completion_tokens=1024,
    )
    application = apply_local_patch(graph, repair, generated.proposal)
    plan = analyze_invalidation(graph, application)
    commit = commit_selective_replay(
        graph,
        application,
        plan,
        live_tools=live_tools,
        timeout_seconds=timeout_seconds,
    )
    if commit.committed:
        status = "committed_shadow"
    elif commit.replay.reset_required:
        status = "reset_required"
    else:
        status = "replay_failed"
    return {
        **base,
        "status": status,
        "selected_failure_index": 0,
        "model_request_count": 1,
        "selected_failure": failure.to_dict(),
        "generated_patch": asdict(generated),
        "application": application.to_dict(),
        "invalidation": plan.to_dict(),
        "replay": {
            "success": commit.replay.success,
            "reset_required": commit.replay.reset_required,
            "error": commit.replay.error,
            "reused_tool_call_count": commit.replay.reused_tool_call_count,
            "executed_tool_call_count": commit.replay.executed_tool_call_count,
            "unconsumed_reusable_tool_node_ids": list(
                commit.replay.unconsumed_reusable_tool_node_ids
            ),
            "tool_events": [asdict(event) for event in commit.replay.tool_events],
            "blocks": [asdict(block) for block in commit.replay.blocks],
        },
        "commit": {
            "committed": commit.committed,
            "execution_version_id": (
                None if commit.execution_version is None else commit.execution_version.id
            ),
            "episode_id": (
                None
                if commit.execution_version is None
                else commit.execution_version.episode_id
            ),
            "event_count": len(commit.events),
            "node_count": 0 if commit.graph is None else len(commit.graph.nodes),
            "edge_count": 0 if commit.graph is None else len(commit.graph.edges),
            "artifact_count": (
                0 if commit.graph is None else len(commit.graph.artifacts)
            ),
        },
    }
