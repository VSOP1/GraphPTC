from __future__ import annotations

from dataclasses import asdict
from typing import Any, Callable, Iterable

from .browsecomp_plus_benchmark import BROWSECOMP_PLUS_RUNTIME_TOOL_MANIFEST
from .failure_attribution import build_failure_contexts
from .invalidation import analyze_invalidation
from .patch_controller import apply_local_patch, build_repair_context
from .persistent_runtime import PersistentIpcRuntime
from .selective_replay import selective_replay_patch
from .stage2_graph import build_dependency_graphs
from .stage4_repair import RepairModel, request_local_patch


def repair_active_block(
    events: Iterable[dict[str, Any]],
    *,
    block_id: str,
    repair_model: RepairModel,
    live_tools: dict[str, Callable[..., Any]],
    runtime: PersistentIpcRuntime,
    timeout_seconds: float,
) -> dict[str, Any]:
    event_list = list(events)
    if not event_list or event_list[-1].get("type") == "episode.finished":
        raise ValueError("active repair requires an incomplete episode")
    terminal = {
        "schema_version": 1,
        "sequence": int(event_list[-1]["sequence"]) + 1,
        "type": "episode.finished",
        "episode_id": event_list[-1]["episode_id"],
        "task_id": event_list[-1]["task_id"],
        "block_id": None,
        "data": {
            "status": "failed",
            "answer": "",
            "error": "active repair snapshot",
            "ptc_blocks": sum(event.get("type") == "block.finished" for event in event_list),
        },
    }
    graphs = build_dependency_graphs((*event_list, terminal))
    if len(graphs) != 1:
        raise ValueError("active repair requires exactly one episode graph")
    graph = graphs[0]
    contexts = [
        context
        for context in build_failure_contexts(graph)
        if context.anchor.block_id == block_id and context.anchor.location is not None
    ]
    if len(contexts) != 1:
        return {"status": "not_repairable", "model_request_count": 0}

    repair = build_repair_context(
        graph,
        contexts[0],
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

    runtime.close()
    replay = selective_replay_patch(
        graph,
        application,
        plan,
        live_tools=live_tools,
        timeout_seconds=timeout_seconds,
        replay_runtime=runtime,
        close_runtime=False,
    )
    if not replay.success or not replay.blocks[-1].stdout.strip():
        runtime.close()
        return {
            "status": "replay_failed",
            "model_request_count": 1,
            "generated_patch": asdict(generated),
            "error": replay.error or "repaired block produced no stdout",
        }
    target = replay.blocks[-1]
    return {
        "status": "repaired_active",
        "model_request_count": 1,
        "source_events_sha256": graph.source_events_sha256,
        "generated_patch": asdict(generated),
        "patched_code": application.patched.code,
        "output": target.stdout,
        "runtime_trace": target.runtime_trace,
        "replay": {
            "reused_tool_call_count": replay.reused_tool_call_count,
            "executed_tool_call_count": replay.executed_tool_call_count,
            "tool_events": [asdict(event) for event in replay.tool_events],
        },
    }
