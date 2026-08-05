from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from typing import Any, Callable

from .invalidation import InvalidationPlan
from .patch_controller import PatchApplication
from .selective_replay import SelectiveReplayResult, selective_replay_patch
from .stage2_graph import DependencyGraph, build_dependency_graph


@dataclass(frozen=True)
class ExecutionVersion:
    id: str
    episode_id: str
    parent_source_events_sha256: str
    program_version_id: str


@dataclass(frozen=True)
class ReplayCommitResult:
    committed: bool
    execution_version: ExecutionVersion | None
    replay: SelectiveReplayResult
    events: tuple[dict[str, Any], ...]
    graph: DependencyGraph | None


def commit_selective_replay(
    source_graph: DependencyGraph,
    application: PatchApplication,
    plan: InvalidationPlan,
    *,
    live_tools: dict[str, Callable[..., Any]],
    timeout_seconds: float = 120.0,
) -> ReplayCommitResult:
    replay = selective_replay_patch(
        source_graph,
        application,
        plan,
        live_tools=live_tools,
        timeout_seconds=timeout_seconds,
    )
    if not replay.success:
        return ReplayCommitResult(
            committed=False,
            execution_version=None,
            replay=replay,
            events=(),
            graph=None,
        )

    version = _execution_version(source_graph, application, replay)
    events = _build_committed_events(source_graph, application, replay, version)
    graph = build_dependency_graph(events)
    _validate_commit(source_graph, plan, replay, version, graph)
    return ReplayCommitResult(
        committed=True,
        execution_version=version,
        replay=replay,
        events=events,
        graph=graph,
    )


def _execution_version(
    graph: DependencyGraph,
    application: PatchApplication,
    replay: SelectiveReplayResult,
) -> ExecutionVersion:
    identity = {
        "parent_source_events_sha256": graph.source_events_sha256,
        "program_version_id": application.patched.id,
        "blocks": [asdict(block) for block in replay.blocks],
        "tool_events": [asdict(event) for event in replay.tool_events],
    }
    digest = _sha256(identity)
    return ExecutionVersion(
        id=f"execution-version:{digest}",
        episode_id=f"replay:{digest}",
        parent_source_events_sha256=graph.source_events_sha256,
        program_version_id=application.patched.id,
    )


def _build_committed_events(
    source_graph: DependencyGraph,
    application: PatchApplication,
    replay: SelectiveReplayResult,
    version: ExecutionVersion,
) -> tuple[dict[str, Any], ...]:
    episode_node = next(node for node in source_graph.nodes if node.type == "EPISODE")
    events: list[dict[str, Any]] = []

    def emit(
        event_type: str,
        *,
        block_id: str | None = None,
        data: dict[str, Any],
    ) -> None:
        events.append(
            {
                "schema_version": 1,
                "sequence": len(events) + 1,
                "type": event_type,
                "episode_id": version.episode_id,
                "task_id": source_graph.task_id,
                "block_id": block_id,
                "data": data,
            }
        )

    emit(
        "episode.started",
        data={
            "task": episode_node.data.get("task"),
            "execution_version_id": version.id,
            "parent_episode_id": source_graph.episode_id,
            "parent_source_events_sha256": source_graph.source_events_sha256,
            "program_version_id": application.patched.id,
        },
    )
    tool_events_by_block: dict[str, list[Any]] = {}
    for tool_event in replay.tool_events:
        tool_events_by_block.setdefault(tool_event.source_block_id, []).append(
            tool_event
        )

    for ordinal, block in enumerate(replay.blocks, 1):
        new_block_id = f"{version.episode_id}:block:{ordinal}"
        source_block = source_graph.node(f"block:{block.block_id}")
        code = (
            application.patched.code
            if block.block_id == replay.target_block_id
            else str(source_block.data.get("code", ""))
        )
        program_version_id = (
            application.patched.id
            if block.block_id == replay.target_block_id
            else None
        )
        emit(
            "block.started",
            block_id=new_block_id,
            data={
                "turn": ordinal,
                "code": code,
                "execution_version_id": version.id,
                "source_block_id": block.block_id,
                "program_version_id": program_version_id,
            },
        )
        for tool_event in tool_events_by_block.get(block.block_id, ()):
            emit(
                "tool.called",
                block_id=new_block_id,
                data={
                    "tool": tool_event.tool,
                    "arguments": tool_event.arguments,
                    "success": True,
                    "result": tool_event.result,
                    "call_site": tool_event.call_site,
                    "replay_action": tool_event.action,
                    "source_block_id": tool_event.source_block_id,
                    "source_tool_node_id": tool_event.source_tool_node_id,
                    "source_artifact_id": tool_event.source_artifact_id,
                },
            )
        emit(
            "block.finished",
            block_id=new_block_id,
            data={
                "turn": ordinal,
                "code": code,
                "stdout": block.stdout,
                "stdout_chars": len(block.stdout),
                "stdout_truncated": False,
                "success": block.success,
                "error_type": None,
                "error_message": None,
                "runtime_trace": block.runtime_trace,
                "execution_version_id": version.id,
                "source_block_id": block.block_id,
                "program_version_id": program_version_id,
            },
        )
    emit(
        "episode.finished",
        data={
            "status": "success",
            "answer": replay.blocks[-1].stdout.strip(),
            "error": None,
            "ptc_blocks": len(replay.blocks),
            "execution_version_id": version.id,
            "program_version_id": application.patched.id,
        },
    )
    return tuple(events)


def _validate_commit(
    source_graph: DependencyGraph,
    plan: InvalidationPlan,
    replay: SelectiveReplayResult,
    version: ExecutionVersion,
    graph: DependencyGraph,
) -> None:
    if graph.episode_id != version.episode_id:
        raise ValueError("committed graph belongs to the wrong execution version")
    if set(plan.invalidated_artifact_ids) & {artifact.id for artifact in graph.artifacts}:
        raise ValueError("invalidated artifact ID leaked into committed graph")
    replay_by_source = {
        event.source_tool_node_id: event for event in replay.tool_events
    }
    for node in graph.nodes:
        if node.type != "TOOL":
            continue
        source_node_id = node.data.get("source_tool_node_id")
        if source_node_id is None:
            source_block_id = node.data.get("source_block_id")
            if node.data.get("replay_action") != "EXECUTE_NEW":
                raise ValueError("source-free tool node is not a new replay call")
            if f"block:{source_block_id}" not in source_graph.node_ids:
                raise ValueError("new replay tool has unknown source block")
            if node.data.get("source_artifact_id") is not None:
                raise ValueError("new replay tool cannot reuse a source artifact")
            continue
        if source_node_id not in source_graph.node_ids:
            raise ValueError("committed tool node has unknown source provenance")
        source_event = replay_by_source.get(str(source_node_id))
        if source_event is None:
            raise ValueError("committed tool node is absent from replay provenance")
        if node.data.get("replay_action") != source_event.action:
            raise ValueError("committed tool replay action does not match replay")
        if node.data.get("source_artifact_id") != source_event.source_artifact_id:
            raise ValueError("committed tool artifact provenance does not match replay")
        if source_event.source_artifact_id is not None:
            source_graph.artifact(source_event.source_artifact_id)


def _sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
