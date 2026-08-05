from __future__ import annotations

from pathlib import Path
from typing import Any

from graphptc.failure_attribution import build_failure_contexts
from graphptc.invalidation import analyze_invalidation
from graphptc.patch_controller import (
    LocalPatchProposal,
    apply_local_patch,
    build_repair_context,
)
from graphptc.replay_commit import commit_selective_replay
from graphptc.stage2_graph import build_dependency_graphs, load_execution_events


def _case(episode_id: str, proposal: LocalPatchProposal):  # type: ignore[no-untyped-def]
    root = Path(__file__).parents[2]
    graphs = build_dependency_graphs(
        load_execution_events(root / "data" / "stage3" / "failure-audit.events.jsonl")
    )
    graph = next(graph for graph in graphs if graph.episode_id == episode_id)
    repair = build_repair_context(graph, build_failure_contexts(graph)[0])
    application = apply_local_patch(graph, repair, proposal)
    return graph, application


def test_successful_commit_creates_versioned_graph_without_mutating_source() -> None:
    graph, application = _case(
        "audit-runtime",
        LocalPatchProposal(
            block_id="audit-runtime:block:2",
            start_line=1,
            end_line=1,
            expected_code="print(hits[2]['title'])",
            replacement_code="print(hits[0]['title'])",
        ),
    )
    plan = analyze_invalidation(graph, application)
    source_snapshot = graph.to_dict()

    commit = commit_selective_replay(
        graph,
        application,
        plan,
        live_tools={},
        timeout_seconds=5,
    )

    assert commit.committed is True
    assert commit.execution_version is not None
    assert commit.graph is not None
    assert graph.to_dict() == source_snapshot
    assert commit.execution_version.parent_source_events_sha256 == (
        graph.source_events_sha256
    )
    assert commit.execution_version.program_version_id == application.patched.id
    assert commit.graph.episode_id == commit.execution_version.episode_id
    assert commit.graph.episode_id != graph.episode_id
    assert commit.graph.source_event_count == len(commit.events)
    assert [event["sequence"] for event in commit.events] == list(
        range(1, len(commit.events) + 1)
    )
    assert not set(plan.invalidated_artifact_ids) & {
        artifact.id for artifact in commit.graph.artifacts
    }

    tool = next(node for node in commit.graph.nodes if node.type == "TOOL")
    assert tool.data["replay_action"] == "REUSE_RESULT"
    assert tool.data["source_tool_node_id"] == "tool:audit-runtime:block:1:1"
    assert tool.data["source_artifact_id"] == (
        "artifact:tool:audit-runtime:block:1:1:result"
    )
    target = next(
        node
        for node in commit.graph.nodes
        if node.type == "BLOCK" and node.data["source_block_id"] == plan.target_block_id
    )
    assert target.data["program_version_id"] == application.patched.id
    assert target.data["code"] == application.patched.code
    assert commit.replay.blocks[-1].stdout.strip() == "Alpha"


def test_reexecuted_tool_gets_new_artifact_with_old_tool_provenance() -> None:
    graph, application = _case(
        "audit-tool",
        LocalPatchProposal(
            block_id="audit-tool:block:1",
            start_line=1,
            end_line=1,
            expected_code="print(search(query='alpha', timeout=30))",
            replacement_code="print(search(query='alpha'))",
        ),
    )
    plan = analyze_invalidation(graph, application)
    calls: list[dict[str, Any]] = []

    def search(**kwargs: Any) -> list[dict[str, Any]]:
        calls.append(kwargs)
        return [{"docid": "fresh-alpha"}]

    commit = commit_selective_replay(
        graph,
        application,
        plan,
        live_tools={"search": search},
        timeout_seconds=5,
    )

    assert commit.committed is True
    assert commit.graph is not None
    assert calls == [{"query": "alpha"}]
    tool = next(node for node in commit.graph.nodes if node.type == "TOOL")
    assert tool.data["replay_action"] == "REEXECUTE"
    assert tool.data["source_tool_node_id"] == "tool:audit-tool:block:1:1"
    assert tool.data["source_artifact_id"] is None
    assert len(tool.artifact_ids) == 1
    assert tool.artifact_ids[0] not in plan.invalidated_artifact_ids
    assert commit.graph.artifact(tool.artifact_ids[0]).value == [
        {"docid": "fresh-alpha"}
    ]


def test_new_tool_call_commits_with_patched_block_provenance() -> None:
    graph, application = _case(
        "audit-runtime",
        LocalPatchProposal(
            block_id="audit-runtime:block:2",
            start_line=1,
            end_line=1,
            expected_code="print(hits[2]['title'])",
            replacement_code=(
                "fresh = search(query='beta')\nprint(fresh[0]['title'])"
            ),
        ),
    )
    plan = analyze_invalidation(graph, application)

    commit = commit_selective_replay(
        graph,
        application,
        plan,
        live_tools={"search": lambda **_: [{"title": "Beta"}]},
        timeout_seconds=5,
    )

    assert commit.committed is True
    assert commit.graph is not None
    tool = next(
        node
        for node in commit.graph.nodes
        if node.type == "TOOL" and node.data["replay_action"] == "EXECUTE_NEW"
    )
    assert tool.data["source_tool_node_id"] is None
    assert tool.data["source_block_id"] == "audit-runtime:block:2"
    assert tool.data["source_artifact_id"] is None


def test_reset_required_has_zero_commit_and_preserves_source_graph() -> None:
    graph, application = _case(
        "audit-tool",
        LocalPatchProposal(
            block_id="audit-tool:block:1",
            start_line=1,
            end_line=1,
            expected_code="print(search(query='alpha', timeout=30))",
            replacement_code="print(search(query='alpha'))",
        ),
    )
    plan = analyze_invalidation(
        graph,
        application,
        read_only_tool_names=frozenset(),
    )
    source_snapshot = graph.to_dict()

    commit = commit_selective_replay(
        graph,
        application,
        plan,
        live_tools={},
        timeout_seconds=5,
    )

    assert commit.committed is False
    assert commit.execution_version is None
    assert commit.graph is None
    assert commit.events == ()
    assert commit.replay.reset_required is True
    assert graph.to_dict() == source_snapshot


def test_successful_commit_is_deterministic() -> None:
    graph, application = _case(
        "audit-runtime",
        LocalPatchProposal(
            block_id="audit-runtime:block:2",
            start_line=1,
            end_line=1,
            expected_code="print(hits[2]['title'])",
            replacement_code="print(hits[0]['title'])",
        ),
    )
    plan = analyze_invalidation(graph, application)

    first = commit_selective_replay(
        graph, application, plan, live_tools={}, timeout_seconds=5
    )
    second = commit_selective_replay(
        graph, application, plan, live_tools={}, timeout_seconds=5
    )

    assert first.execution_version == second.execution_version
    assert first.events == second.events
    assert first.graph == second.graph
