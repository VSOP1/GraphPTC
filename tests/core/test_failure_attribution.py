from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from graphptc.failure_attribution import (
    build_failure_contexts,
    expand_failure_context,
    failure_expansion_options,
    find_failure_anchors,
    write_failure_attribution_report,
)
from graphptc.stage2_graph import (
    DependencyGraph,
    GraphNode,
    build_dependency_graph,
    write_dependency_graph_report,
)
from graphptc.stage3_audit import write_stage3_audit_report
from graphptc.stage3_gate import write_stage3_precision_gate_report


def _event(
    sequence: int,
    event_type: str,
    *,
    block_id: str | None = None,
    data: dict[str, Any] | None = None,
    episode_id: str = "episode-failure",
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "sequence": sequence,
        "type": event_type,
        "episode_id": episode_id,
        "task_id": "task-failure",
        "block_id": block_id,
        "recorded_at": f"2026-08-05T00:00:0{sequence}+00:00",
        "data": data or {},
    }


def _runtime_failure_events() -> list[dict[str, Any]]:
    first = "episode-failure:block:1"
    second = "episode-failure:block:2"
    return [
        _event(1, "episode.started", data={"task": "find a title"}),
        _event(
            2,
            "block.started",
            block_id=first,
            data={
                "turn": 1,
                "tool_call_id": "call-1",
                "code": "hits = search(query='alpha')\nprint(hits)",
            },
        ),
        _event(
            3,
            "tool.called",
            block_id=first,
            data={
                "tool": "search",
                "arguments": {"query": "alpha"},
                "success": True,
                "result": [{"docid": "doc-alpha", "title": "Alpha"}],
                "call_site": {
                    "line": 1,
                    "column": 7,
                    "end_line": 1,
                    "end_column": 28,
                },
            },
        ),
        _event(
            4,
            "block.finished",
            block_id=first,
            data={
                "turn": 1,
                "code": "hits = search(query='alpha')\nprint(hits)",
                "stdout": "[{'docid': 'doc-alpha', 'title': 'Alpha'}]\n",
                "stdout_chars": 47,
                "stdout_truncated": False,
                "success": True,
                "error_type": None,
                "error_message": None,
                "runtime_trace": {
                    "state_before": {},
                    "state_after": {"hits": "list"},
                    "loaded_names": ["hits", "print", "search"],
                    "stored_names": ["hits"],
                },
            },
        ),
        _event(
            5,
            "block.started",
            block_id=second,
            data={
                "turn": 2,
                "tool_call_id": "call-2",
                "code": "print(hits[2]['title'])",
            },
        ),
        _event(
            6,
            "block.finished",
            block_id=second,
            data={
                "turn": 2,
                "code": "print(hits[2]['title'])",
                "stdout": "PTC_ERROR {...}",
                "stdout_chars": 15,
                "stdout_truncated": False,
                "success": False,
                "error_type": "IndexError",
                "error_message": "list index out of range",
                "runtime_trace": {
                    "state_before": {"hits": "list"},
                    "state_after": {"hits": "list"},
                    "loaded_names": ["hits", "print"],
                    "stored_names": [],
                    "error_location": {
                        "line": 1,
                        "column": 6,
                        "end_line": 1,
                        "end_column": 15,
                    },
                },
            },
        ),
        _event(
            7,
            "episode.finished",
            data={
                "status": "failed",
                "answer": "",
                "error": "IndexError: list index out of range",
                "ptc_blocks": 2,
            },
        ),
    ]


def test_runtime_failure_context_traces_cross_block_state_and_artifact() -> None:
    graph = build_dependency_graph(_runtime_failure_events())

    anchors = find_failure_anchors(graph)
    contexts = build_failure_contexts(graph, preview_chars=20)

    assert len(anchors) == 1
    assert anchors[0].kind == "RUNTIME_ERROR"
    assert anchors[0].error_type == "IndexError"
    assert anchors[0].location == {
        "line": 1,
        "column": 6,
        "end_line": 1,
        "end_column": 15,
    }
    assert len(contexts) == 1
    context = contexts[0]
    assert context.anchor == anchors[0]
    assert {edge.type for edge in context.edges} == {"DATA", "RESULT_OF", "STATE"}
    assert any(node.type == "TOOL" for node in context.nodes)
    assert any(node.type == "STATE" for node in context.nodes)
    assert not any(
        node.type == "OUTPUT" and node.data.get("scope") == "episode"
        for node in context.nodes
    )
    assert all("code" not in node.data for node in context.nodes)
    assert {region.block_id for region in context.code_regions} == {
        "episode-failure:block:1",
        "episode-failure:block:2",
    }
    assert {artifact.kind for artifact in context.artifacts} == {
        "BLOCK_STDOUT",
        "TOOL_RESULT",
    }
    tool_result = next(
        artifact for artifact in context.artifacts if artifact.kind == "TOOL_RESULT"
    )
    assert len(tool_result.preview) <= 20
    assert tool_result.preview_truncated is True

    bounded = build_failure_contexts(graph, max_nodes=2)[0]
    assert len(bounded.nodes) == 2
    assert bounded.truncated is True


def test_failure_context_expansion_is_explicit_and_causally_bounded() -> None:
    graph = build_dependency_graph(_runtime_failure_events())
    bounded = build_failure_contexts(graph, max_nodes=2, preview_chars=20)[0]

    options = failure_expansion_options(graph, bounded)

    assert options.boundary_node_ids
    node_expansion = expand_failure_context(
        graph,
        bounded,
        max_nodes=64,
        preview_chars=20,
    )
    assert node_expansion.context.truncated is False
    assert any(node.type == "TOOL" for node in node_expansion.context.nodes)
    assert any(node.type == "STATE" for node in node_expansion.context.nodes)
    assert node_expansion.options.boundary_node_ids == ()
    tool_result_id = next(
        artifact.id
        for artifact in node_expansion.context.artifacts
        if artifact.kind == "TOOL_RESULT"
    )
    assert tool_result_id in node_expansion.options.artifact_ids
    assert "episode-failure:block:1" in node_expansion.options.code_block_ids

    expanded = expand_failure_context(
        graph,
        node_expansion.context,
        max_nodes=64,
        artifact_ids=(tool_result_id,),
        code_block_ids=("episode-failure:block:1",),
        preview_chars=20,
    )

    assert expanded.artifacts[0].value == [
        {"docid": "doc-alpha", "title": "Alpha"}
    ]
    assert expanded.code_blocks[0].code == (
        "hits = search(query='alpha')\nprint(hits)"
    )
    assert expanded.options.boundary_node_ids == ()


def test_failure_context_expansion_rejects_shrinking_and_unrelated_content() -> None:
    graph = build_dependency_graph(_runtime_failure_events())
    context = build_failure_contexts(graph, max_nodes=3)[0]
    unrelated_artifact_id = next(
        artifact.id
        for artifact in graph.artifacts
        if artifact.kind == "FINAL_ANSWER"
    )

    try:
        expand_failure_context(graph, context, max_nodes=2)
    except ValueError as error:
        assert str(error) == "max_nodes cannot shrink the current context"
    else:
        raise AssertionError("shrinking a failure context must fail")

    try:
        expand_failure_context(
            graph,
            context,
            artifact_ids=(unrelated_artifact_id,),
        )
    except ValueError as error:
        assert "artifact is outside the expanded causal context" in str(error)
    else:
        raise AssertionError("unrelated artifact expansion must fail")


def test_tool_failure_is_the_most_specific_anchor() -> None:
    block_id = "episode-tool:block:1"
    events = [
        _event(
            1,
            "episode.started",
            data={"task": "tool failure"},
            episode_id="episode-tool",
        ),
        _event(
            2,
            "block.started",
            block_id=block_id,
            data={"turn": 1, "code": "search(query='x')"},
            episode_id="episode-tool",
        ),
        _event(
            3,
            "tool.called",
            block_id=block_id,
            data={
                "tool": "search",
                "arguments": {"query": "x"},
                "success": False,
                "error": "TimeoutError: retriever timed out",
                "call_site": {
                    "line": 1,
                    "column": 0,
                    "end_line": 1,
                    "end_column": 17,
                },
            },
            episode_id="episode-tool",
        ),
        _event(
            4,
            "block.finished",
            block_id=block_id,
            data={
                "turn": 1,
                "code": "search(query='x')",
                "stdout": "PTC_ERROR {...}",
                "success": False,
                "error_type": "RuntimeError",
                "error_message": "TimeoutError: retriever timed out",
            },
            episode_id="episode-tool",
        ),
        _event(
            5,
            "episode.finished",
            data={"status": "failed", "error": "tool failed", "ptc_blocks": 1},
            episode_id="episode-tool",
        ),
    ]

    anchors = find_failure_anchors(build_dependency_graph(events))

    assert len(anchors) == 1
    assert anchors[0].kind == "TOOL_ERROR"
    assert anchors[0].error_type == "TimeoutError"
    assert anchors[0].message == "retriever timed out"


def test_handled_tool_failure_does_not_hide_later_episode_failure() -> None:
    graph = DependencyGraph(
        episode_id="episode-handled",
        task_id="task-handled",
        source_event_count=0,
        source_events_sha256="0" * 64,
        nodes=(
            GraphNode(
                id="tool:handled",
                type="TOOL",
                episode_id="episode-handled",
                block_id="episode-handled:block:1",
                data={"success": False, "error": "TimeoutError: recovered"},
            ),
            GraphNode(
                id="output:handled:block",
                type="OUTPUT",
                episode_id="episode-handled",
                block_id="episode-handled:block:1",
                data={"scope": "block", "success": True},
            ),
            GraphNode(
                id="output:handled:final",
                type="OUTPUT",
                episode_id="episode-handled",
                block_id=None,
                data={
                    "scope": "episode",
                    "status": "failed",
                    "error": "APIConnectionError: Connection error.",
                },
            ),
        ),
        edges=(),
        artifacts=(),
    )

    anchors = find_failure_anchors(graph)

    assert [anchor.kind for anchor in anchors] == ["TOOL_ERROR", "EPISODE_ERROR"]


def test_episode_failure_is_fallback_and_report_is_deterministic(
    tmp_path: Path,
) -> None:
    events = [
        _event(1, "episode.started", data={"task": "api failure"}),
        _event(
            2,
            "episode.finished",
            data={
                "status": "failed",
                "answer": "",
                "error": "APIConnectionError: Connection error.",
                "ptc_blocks": 0,
            },
        ),
    ]
    graph = build_dependency_graph(events)
    anchors = find_failure_anchors(graph)
    events_path = tmp_path / "events.jsonl"
    graph_path = tmp_path / "graph.json"
    output_path = tmp_path / "failures.json"
    events_path.write_text(
        "\n".join(json.dumps(event) for event in events) + "\n",
        encoding="utf-8",
    )
    write_dependency_graph_report(events_path, graph_path)

    first = write_failure_attribution_report(graph_path, output_path)
    first_bytes = output_path.read_bytes()
    second = write_failure_attribution_report(graph_path, output_path)

    assert len(anchors) == 1
    assert anchors[0].kind == "EPISODE_ERROR"
    assert anchors[0].error_type == "APIConnectionError"
    assert first == second
    assert output_path.read_bytes() == first_bytes
    assert first["schema_version"] == 1
    assert first["failure_count"] == 1
    assert first["episodes"][0]["contexts"][0]["anchor"]["kind"] == "EPISODE_ERROR"


def test_fixed_stage3_audit_set_passes_and_is_deterministic(tmp_path: Path) -> None:
    root = Path(__file__).parents[2]
    events_path = root / "data" / "stage3" / "failure-audit.events.jsonl"
    expectations_path = root / "configs" / "stage3.failure-audit.json"
    graph_path = tmp_path / "graphs.json"
    output_path = tmp_path / "audit.json"
    write_dependency_graph_report(events_path, graph_path)

    first = write_stage3_audit_report(graph_path, expectations_path, output_path)
    first_bytes = output_path.read_bytes()
    second = write_stage3_audit_report(graph_path, expectations_path, output_path)

    assert first == second
    assert output_path.read_bytes() == first_bytes
    assert first["passed"] is True
    assert first["case_count"] == 8
    assert first["passed_case_count"] == 8
    assert first["failure_count"] == 8
    assert all(case["passed"] for case in first["cases"])


def test_stage3_precision_gate_passes_exactly_and_is_deterministic(
    tmp_path: Path,
) -> None:
    root = Path(__file__).parents[2]
    events_path = root / "data" / "stage3" / "failure-audit.events.jsonl"
    expectations_path = root / "configs" / "stage3.precision-gate.json"
    graph_path = tmp_path / "graphs.json"
    output_path = tmp_path / "precision-gate.json"
    write_dependency_graph_report(events_path, graph_path)

    first = write_stage3_precision_gate_report(
        graph_path, expectations_path, output_path
    )
    first_bytes = output_path.read_bytes()
    second = write_stage3_precision_gate_report(
        graph_path, expectations_path, output_path
    )

    assert first == second
    assert output_path.read_bytes() == first_bytes
    assert first["passed"] is True
    assert first["case_count"] == 8
    assert first["context_count"] == 8
    assert first["exact_match_rate"] == 1.0
    assert first["forbidden_leakage_count"] == 0
    assert all(case["passed"] for case in first["cases"])
