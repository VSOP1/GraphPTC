from __future__ import annotations

import ast
import copy
import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

from graphptc.stage2_graph import (
    build_dependency_graph,
    build_dependency_graphs,
    load_dependency_graph_report,
    load_execution_events,
    write_dependency_graph_bundle,
    write_dependency_graph_report,
)


def _event(
    sequence: int,
    event_type: str,
    *,
    block_id: str | None = None,
    data: dict[str, Any] | None = None,
    episode_id: str = "episode-1",
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "sequence": sequence,
        "type": event_type,
        "episode_id": episode_id,
        "task_id": "task-1",
        "block_id": block_id,
        "recorded_at": f"2026-08-04T00:00:0{sequence}+00:00",
        "data": data or {},
    }


def _events() -> list[dict[str, Any]]:
    first = "episode-1:block:1"
    second = "episode-1:block:2"
    return [
        _event(1, "episode.started", data={"task": "research"}),
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
                "result": [{"docid": "doc-alpha", "snippet": "alpha"}],
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
                "stdout": "[{'docid': 'doc-alpha'}]\n",
                "stdout_chars": 28,
                "stdout_truncated": False,
                "success": True,
                "error_type": None,
                "error_message": None,
                "runtime_trace": {
                    "state_before": {},
                    "state_after": {"hits": "list"},
                    "loaded_names": ["print", "search", "hits"],
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
                "code": (
                    "left = fetch(docid='a')\n"
                    "right = fetch(docid='b')\n"
                    "print(hits, left, right)"
                ),
            },
        ),
        _event(
            6,
            "tool.called",
            block_id=second,
            data={
                "tool": "fetch",
                "arguments": {"docid": "missing"},
                "success": False,
                "error": "KeyError: missing",
            },
        ),
        _event(
            7,
            "block.finished",
            block_id=second,
            data={
                "turn": 2,
                "code": (
                    "left = fetch(docid='a')\n"
                    "right = fetch(docid='b')\n"
                    "print(hits, left, right)"
                ),
                "stdout": "PTC_STDOUT_TRUNCATED {...}",
                "stdout_chars": 500,
                "stdout_truncated": True,
                "success": False,
                "error_type": "KeyError",
                "error_message": "missing",
                "runtime_trace": {
                    "state_before": {"hits": "list"},
                    "state_after": {"hits": "list"},
                    "loaded_names": ["fetch", "hits", "print"],
                    "stored_names": [],
                },
            },
        ),
        _event(
            8,
            "episode.finished",
            data={
                "status": "success",
                "answer": "alpha",
                "error": None,
                "ptc_blocks": 2,
            },
        ),
    ]


def test_load_execution_events_validates_schema_and_episode_sequence(
    tmp_path: Path,
) -> None:
    path = tmp_path / "events.jsonl"
    path.write_text(
        "\n".join(json.dumps(event) for event in _events()) + "\n",
        encoding="utf-8",
    )

    loaded = load_execution_events(path)

    assert loaded == tuple(_events())

    invalid = _events()
    invalid[2]["sequence"] = 4
    path.write_text(
        "\n".join(json.dumps(event) for event in invalid) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="sequence"):
        load_execution_events(path)


def test_dependency_graph_accepts_repair_lifecycle_event() -> None:
    events = _events()
    terminal = events.pop()
    events.append(
        _event(
            8,
            "repair.finished",
            block_id="episode-1:block:2",
            data={"status": "repaired_active", "model_request_count": 1},
        )
    )
    terminal["sequence"] = 9
    events.append(terminal)

    graph = build_dependency_graph(events)

    assert graph.source_event_count == 9
    assert sum(node.type == "BLOCK" for node in graph.nodes) == 2


def test_build_dependency_graph_is_deterministic_and_conservative() -> None:
    events = _events()

    first = build_dependency_graph(events)
    second = build_dependency_graph(copy.deepcopy(events))

    assert first.to_dict() == second.to_dict()
    assert first.to_dict()["schema_version"] == 3
    assert first.episode_id == "episode-1"
    assert len(first.nodes) == 9
    assert {node.type for node in first.nodes} == {
        "EPISODE",
        "BLOCK",
        "TOOL",
        "OUTPUT",
        "STATE",
    }
    assert len(first.artifacts) == 4
    assert all(len(artifact.sha256) == 64 for artifact in first.artifacts)
    assert all(edge.source in first.node_ids for edge in first.edges)
    assert all(edge.target in first.node_ids for edge in first.edges)
    assert {edge.type for edge in first.edges} == {
        "CONTAINS",
        "PRECEDES",
        "RESULT_OF",
        "DATA",
        "STATE",
    }
    assert not any(edge.type == "CONTROL" for edge in first.edges)

    first_block = next(node for node in first.nodes if node.type == "BLOCK")
    contained = first.successors(first_block.id, edge_type="CONTAINS")
    assert {node.type for node in contained} == {"TOOL", "OUTPUT", "STATE"}
    first_output = next(
        node
        for node in first.nodes
        if node.type == "OUTPUT" and node.block_id == first_block.block_id
    )
    assert first.predecessors(first_output.id, edge_type="RESULT_OF") == (
        first_block,
    )

    tools = [node for node in first.nodes if node.type == "TOOL"]
    assert tools[0].data["source_alignment"] == "runtime_exact"
    assert len(tools[0].data["source_site_ids"]) == 1
    assert tools[1].data["source_alignment"] == "ambiguous"
    assert len(tools[1].data["source_site_ids"]) == 2

    data_edges = [edge for edge in first.edges if edge.type == "DATA"]
    assert len(data_edges) == 2
    assert {edge.source for edge in data_edges} == {tools[0].id}
    assert {edge.data["evidence"] for edge in data_edges} == {
        "static_value_flow_to_stdout",
        "static_value_flow_to_state",
    }

    state_edges = [edge for edge in first.edges if edge.type == "STATE"]
    assert len(state_edges) == 1
    assert state_edges[0].data == {
        "name": "hits",
        "evidence": "runtime_load_from_persistent_state",
    }
    state_node = state_edges[0].source
    assert first.predecessors(state_node, edge_type="DATA") == (tools[0],)

    outputs = [node for node in first.nodes if node.type == "OUTPUT"]
    truncated = next(
        node
        for node in outputs
        if node.data.get("scope") == "block" and node.data["stdout_truncated"]
    )
    assert truncated.data["success"] is False
    assert truncated.data["error_type"] == "KeyError"


def test_build_dependency_graphs_keeps_episodes_separate() -> None:
    first = _events()
    second = [
        _event(
            event["sequence"],
            event["type"],
            block_id=(
                str(event["block_id"]).replace("episode-1", "episode-2")
                if event["block_id"] is not None
                else None
            ),
            data=copy.deepcopy(event["data"]),
            episode_id="episode-2",
        )
        for event in first
    ]

    graphs = build_dependency_graphs([*first, *second])

    assert [graph.episode_id for graph in graphs] == ["episode-1", "episode-2"]
    assert all(len(graph.nodes) == 9 for graph in graphs)
    with pytest.raises(ValueError, match="one episode"):
        build_dependency_graph([*first, *second])


def test_static_value_flow_tracks_loop_and_container_mutation() -> None:
    block_id = "episode-flow:block:1"
    code = (
        "results = []\n"
        "for query in ['a', 'b']:\n"
        "    hits = search(query=query)\n"
        "    for hit in hits:\n"
        "        results.append(hit)\n"
        "print(results)"
    )
    events = [
        _event(
            1,
            "episode.started",
            data={"task": "flow"},
            episode_id="episode-flow",
        ),
        _event(
            2,
            "block.started",
            block_id=block_id,
            data={"turn": 1, "tool_call_id": "call-flow", "code": code},
            episode_id="episode-flow",
        ),
        *[
            _event(
                sequence,
                "tool.called",
                block_id=block_id,
                data={
                    "tool": "search",
                    "arguments": {"query": query},
                    "success": True,
                    "result": [{"docid": query}],
                    "call_site": {
                        "line": 3,
                        "column": 11,
                        "end_line": 3,
                        "end_column": 30,
                    },
                },
                episode_id="episode-flow",
            )
            for sequence, query in ((3, "a"), (4, "b"))
        ],
        _event(
            5,
            "block.finished",
            block_id=block_id,
            data={
                "turn": 1,
                "code": code,
                "stdout": "[{'docid': 'a'}, {'docid': 'b'}]\n",
                "stdout_chars": 36,
                "stdout_truncated": False,
                "success": True,
                "error_type": None,
                "error_message": None,
            },
            episode_id="episode-flow",
        ),
        _event(
            6,
            "episode.finished",
            data={"status": "success", "answer": "done", "ptc_blocks": 1},
            episode_id="episode-flow",
        ),
    ]

    graph = build_dependency_graph(events)

    data_edges = [edge for edge in graph.edges if edge.type == "DATA"]
    assert len(data_edges) == 2
    assert {edge.data["evidence"] for edge in data_edges} == {
        "static_value_flow_to_stdout"
    }


def test_control_dependencies_preserve_dynamic_tool_instances() -> None:
    block_id = "episode-control:block:1"
    code = (
        "queries = ['a', 'b']\n"
        "if queries:\n"
        "    for query in queries:\n"
        "        hits = search(query=query)\n"
        "try:\n"
        "    raise ValueError('x')\n"
        "except ValueError:\n"
        "    doc = fetch(docid='fallback')\n"
        "print(hits, doc)"
    )
    events = [
        _event(
            1,
            "episode.started",
            data={"task": "control"},
            episode_id="episode-control",
        ),
        _event(
            2,
            "block.started",
            block_id=block_id,
            data={"turn": 1, "tool_call_id": "call-control", "code": code},
            episode_id="episode-control",
        ),
        *[
            _event(
                sequence,
                "tool.called",
                block_id=block_id,
                data={
                    "tool": "search",
                    "arguments": {"query": query},
                    "success": True,
                    "result": [{"docid": query}],
                    "call_site": {
                        "line": 4,
                        "column": 15,
                        "end_line": 4,
                        "end_column": 34,
                    },
                },
                episode_id="episode-control",
            )
            for sequence, query in ((3, "a"), (4, "b"))
        ],
        _event(
            5,
            "tool.called",
            block_id=block_id,
            data={
                "tool": "fetch",
                "arguments": {"docid": "fallback"},
                "success": True,
                "result": {"docid": "fallback"},
                "call_site": {
                    "line": 8,
                    "column": 10,
                    "end_line": 8,
                    "end_column": 33,
                },
            },
            episode_id="episode-control",
        ),
        _event(
            6,
            "block.finished",
            block_id=block_id,
            data={
                "turn": 1,
                "code": code,
                "stdout": "[{'docid': 'b'}] {'docid': 'fallback'}\n",
                "stdout_chars": 43,
                "stdout_truncated": False,
                "success": True,
                "error_type": None,
                "error_message": None,
            },
            episode_id="episode-control",
        ),
        _event(
            7,
            "episode.finished",
            data={"status": "success", "answer": "done", "ptc_blocks": 1},
            episode_id="episode-control",
        ),
    ]

    graph = build_dependency_graph(events)

    transforms = [node for node in graph.nodes if node.type == "TRANSFORM"]
    control_edges = [edge for edge in graph.edges if edge.type == "CONTROL"]
    assert len(transforms) == 5
    assert len(control_edges) == 5
    assert [node.data["control_type"] for node in transforms] == [
        "If",
        "For",
        "If",
        "For",
        "Try",
    ]
    assert [node.data["branch"] for node in transforms] == [
        "body",
        "body",
        "body",
        "body",
        "handler",
    ]
    search_tools = [
        node
        for node in graph.nodes
        if node.type == "TOOL" and node.data["tool"] == "search"
    ]
    assert len(search_tools) == 2
    assert all(
        len(graph.predecessors(tool.id, edge_type="CONTROL")) == 1
        for tool in search_tools
    )
    assert len({node.id for node in transforms}) == 5


def test_control_dependencies_require_runtime_exact_alignment() -> None:
    events = _events()
    events[1]["data"]["code"] = (
        "if enabled:\n    hits = search(query='alpha')\nprint(hits)"
    )
    events[3]["data"]["code"] = events[1]["data"]["code"]
    events[2]["data"].pop("call_site")

    graph = build_dependency_graph(events)

    assert not any(node.type == "TRANSFORM" for node in graph.nodes)
    assert not any(edge.type == "CONTROL" for edge in graph.edges)


def test_tool_call_in_condition_is_not_controlled_by_that_condition() -> None:
    events = _events()
    code = "if search(query='alpha'):\n    print('yes')"
    events[1]["data"]["code"] = code
    events[3]["data"]["code"] = code
    events[2]["data"]["call_site"] = {
        "line": 1,
        "column": 3,
        "end_line": 1,
        "end_column": 24,
    }

    graph = build_dependency_graph(events)

    tool = next(node for node in graph.nodes if node.type == "TOOL")
    assert tool.data["source_alignment"] == "runtime_exact"
    assert not any(node.type == "TRANSFORM" for node in graph.nodes)
    assert not any(edge.type == "CONTROL" for edge in graph.edges)


def test_transform_lineage_requires_runtime_execution_spans() -> None:
    block_id = "episode-transform:block:1"
    code = (
        "hits = search(query='alpha')\n"
        "filtered = [hit for hit in hits if hit['score'] > 0]\n"
        "unique = set(hit['docid'] for hit in filtered)\n"
        "count = len(unique)\n"
        "print(count)"
    )
    tree = ast.parse(code)
    transform_nodes = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.ListComp)
        or (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id in {"set", "len"}
        )
    ]
    executed_spans = [
        {
            "line": node.lineno,
            "column": node.col_offset,
            "end_line": node.end_lineno,
            "end_column": node.end_col_offset,
        }
        for node in transform_nodes
    ]
    events = [
        _event(
            1,
            "episode.started",
            data={"task": "transform"},
            episode_id="episode-transform",
        ),
        _event(
            2,
            "block.started",
            block_id=block_id,
            data={"turn": 1, "tool_call_id": "call-transform", "code": code},
            episode_id="episode-transform",
        ),
        _event(
            3,
            "tool.called",
            block_id=block_id,
            data={
                "tool": "search",
                "arguments": {"query": "alpha"},
                "success": True,
                "result": [{"docid": "a", "score": 1}],
                "call_site": {
                    "line": 1,
                    "column": 7,
                    "end_line": 1,
                    "end_column": 28,
                },
            },
            episode_id="episode-transform",
        ),
        _event(
            4,
            "block.finished",
            block_id=block_id,
            data={
                "turn": 1,
                "code": code,
                "stdout": "1\n",
                "stdout_chars": 2,
                "stdout_truncated": False,
                "success": True,
                "error_type": None,
                "error_message": None,
                "runtime_trace": {
                    "state_before": {},
                    "state_after": {
                        "count": "int",
                        "filtered": "list",
                        "hits": "list",
                        "unique": "set",
                    },
                    "loaded_names": [
                        "count",
                        "filtered",
                        "hits",
                        "len",
                        "print",
                        "search",
                        "set",
                        "unique",
                    ],
                    "stored_names": ["count", "filtered", "hits", "unique"],
                    "executed_spans": executed_spans,
                },
            },
            episode_id="episode-transform",
        ),
        _event(
            5,
            "episode.finished",
            data={"status": "success", "answer": "one", "ptc_blocks": 1},
            episode_id="episode-transform",
        ),
    ]

    graph = build_dependency_graph(events)

    transforms = [node for node in graph.nodes if node.type == "TRANSFORM"]
    assert [node.data["transform_type"] for node in transforms] == [
        "FILTER",
        "DEDUP",
        "AGGREGATE",
    ]
    assert all(node.data["evidence"] == "runtime_exact_span" for node in transforms)
    tool = next(node for node in graph.nodes if node.type == "TOOL")
    output = next(
        node
        for node in graph.nodes
        if node.type == "OUTPUT" and node.data["scope"] == "block"
    )
    data_edges = [
        edge
        for edge in graph.edges
        if edge.type == "DATA"
        and edge.data["evidence"] != "static_value_flow_to_state"
    ]
    state_data_edges = [
        edge
        for edge in graph.edges
        if edge.type == "DATA"
        and edge.data["evidence"] == "static_value_flow_to_state"
    ]
    assert len(data_edges) == 4
    assert len(state_data_edges) == 4
    assert data_edges[0].source == tool.id
    assert data_edges[-1].target == output.id
    assert [edge.data["evidence"] for edge in data_edges] == [
        "static_lineage_with_runtime_span",
        "static_lineage_with_runtime_span",
        "static_lineage_with_runtime_span",
        "static_transform_to_stdout",
    ]

    without_spans = copy.deepcopy(events)
    without_spans[3]["data"]["runtime_trace"].pop("executed_spans")
    conservative = build_dependency_graph(without_spans)
    assert not any(node.type == "TRANSFORM" for node in conservative.nodes)


def test_write_dependency_graph_report_is_deterministic(tmp_path: Path) -> None:
    events_path = tmp_path / "events.jsonl"
    output_path = tmp_path / "graph.json"
    events_path.write_text(
        "\n".join(json.dumps(event) for event in _events()) + "\n",
        encoding="utf-8",
    )

    first = write_dependency_graph_report(events_path, output_path)
    first_bytes = output_path.read_bytes()
    second = write_dependency_graph_report(events_path, output_path)

    assert first == second
    assert output_path.read_bytes() == first_bytes
    assert first["graph_count"] == 1
    assert first["schema_version"] == 3
    assert first["graphs"][0]["episode_id"] == "episode-1"
    assert [graph.to_dict() for graph in load_dependency_graph_report(output_path)] == [
        build_dependency_graph(_events()).to_dict()
    ]


def test_graph_queries_artifacts_and_cross_block_state() -> None:
    graph = build_dependency_graph(_events())
    first_tool = next(node for node in graph.nodes if node.type == "TOOL")
    second_block = next(
        node
        for node in graph.nodes
        if node.type == "BLOCK" and node.block_id == "episode-1:block:2"
    )

    artifacts = graph.artifacts_for_node(first_tool.id)

    assert len(artifacts) == 1
    assert graph.artifact(artifacts[0].id) == artifacts[0]
    assert artifacts[0].kind == "TOOL_RESULT"
    state = graph.state_dependencies(second_block.id)
    assert len(state) == 1
    assert state[0].type == "STATE"
    assert state[0].block_id == "episode-1:block:1"
    assert state[0].data["name"] == "hits"
    with pytest.raises(KeyError):
        graph.artifact("artifact:missing")


def test_dependency_graph_bundle_round_trip_and_tamper_detection(
    tmp_path: Path,
) -> None:
    events_path = tmp_path / "events.jsonl"
    manifest_path = tmp_path / "graph-bundle.json"
    artifacts_path = tmp_path / "artifacts.jsonl"
    events_path.write_text(
        "\n".join(json.dumps(event) for event in _events()) + "\n",
        encoding="utf-8",
    )

    first = write_dependency_graph_bundle(
        events_path,
        manifest_path,
        artifacts_path,
    )
    first_manifest_bytes = manifest_path.read_bytes()
    first_artifact_bytes = artifacts_path.read_bytes()
    second = write_dependency_graph_bundle(
        events_path,
        manifest_path,
        artifacts_path,
    )

    assert first == second
    assert manifest_path.read_bytes() == first_manifest_bytes
    assert artifacts_path.read_bytes() == first_artifact_bytes
    assert first["schema_version"] == 4
    assert first["artifact_store"]["path"] == "artifacts.jsonl"
    assert first["artifact_store"]["count"] == 4
    assert all(
        "value" not in artifact
        for graph in first["graphs"]
        for artifact in graph["artifacts"]
    )
    loaded = load_dependency_graph_report(manifest_path)
    assert [graph.to_dict() for graph in loaded] == [
        build_dependency_graph(_events()).to_dict()
    ]

    artifact_rows = [
        json.loads(line) for line in artifacts_path.read_text(encoding="utf-8").splitlines()
    ]
    artifact_rows[0]["artifact"]["value"] = {"tampered": True}
    tampered = (
        "\n".join(
            json.dumps(
                row,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            )
            for row in artifact_rows
        )
        + "\n"
    ).encode("utf-8")
    artifacts_path.write_bytes(tampered)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["artifact_store"]["sha256"] = hashlib.sha256(tampered).hexdigest()
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="artifact integrity"):
        load_dependency_graph_report(manifest_path)
