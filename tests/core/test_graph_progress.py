from __future__ import annotations

from types import SimpleNamespace

import pytest

from graphptc.graph_progress import GraphProgressView


def test_graph_progress_is_bounded_and_reports_lineage_counters() -> None:
    tools = SimpleNamespace(
        consumed=3,
        calls=[
            {"operation": "search", "query": "alpha", "docids": ["a", "b"]},
            {"operation": "search", "query": "alpha", "docids": ["a", "b"]},
            {"operation": "fetch", "docid": "a", "docids": ["a"]},
        ],
    )
    snapshot = GraphProgressView(tools, mode="graph", max_tool_calls=10).graph_progress()

    assert len(repr(snapshot)) == 512
    assert snapshot["search_calls"] == 2
    assert snapshot["repeated_queries"] == 1
    assert snapshot["zero_novelty_searches"] == 0
    assert snapshot["unfetched_docids"] == 1


def test_placebo_has_same_schema_and_length_without_graph_values() -> None:
    tools = SimpleNamespace(consumed=3, calls=[{"operation": "search", "query": "alpha", "docids": ["a"]}])
    graph = GraphProgressView(tools, mode="graph", max_tool_calls=10).graph_progress()
    placebo = GraphProgressView(tools, mode="placebo", max_tool_calls=10).graph_progress()

    assert graph.keys() == placebo.keys()
    assert len(repr(graph)) == len(repr(placebo)) == 512
    assert placebo["search_calls"] == 0
    assert placebo["remaining_tool_calls"] == 10


def test_automatic_capsules_match_length_and_schema() -> None:
    tools = SimpleNamespace(consumed=1, calls=[{"operation": "search", "query": "alpha", "docids": ["a"]}])
    graph_view = GraphProgressView(tools, mode="graph", max_tool_calls=10)
    placebo_view = GraphProgressView(tools, mode="placebo", max_tool_calls=10)

    graph = graph_view.capsule()
    placebo = placebo_view.capsule()

    assert len(graph) == len(placebo) == 512
    assert graph.startswith("GRAPH_PROGRESS_SNAPSHOT ")
    assert graph_view.telemetry()["snapshot_calls"] == 1
    assert placebo_view.telemetry()["snapshot_calls"] == 1


def test_invalid_progress_mode_rejected() -> None:
    with pytest.raises(ValueError, match="unsupported graph progress mode"):
        GraphProgressView(SimpleNamespace(calls=[], consumed=0), mode="off", max_tool_calls=10)
