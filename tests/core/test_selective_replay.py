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
from graphptc.selective_replay import (
    selective_replay_patch,
    write_selective_replay_audit_report,
)
from graphptc.stage2_graph import (
    build_dependency_graphs,
    load_execution_events,
    write_dependency_graph_report,
)


def _graph(episode_id: str):  # type: ignore[no-untyped-def]
    root = Path(__file__).parents[2]
    graphs = build_dependency_graphs(
        load_execution_events(
            root / "data" / "stage3" / "failure-audit.events.jsonl"
        )
    )
    return next(graph for graph in graphs if graph.episode_id == episode_id)


def _case(episode_id: str, proposal: LocalPatchProposal):  # type: ignore[no-untyped-def]
    graph = _graph(episode_id)
    repair = build_repair_context(graph, build_failure_contexts(graph)[0])
    application = apply_local_patch(graph, repair, proposal)
    return graph, application, analyze_invalidation(graph, application)


def test_same_block_replay_reuses_both_search_results_without_live_calls() -> None:
    graph, application, plan = _case(
        "audit-multitool",
        LocalPatchProposal(
            block_id="audit-multitool:block:1",
            start_line=3,
            end_line=3,
            expected_code="print(right[3]['docid'])",
            replacement_code="print(right[0]['docid'])",
        ),
    )
    live_calls: list[dict[str, Any]] = []

    def search(**kwargs: Any) -> list[dict[str, Any]]:
        live_calls.append(kwargs)
        raise AssertionError("reusable search must not execute")

    result = selective_replay_patch(
        graph,
        application,
        plan,
        live_tools={"search": search},
        timeout_seconds=5,
    )

    assert result.success is True
    assert result.blocks[-1].stdout.strip() == "right-1"
    assert result.reused_tool_call_count == 2
    assert result.executed_tool_call_count == 0
    assert result.reset_required is False
    assert live_calls == []
    assert [event.action for event in result.tool_events] == [
        "REUSE_RESULT",
        "REUSE_RESULT",
    ]
    assert all(event.source_artifact_id for event in result.tool_events)


def test_cross_block_replay_rebuilds_state_from_cached_tool_artifact() -> None:
    graph, application, plan = _case(
        "audit-runtime",
        LocalPatchProposal(
            block_id="audit-runtime:block:2",
            start_line=1,
            end_line=1,
            expected_code="print(hits[2]['title'])",
            replacement_code="print(hits[0]['title'])",
        ),
    )

    result = selective_replay_patch(
        graph,
        application,
        plan,
        live_tools={},
        timeout_seconds=5,
    )

    assert result.success is True
    assert [block.block_id for block in result.blocks] == [
        "audit-runtime:block:1",
        "audit-runtime:block:2",
    ]
    assert result.blocks[-1].stdout.strip() == "Alpha"
    assert result.reused_tool_call_count == 1
    assert result.executed_tool_call_count == 0


def test_invalidated_tool_call_executes_live_and_records_provenance() -> None:
    graph, application, plan = _case(
        "audit-tool",
        LocalPatchProposal(
            block_id="audit-tool:block:1",
            start_line=1,
            end_line=1,
            expected_code="print(search(query='alpha', timeout=30))",
            replacement_code="print(search(query='alpha'))",
        ),
    )
    calls: list[dict[str, Any]] = []

    def search(**kwargs: Any) -> list[dict[str, Any]]:
        calls.append(kwargs)
        return [{"docid": "fresh-alpha"}]

    result = selective_replay_patch(
        graph,
        application,
        plan,
        live_tools={"search": search},
        timeout_seconds=5,
    )

    assert result.success is True
    assert result.blocks[-1].stdout.strip() == "[{'docid': 'fresh-alpha'}]"
    assert result.reused_tool_call_count == 0
    assert result.executed_tool_call_count == 1
    assert calls == [{"query": "alpha"}]
    event = result.tool_events[0]
    assert event.action == "REEXECUTE"
    assert event.source_tool_node_id == "tool:audit-tool:block:1:1"
    assert event.source_artifact_id is None


def test_patch_introduced_tool_call_executes_live_without_source_tool_node() -> None:
    graph, application, plan = _case(
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
    calls: list[dict[str, Any]] = []

    def search(**kwargs: Any) -> list[dict[str, Any]]:
        calls.append(kwargs)
        return [{"title": "Beta"}]

    result = selective_replay_patch(
        graph,
        application,
        plan,
        live_tools={"search": search},
        timeout_seconds=5,
    )

    assert result.success is True
    assert result.blocks[-1].stdout.strip() == "Beta"
    assert result.reused_tool_call_count == 1
    assert result.executed_tool_call_count == 1
    assert calls == [{"query": "beta"}]
    event = result.tool_events[-1]
    assert event.action == "EXECUTE_NEW"
    assert event.source_block_id == "audit-runtime:block:2"
    assert event.source_tool_node_id is None
    assert event.source_artifact_id is None


def test_reset_required_plan_stops_before_program_execution() -> None:
    graph, application, _ = _case(
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

    result = selective_replay_patch(
        graph,
        application,
        plan,
        live_tools={},
        timeout_seconds=5,
    )

    assert result.success is False
    assert result.reset_required is True
    assert result.blocks == ()
    assert result.tool_events == ()
    assert "tool:audit-tool:block:1:1" in result.error


def test_stage5_selective_replay_audit_is_exact_and_deterministic(
    tmp_path: Path,
) -> None:
    root = Path(__file__).parents[2]
    graph_path = tmp_path / "graphs.json"
    output_path = tmp_path / "selective-replay.json"
    write_dependency_graph_report(
        root / "data" / "stage3" / "failure-audit.events.jsonl",
        graph_path,
    )

    first = write_selective_replay_audit_report(
        graph_path,
        root / "configs" / "stage5.selective-replay-audit.json",
        output_path,
    )
    first_bytes = output_path.read_bytes()
    second = write_selective_replay_audit_report(
        graph_path,
        root / "configs" / "stage5.selective-replay-audit.json",
        output_path,
    )

    assert first == second
    assert output_path.read_bytes() == first_bytes
    assert first["passed"] is True
    assert first["case_count"] == 4
    assert first["exact_match_rate"] == 1.0
    assert all(case["passed"] for case in first["cases"])
