from __future__ import annotations

from pathlib import Path

from graphptc.failure_attribution import build_failure_contexts
from graphptc.invalidation import analyze_invalidation, write_invalidation_audit_report
from graphptc.patch_controller import (
    LocalPatchProposal,
    apply_local_patch,
    build_repair_context,
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


def _application(episode_id: str, proposal: LocalPatchProposal):  # type: ignore[no-untyped-def]
    graph = _graph(episode_id)
    context = build_failure_contexts(graph)[0]
    repair = build_repair_context(graph, context)
    return graph, apply_local_patch(graph, repair, proposal)


def test_output_only_patch_reuses_same_block_tool_results() -> None:
    graph, application = _application(
        "audit-multitool",
        LocalPatchProposal(
            block_id="audit-multitool:block:1",
            start_line=3,
            end_line=3,
            expected_code="print(right[3]['docid'])",
            replacement_code="print(right[0]['docid'])",
        ),
    )

    plan = analyze_invalidation(graph, application)

    assert set(plan.invalidated_node_ids) == {
        "episode:audit-multitool",
        "block:audit-multitool:block:1",
        "output:audit-multitool:block:1",
        "output:audit-multitool:final",
    }
    assert {
        "tool:audit-multitool:block:1:1",
        "tool:audit-multitool:block:1:2",
        "state:audit-multitool:block:1:left",
        "state:audit-multitool:block:1:right",
    }.issubset(plan.reusable_node_ids)
    assert {
        "artifact:tool:audit-multitool:block:1:1:result",
        "artifact:tool:audit-multitool:block:1:2:result",
    }.issubset(plan.reusable_artifact_ids)
    assert {decision.action for decision in plan.tool_decisions} == {"REUSE_RESULT"}


def test_cross_block_patch_keeps_predecessor_state_reusable() -> None:
    graph, application = _application(
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

    assert "state:audit-runtime:block:1:hits" in plan.reusable_node_ids
    assert "tool:audit-runtime:block:1:1" in plan.reusable_node_ids
    assert "block:audit-runtime:block:2" in plan.reexecute_node_ids
    assert "output:audit-runtime:block:2" in plan.reexecute_node_ids
    assert plan.tool_decisions[0].action == "REUSE_RESULT"


def test_tool_patch_requires_read_only_reexecution_or_unknown_tool_reset() -> None:
    graph, application = _application(
        "audit-tool",
        LocalPatchProposal(
            block_id="audit-tool:block:1",
            start_line=1,
            end_line=1,
            expected_code="print(search(query='alpha', timeout=30))",
            replacement_code="print(search(query='alpha'))",
        ),
    )

    read_only = analyze_invalidation(graph, application)
    unknown = analyze_invalidation(graph, application, read_only_tool_names=frozenset())

    assert "tool:audit-tool:block:1:1" in read_only.invalidated_node_ids
    assert read_only.tool_decisions[0].action == "REEXECUTE"
    assert unknown.tool_decisions[0].action == "RESET_REQUIRED"
    assert unknown.tool_decisions[0].computation_invalidated is True


def test_transform_patch_propagates_only_to_causal_downstream_nodes() -> None:
    graph, application = _application(
        "audit-transform",
        LocalPatchProposal(
            block_id="audit-transform:block:1",
            start_line=2,
            end_line=2,
            expected_code="filtered = [hit for hit in hits if hit['score'] > 0]",
            replacement_code="filtered = [hit for hit in hits if hit['score'] >= 0]",
        ),
    )

    plan = analyze_invalidation(graph, application)

    assert "tool:audit-transform:block:1:1" in plan.reusable_node_ids
    assert "state:audit-transform:block:1:hits" in plan.reusable_node_ids
    assert {
        "transform:audit-transform:block:1:data:1",
        "transform:audit-transform:block:1:data:2",
        "transform:audit-transform:block:1:data:3",
        "state:audit-transform:block:1:filtered",
        "state:audit-transform:block:1:unique",
        "state:audit-transform:block:1:count",
        "output:audit-transform:block:1",
    }.issubset(plan.invalidated_node_ids)
    assert any(edge.type == "DATA" for edge in plan.propagation_edges)


def test_assertion_only_patch_keeps_transform_chain_reusable() -> None:
    graph, application = _application(
        "audit-transform",
        LocalPatchProposal(
            block_id="audit-transform:block:1",
            start_line=5,
            end_line=5,
            expected_code="assert count > 1, 'expected multiple unique hits'",
            replacement_code="assert count >= 1, 'expected multiple unique hits'",
        ),
    )

    plan = analyze_invalidation(graph, application)

    assert all(
        node_id in plan.reusable_node_ids
        for node_id in (
            "tool:audit-transform:block:1:1",
            "transform:audit-transform:block:1:data:1",
            "transform:audit-transform:block:1:data:2",
            "transform:audit-transform:block:1:data:3",
        )
    )
    assert plan.tool_decisions[0].action == "REUSE_RESULT"


def test_stage5_invalidation_audit_is_exact_and_deterministic(tmp_path: Path) -> None:
    root = Path(__file__).parents[2]
    graph_path = tmp_path / "graphs.json"
    output_path = tmp_path / "invalidation.json"
    write_dependency_graph_report(
        root / "data" / "stage3" / "failure-audit.events.jsonl",
        graph_path,
    )

    first = write_invalidation_audit_report(
        graph_path,
        root / "configs" / "stage5.invalidation-audit.json",
        output_path,
    )
    first_bytes = output_path.read_bytes()
    second = write_invalidation_audit_report(
        graph_path,
        root / "configs" / "stage5.invalidation-audit.json",
        output_path,
    )

    assert first == second
    assert output_path.read_bytes() == first_bytes
    assert first["passed"] is True
    assert first["case_count"] == 5
    assert first["exact_match_rate"] == 1.0
    assert all(case["passed"] for case in first["cases"])
