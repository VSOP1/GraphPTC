from __future__ import annotations

from pathlib import Path

import pytest

from graphptc.failure_attribution import build_failure_contexts
from graphptc.patch_controller import (
    GRAPHPTC_REPAIR_PROMPT_VARIANT,
    LocalPatchProposal,
    apply_local_patch,
    build_repair_context,
    write_stage4_patch_report,
)
from graphptc.stage2_graph import build_dependency_graphs, load_execution_events


def _multitool_graph():  # type: ignore[no-untyped-def]
    root = Path(__file__).parents[2]
    graphs = build_dependency_graphs(
        load_execution_events(
            root / "data" / "stage3" / "failure-audit.events.jsonl"
        )
    )
    return next(graph for graph in graphs if graph.episode_id == "audit-multitool")


def test_repair_context_and_local_patch_are_bounded_and_versioned() -> None:
    graph = _multitool_graph()
    failure = build_failure_contexts(graph)[0]
    repair = build_repair_context(graph, failure)
    proposal = LocalPatchProposal(
        block_id="audit-multitool:block:1",
        start_line=3,
        end_line=3,
        expected_code="print(right[3]['docid'])",
        replacement_code="print(right[0]['docid'])",
    )

    first = apply_local_patch(graph, repair, proposal)
    second = apply_local_patch(graph, repair, proposal)

    assert repair.task == "exclude an unrelated tool result in the same block"
    assert repair.prompt_variant == GRAPHPTC_REPAIR_PROMPT_VARIANT
    assert repair.prompt_variant == "fewshot-ptc-v1"
    assert [region.block_id for region in repair.patchable_regions] == [
        "audit-multitool:block:1"
    ]
    assert first == second
    assert first.original.parent_version_id is None
    assert first.patched.parent_version_id == first.original.id
    assert first.original.code.endswith("print(right[3]['docid'])")
    assert first.patched.code.endswith("print(right[0]['docid'])")
    assert first.mapping.old_start_line == 3
    assert first.mapping.old_end_line == 3
    assert first.mapping.new_start_line == 3
    assert first.mapping.new_end_line == 3


@pytest.mark.parametrize(
    ("proposal", "message"),
    [
        (
            LocalPatchProposal(
                block_id="audit-multitool:block:99",
                start_line=1,
                end_line=1,
                expected_code="x = 1",
                replacement_code="x = 2",
            ),
            "outside the repair context",
        ),
        (
            LocalPatchProposal(
                block_id="audit-multitool:block:1",
                start_line=3,
                end_line=3,
                expected_code="print(right[2]['docid'])",
                replacement_code="print(right[0]['docid'])",
            ),
            "expected_code does not match",
        ),
        (
            LocalPatchProposal(
                block_id="audit-multitool:block:1",
                start_line=3,
                end_line=3,
                expected_code="print(right[3]['docid'])",
                replacement_code="print(",
            ),
            "patched program is not valid Python",
        ),
    ],
)
def test_local_patch_rejects_unrelated_stale_or_invalid_changes(
    proposal: LocalPatchProposal,
    message: str,
) -> None:
    graph = _multitool_graph()
    repair = build_repair_context(graph, build_failure_contexts(graph)[0])

    with pytest.raises(ValueError, match=message):
        apply_local_patch(graph, repair, proposal)


def test_stage4_patch_report_is_deterministic(tmp_path: Path) -> None:
    root = Path(__file__).parents[2]
    graph_path = tmp_path / "graphs.json"
    output_path = tmp_path / "patch.json"
    from graphptc.stage2_graph import write_dependency_graph_report

    write_dependency_graph_report(
        root / "data" / "stage3" / "failure-audit.events.jsonl",
        graph_path,
    )
    first = write_stage4_patch_report(
        graph_path,
        root / "configs" / "stage4.local-patch-smoke.json",
        output_path,
    )
    first_bytes = output_path.read_bytes()
    second = write_stage4_patch_report(
        graph_path,
        root / "configs" / "stage4.local-patch-smoke.json",
        output_path,
    )

    assert first == second
    assert output_path.read_bytes() == first_bytes
    assert first["prompt_variant"] == "fewshot-ptc-v1"
    assert first["application"]["patched"]["parent_version_id"] == (
        first["application"]["original"]["id"]
    )
