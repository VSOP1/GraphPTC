from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from graphptc.failure_attribution import build_failure_contexts
from graphptc.model import ModelTurn, TokenUsage, ToolCall
from graphptc.patch_controller import (
    LocalPatchProposal,
    apply_local_patch,
    build_repair_context,
)
from graphptc.stage2_graph import build_dependency_graphs, load_execution_events
from graphptc.stage4_repair import (
    LOCAL_PATCH_TOOL_SPEC,
    reexecute_patch_prefix,
    request_local_patch,
    reexecute_program_version,
)


class FakeRepairModel:
    def __init__(self, turn: ModelTurn) -> None:
        self.turn = turn
        self.requests: list[dict[str, Any]] = []

    def create_turn(self, **kwargs: Any) -> ModelTurn:
        self.requests.append(kwargs)
        return self.turn


def _graph(episode_id: str):  # type: ignore[no-untyped-def]
    root = Path(__file__).parents[2]
    graphs = build_dependency_graphs(
        load_execution_events(
            root / "data" / "stage3" / "failure-audit.events.jsonl"
        )
    )
    return next(graph for graph in graphs if graph.episode_id == episode_id)


def _repair_context():  # type: ignore[no-untyped-def]
    graph = _graph("audit-multitool")
    return graph, build_repair_context(graph, build_failure_contexts(graph)[0])


def _patch_turn(*, extra_call: bool = False) -> ModelTurn:
    calls = [
        ToolCall(
            id="patch-1",
            name="submit_local_patch",
            input={
                "block_id": "audit-multitool:block:1",
                "start_line": 3,
                "end_line": 3,
                "expected_code": "print(right[3]['docid'])",
                "replacement_code": "print(right[0]['docid'])",
            },
        )
    ]
    if extra_call:
        calls.append(calls[0])
    return ModelTurn(
        assistant_message={"role": "assistant", "content": None},
        text="",
        tool_calls=calls,
        usage=TokenUsage(input_tokens=100, output_tokens=20),
        stop_reason="tool_calls",
    )


def test_repair_request_is_constrained_and_full_block_reexecution_succeeds() -> None:
    graph, repair = _repair_context()
    model = FakeRepairModel(_patch_turn())

    generated = request_local_patch(model, repair)
    application = apply_local_patch(graph, repair, generated.proposal)
    searches: list[str] = []

    def search(*, query: str) -> list[dict[str, Any]]:
        searches.append(query)
        return [{"docid": f"doc-{query}", "score": 1}]

    original = reexecute_program_version(
        application.original,
        namespace={"search": search},
        timeout_seconds=5,
    )
    patched = reexecute_program_version(
        application.patched,
        namespace={"search": search},
        timeout_seconds=5,
    )

    request = model.requests[0]
    assert request["tools"] == [LOCAL_PATCH_TOOL_SPEC]
    assert request["thinking"] == "disabled"
    assert "fewshot-ptc-v1" in request["messages"][0]["content"]
    assert generated.usage == TokenUsage(input_tokens=100, output_tokens=20)
    assert original.success is False
    assert patched.success is True
    assert patched.stdout.strip() == "doc-beta"
    assert searches == ["alpha", "beta", "alpha", "beta"]
    assert patched.runtime_trace["stored_names"] == ["left", "right"]


def test_patch_prefix_reexecution_rebuilds_cross_block_state_without_reuse() -> None:
    graph = _graph("audit-runtime")
    repair = build_repair_context(graph, build_failure_contexts(graph)[0])
    application = apply_local_patch(
        graph,
        repair,
        LocalPatchProposal(
            block_id="audit-runtime:block:2",
            start_line=1,
            end_line=1,
            expected_code="print(hits[2]['title'])",
            replacement_code="print(hits[0]['title'])",
        ),
    )
    searches: list[str] = []

    def search(*, query: str) -> list[dict[str, Any]]:
        searches.append(query)
        return [{"docid": "doc-alpha", "title": "Alpha"}]

    result = reexecute_patch_prefix(
        graph,
        application,
        namespace={"search": search},
        timeout_seconds=5,
    )

    assert result.success is True
    assert result.reused_block_ids == ()
    assert [block.block_id for block in result.blocks] == [
        "audit-runtime:block:1",
        "audit-runtime:block:2",
    ]
    assert [block.success for block in result.blocks] == [True, True]
    assert result.blocks[-1].stdout.strip() == "Alpha"
    assert searches == ["alpha"]


def test_repair_request_rejects_multiple_or_wrong_tool_calls() -> None:
    _, repair = _repair_context()

    with pytest.raises(ValueError, match="exactly one local patch"):
        request_local_patch(FakeRepairModel(_patch_turn(extra_call=True)), repair)

    wrong = _patch_turn()
    wrong.tool_calls[0] = ToolCall(id="x", name="programmatic_tool_call", input={})
    with pytest.raises(ValueError, match="submit_local_patch"):
        request_local_patch(FakeRepairModel(wrong), repair)
