from __future__ import annotations

from graphptc.graph_adaptation import (
    ActionableFrontierBuilder,
    BlockAssessment,
    GraphActionPolicy,
    GraphTrigger,
    GraphTriggerDetector,
    project_shadow_adaptation,
)


def test_trigger_requires_whole_block_stagnation() -> None:
    detector = GraphTriggerDetector()

    assert detector.detect(BlockAssessment(2, False, False, (), ("new",))) is None
    trigger = detector.detect(BlockAssessment(2, True, True, (), ()))

    assert trigger is not None
    assert trigger.reasons == ("all_queries_repeated", "all_searches_zero_novelty")


def test_frontier_is_recent_bounded_and_excludes_fetched() -> None:
    documents = {
        "old": _document("old", source_call=1),
        "recent": _document("recent", source_call=3),
        "fetched": _document("fetched", source_call=4),
    }

    frontier = ActionableFrontierBuilder(max_items=1).build(documents, {"fetched"})

    assert [item.docid for item in frontier] == ["recent"]
    assert frontier[0].source_query == "query recent"


def test_policy_prioritizes_reuse_for_repeated_fetch() -> None:
    proposal = GraphActionPolicy().propose(
        GraphTrigger(("repeated_fetch", "all_queries_repeated"), ("d1",)),
        (),
    )

    assert proposal.action == "REUSE_FETCHED_ARTIFACT"
    assert proposal.target_docids == ("d1",)
    assert proposal.model_visible is False
    assert proposal.action_taken is None


def test_projection_emits_shadow_proposal_without_action() -> None:
    events = [
        _event("episode.started", 1),
        _event("block.started", 2, block=1, data={"turn": 1}),
        _event("tool.called", 3, block=1, data={"tool": "search", "success": True, "arguments": {"query": "alpha"}, "result": [{"docid": "d1"}]}),
        _event("block.finished", 4, block=1, data={"success": True}),
        _event("block.started", 5, block=2, data={"turn": 2}),
        _event("tool.called", 6, block=2, data={"tool": "search", "success": True, "arguments": {"query": "alpha"}, "result": [{"docid": "d1"}]}),
        _event("block.finished", 7, block=2, data={"success": True}),
        _event("episode.finished", 8),
    ]

    result = project_shadow_adaptation(events)

    assert result["triggered_blocks"] == 1
    assert result["model_visible"] is False
    assert result["action_taken"] is None
    assert result["proposals"][0]["proposed_action"]["action"] == "CHANGE_QUERY_DIRECTION"


def _document(docid: str, *, source_call: int) -> dict[str, object]:
    return {
        "docid": docid,
        "source_query": f"query {docid}",
        "source_turn": 1,
        "source_call": source_call,
        "result_rank": 1,
        "retrieval_count": 1,
    }


def _event(
    event_type: str,
    sequence: int,
    *,
    block: int | None = None,
    data: dict[str, object] | None = None,
) -> dict[str, object]:
    return {
        "type": event_type,
        "sequence": sequence,
        "episode_id": "run:1",
        "task_id": "1",
        "block_id": None if block is None else f"run:1:block:{block}",
        "data": data or {},
    }
