from __future__ import annotations

from graphptc.actionable_frontier import project_actionable_frontier


def test_frontier_triggers_on_repeat_and_carries_concrete_lineage() -> None:
    events = [
        _event("episode.started", 1),
        _event("block.started", 2, block=1, data={"turn": 1}),
        _event("tool.called", 3, block=1, data={"tool": "search", "arguments": {"query": "alpha"}, "result": [{"docid": "d1"}, {"docid": "d2"}]}),
        _event("block.finished", 4, block=1, data={"success": True}),
        _event("block.started", 5, block=2, data={"turn": 2}),
        _event("tool.called", 6, block=2, data={"tool": "search", "arguments": {"query": "alpha"}, "result": [{"docid": "d1"}, {"docid": "d2"}]}),
        _event("block.finished", 7, block=2, data={"success": True}),
        _event("block.started", 8, block=3, data={"turn": 3}),
        _event("tool.called", 9, block=3, data={"tool": "fetch", "arguments": {"docid": "d1"}, "result": {"docid": "d1"}}),
        _event("block.finished", 10, block=3, data={"success": True}),
        _event("episode.finished", 11),
    ]

    result = project_actionable_frontier(events, max_items=1)

    assert result["triggered_blocks"] == 1
    assert result["actionable_opportunities"] == 1
    opportunity = result["opportunities"][0]
    assert opportunity["trigger_reasons"] == ["exact_query_repeat", "zero_novelty_search"]
    assert opportunity["frontiers"]["graph"][0]["docid"] == "d1"
    assert opportunity["frontiers"]["graph"][0]["source_query"] == "alpha"
    assert opportunity["next_action"]["eligible_first_time_fetches"] == ["d1"]
    assert result["summary"]["ranking"]["graph"]["opportunity_coverage"] == 1.0


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
