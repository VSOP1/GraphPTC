from __future__ import annotations

from graphptc.actionable_frontier_r2 import project_actionable_frontier_r2


def test_r2_uses_strict_block_trigger_and_recent_lineage() -> None:
    events = [
        _event("episode.started", 1),
        _event("block.started", 2, block=1, data={"turn": 1}),
        _search(3, 1, "alpha", ["d1", "d2"]),
        _event("block.finished", 4, block=1, data={"success": True}),
        _event("block.started", 5, block=2, data={"turn": 2}),
        _search(6, 2, "alpha", ["d1", "d2"]),
        _search(7, 2, "beta", ["d3"]),
        _event("block.finished", 8, block=2, data={"success": True}),
        _event("block.started", 9, block=3, data={"turn": 3}),
        _search(10, 3, "beta", ["d3"]),
        _event("block.finished", 11, block=3, data={"success": True}),
        _event("block.started", 12, block=4, data={"turn": 4}),
        _event("tool.called", 13, block=4, data={"tool": "fetch", "success": True, "arguments": {"docid": "d3"}, "result": {"docid": "d3"}}),
        _event("block.finished", 14, block=4, data={"success": True}),
        _event("episode.finished", 15),
    ]

    result = project_actionable_frontier_r2(events, max_items=1)

    assert result["triggered_blocks"] == 1
    assert result["actionable_opportunities"] == 1
    opportunity = result["opportunities"][0]
    assert opportunity["source_turn"] == 3
    assert opportunity["trigger_reasons"] == ["all_queries_repeated", "all_searches_zero_novelty"]
    assert opportunity["frontiers"]["lineage_recency"][0]["docid"] == "d3"
    assert opportunity["next_action"]["eligible_first_time_fetches"] == ["d3"]


def _search(sequence: int, block: int, query: str, docids: list[str]) -> dict[str, object]:
    return _event(
        "tool.called",
        sequence,
        block=block,
        data={
            "tool": "search",
            "success": True,
            "arguments": {"query": query},
            "result": [{"docid": docid} for docid in docids],
        },
    )


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
