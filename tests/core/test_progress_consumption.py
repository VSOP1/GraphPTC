from __future__ import annotations

from graphptc.progress_consumption import project_capsule_consumption


def test_projects_snapshot_into_next_block_action() -> None:
    events = [
        _event("episode.started", 1),
        _event("block.started", 2, block=1, data={"turn": 1, "code": "search(query='a')"}),
        _event("tool.called", 3, block=1, data={"tool": "search", "arguments": {"query": "a"}, "result": [{"docid": "d1"}]}),
        _event("block.finished", 4, block=1, data={"success": True}),
        _event("block.started", 5, block=2, data={"turn": 2, "code": "fetch(docid='d1')"}),
        _event("tool.called", 6, block=2, data={"tool": "fetch", "arguments": {"docid": "d1"}, "result": {"docid": "d1"}}),
        _event("block.finished", 7, block=2, data={"success": True}),
        _event("episode.finished", 8),
    ]

    result = project_capsule_consumption(events, max_tool_calls=10)

    assert result["episode_count"] == 1
    assert result["successful_blocks"] == result["transition_count"] == 2
    first = result["transitions"][0]
    assert first["snapshot"]["unfetched_docids"] == 1
    assert first["next_action"]["known_unfetched_fetches"] == 1
    assert result["transitions"][1]["next_action"]["terminal"] is True


def test_reconstructs_repeat_and_zero_novelty_signals() -> None:
    events = [
        _event("episode.started", 1),
        _event("block.started", 2, block=1, data={"turn": 1, "code": ""}),
        _event("tool.called", 3, block=1, data={"tool": "search", "arguments": {"query": " Alpha "}, "result": [{"docid": "d1"}]}),
        _event("block.finished", 4, block=1, data={"success": True}),
        _event("block.started", 5, block=2, data={"turn": 2, "code": ""}),
        _event("tool.called", 6, block=2, data={"tool": "search", "arguments": {"query": "alpha"}, "result": [{"docid": "d1"}]}),
        _event("block.finished", 7, block=2, data={"success": True}),
        _event("episode.finished", 8),
    ]

    result = project_capsule_consumption(events, max_tool_calls=10)

    second = result["transitions"][1]["snapshot"]
    assert second["repeated_queries"] == 1
    assert second["zero_novelty_searches"] == 0
    assert result["transitions"][0]["next_action"]["exact_repeat_searches"] == 1


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
