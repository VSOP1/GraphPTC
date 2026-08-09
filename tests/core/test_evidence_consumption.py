from __future__ import annotations

from typing import Any

from graphptc.evidence_consumption import project_evidence_consumption


def _event(sequence: int, event_type: str, *, block_id: str | None = None, data: dict[str, Any] | None = None) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "sequence": sequence,
        "type": event_type,
        "episode_id": "episode-1",
        "task_id": "1",
        "block_id": block_id,
        "recorded_at": "2026-08-06T00:00:00+00:00",
        "data": data or {},
    }


def _events() -> list[dict[str, Any]]:
    first = "episode-1:block:1"
    second = "episode-1:block:2"
    return [
        _event(1, "episode.started", data={"task": "find alpha"}),
        _event(2, "block.started", block_id=first, data={"turn": 1, "code": "hits = search(query='one')\nprint(hits)"}),
        _event(3, "tool.called", block_id=first, data={"tool": "search", "arguments": {"query": "one"}, "success": True, "result": [{"docid": "d1", "snippet": "Alpha Person"}], "call_site": {"line": 1, "column": 7, "end_line": 1, "end_column": 26}}),
        _event(4, "block.finished", block_id=first, data={"turn": 1, "code": "hits = search(query='one')\nprint(hits)", "stdout": "d1 Alpha Person\n", "stdout_chars": 16, "stdout_truncated": False, "success": True, "runtime_trace": {"state_before": {}, "state_after": {"hits": "list"}, "loaded_names": ["search", "print", "hits"], "stored_names": ["hits"]}}),
        _event(5, "block.started", block_id=second, data={"turn": 2, "code": "again = search(query='two')\npage = fetch(docid='d1')\nprint(page['content'])"}),
        _event(6, "tool.called", block_id=second, data={"tool": "search", "arguments": {"query": "two"}, "success": True, "result": [{"docid": "d1", "snippet": "Alpha Person"}], "call_site": {"line": 1, "column": 8, "end_line": 1, "end_column": 27}}),
        _event(7, "tool.called", block_id=second, data={"tool": "fetch", "arguments": {"docid": "d1"}, "success": True, "result": {"docid": "d1", "content": "Biography of Alpha Person"}, "call_site": {"line": 2, "column": 7, "end_line": 2, "end_column": 24}}),
        _event(8, "block.finished", block_id=second, data={"turn": 2, "code": "again = search(query='two')\npage = fetch(docid='d1')\nprint(page['content'])", "stdout": "Biography of Alpha Person\n", "stdout_chars": 26, "stdout_truncated": False, "success": True, "runtime_trace": {"state_before": {"hits": "list"}, "state_after": {"again": "list", "page": "dict"}, "loaded_names": ["search", "fetch", "print", "page"], "stored_names": ["again", "page"]}}),
        _event(9, "episode.finished", data={"status": "success", "answer": "<result>Alpha Person</result>", "error": None, "ptc_blocks": 2}),
    ]


def test_projects_fetch_and_zero_novelty_query_consumption() -> None:
    projection = project_evidence_consumption(_events())

    assert projection["metrics"]["successful_fetches"] == 1
    assert projection["fetches"][0]["classification"] == "answer_lexical_support"
    assert projection["fetches"][0]["stdout_lineage"] is True
    assert projection["metrics"]["zero_novelty_queries"] == 1
    assert projection["zero_novelty_query_frontier"][0]["classification"] == "answer_support_followup"
    assert "Alpha Person" not in str(projection)


def test_projection_is_deterministic() -> None:
    assert project_evidence_consumption(_events()) == project_evidence_consumption(_events())
