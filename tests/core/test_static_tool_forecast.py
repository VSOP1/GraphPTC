from __future__ import annotations

from graphptc.static_tool_forecast import forecast_tool_calls


def test_expands_literal_and_named_collection_loops() -> None:
    result = forecast_tool_calls(
        "queries = ['a', 'b']\nfor q in queries:\n    search(query=q)\n"
        "for docid in ['1', '2', '3']:\n    fetch(docid=docid)\n"
    )

    assert result["fully_determined"] is True
    assert result["known_search_calls"] == 2
    assert result["known_fetch_calls"] == 3


def test_marks_calls_in_unknown_loops_as_unknown() -> None:
    result = forecast_tool_calls(
        "for hit in search(query='a'):\n    fetch(docid=hit['docid'])\n"
    )

    assert result["known_search_calls"] == 1
    assert result["known_fetch_calls"] == 0
    assert result["unknown_fetch_sites"] == 1
    assert result["fully_determined"] is False
