from __future__ import annotations

from pathlib import Path

import pytest

from graphptc.retrieval.local import (
    OfficialCorpusSearchTools,
    SQLiteCorpusSearchTools,
    build_sqlite_fts_index,
    index_document_count,
)


def test_sqlite_corpus_search_is_local_and_records_docids(tmp_path: Path) -> None:
    index = tmp_path / "corpus.sqlite3"
    count = build_sqlite_fts_index(
        index,
        [
            ("1", "alpha alpha relevant evidence", "https://example.com/1"),
            ("2", "beta unrelated", "https://example.com/2"),
        ],
    )
    tools = SQLiteCorpusSearchTools(index, top_k=1, snippet_max_chars=12)

    results = tools.search(query="alpha")

    assert count == index_document_count(index) == 2
    assert results[0]["docid"] == "1"
    assert len(results[0]["snippet"]) == 12
    assert tools.calls[0]["operation"] == "search"
    assert tools.calls[0]["query"] == "alpha"
    assert tools.calls[0]["docids"] == ("1",)


def test_sqlite_corpus_search_rejects_punctuation_only_query(tmp_path: Path) -> None:
    index = tmp_path / "corpus.sqlite3"
    build_sqlite_fts_index(index, [("1", "alpha", "")])
    tools = SQLiteCorpusSearchTools(index)

    with pytest.raises(ValueError, match="at least one word"):
        tools.search(query="---")


def test_sqlite_corpus_fetch_returns_complete_document_and_records_docid(
    tmp_path: Path,
) -> None:
    index = tmp_path / "corpus.sqlite3"
    text = "prefix " + "evidence " * 1000
    build_sqlite_fts_index(index, [("doc-1", text, "https://example.com/1")])
    tools = SQLiteCorpusSearchTools(index, snippet_max_chars=20)

    result = tools.fetch(docid="doc-1")

    assert result == {
        "docid": "doc-1",
        "url": "https://example.com/1",
        "content": text,
    }
    assert tools.calls[0]["operation"] == "fetch"
    assert tools.calls[0]["docid"] == "doc-1"
    assert tools.calls[0]["output_chars"] == len(text)


def test_sqlite_corpus_fetch_rejects_unknown_docid(tmp_path: Path) -> None:
    index = tmp_path / "corpus.sqlite3"
    build_sqlite_fts_index(index, [("doc-1", "alpha", "")])
    tools = SQLiteCorpusSearchTools(index)

    with pytest.raises(KeyError, match="missing"):
        tools.fetch(docid="missing")

    assert tools.calls[0]["operation"] == "fetch"
    assert tools.calls[0]["success"] is False


def test_official_corpus_client_records_search_and_fetch_separately(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tools = OfficialCorpusSearchTools("http://127.0.0.1:8765")

    def request(path: str, payload: dict[str, str] | None) -> object:
        if path == "/search":
            assert payload == {"query": "alpha"}
            return [{"docid": "1", "score": 2.0, "snippet": "result"}]
        assert path == "/fetch"
        assert payload == {"docid": "1"}
        return {"docid": "1", "content": "complete document"}

    monkeypatch.setattr(tools, "_request_json", request)

    assert tools.search(query="alpha")[0]["docid"] == "1"
    assert tools.fetch(docid="1")["content"] == "complete document"
    assert [call["operation"] for call in tools.calls] == ["search", "fetch"]
    assert tools.calls[0]["docids"] == ("1",)
    assert tools.calls[1]["docid"] == "1"
    assert tools.consumed == 2
