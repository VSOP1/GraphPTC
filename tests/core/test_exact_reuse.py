from graphptc.exact_reuse import ExactReuseSearchTools


class FakeTools:
    def __init__(self) -> None:
        self.search_count = 0
        self.fetch_count = 0

    @property
    def calls(self):
        return []

    @property
    def consumed(self):
        return self.search_count + self.fetch_count

    def metadata(self):
        return {"ok": True}

    def search(self, *, query: str):
        self.search_count += 1
        return [{"docid": "d1", "score": 1, "snippet": query}]

    def fetch(self, *, docid: str):
        self.fetch_count += 1
        return {"docid": docid, "content": "evidence"}


def test_exact_reuse_preserves_values_and_skips_duplicate_live_calls() -> None:
    inner = FakeTools()
    tools = ExactReuseSearchTools(inner)  # type: ignore[arg-type]
    first = tools.search(query="A  B")
    second = tools.search(query="a b")
    fetched = tools.fetch(docid="d1")
    fetched_again = tools.fetch(docid="d1")
    assert first == second
    assert fetched == fetched_again
    assert inner.search_count == 1
    assert inner.fetch_count == 1
    assert tools.cache_hits == 2
    assert len(tools.calls) == 4
    assert tools.consumed == 4


def test_exact_reuse_counts_cache_hits_against_logical_budget() -> None:
    tools = ExactReuseSearchTools(FakeTools(), max_tool_calls=1)  # type: ignore[arg-type]
    tools.search(query="same")
    try:
        tools.search(query="same")
    except Exception as exc:
        assert "Tool budget exceeded" in str(exc)
    else:
        raise AssertionError("cache hit bypassed logical tool budget")
