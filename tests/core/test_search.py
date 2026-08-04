from __future__ import annotations

import pytest

from graphptc.search import TavilySearchTools, _deduplicate


def test_batch_input_limit_is_explicit_instead_of_truncating() -> None:
    values = [f"query-{index}" for index in range(21)]

    with pytest.raises(ValueError, match="At most 20"):
        _deduplicate(values, maximum=20)


def test_fetch_keeps_content_beyond_old_fifty_thousand_character_limit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    content = "x" * 60_000

    class FakeClient:
        def extract(self, **kwargs: object) -> dict[str, object]:
            return {
                "results": [{"url": "https://example.com", "raw_content": content}],
                "failed_results": [],
            }

    monkeypatch.setattr("graphptc.search.TavilyClient", lambda api_key: FakeClient())
    tools = TavilySearchTools("test-key")

    result = tools.fetch_url("https://example.com")

    assert len(result["content"]) == 60_000
    assert result["truncated"] is False
