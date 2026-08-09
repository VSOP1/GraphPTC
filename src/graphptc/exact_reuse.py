from __future__ import annotations

import re
import time
from typing import Any

from .local_search import OfficialCorpusSearchTools
from .search import ToolBudgetExceeded


class ExactReuseSearchTools:
    """Semantic-preserving per-episode cache for exact search/fetch calls."""

    def __init__(self, inner: OfficialCorpusSearchTools, *, max_tool_calls: int = 1_000) -> None:
        self._inner = inner
        if max_tool_calls < 1:
            raise ValueError("max_tool_calls must be positive")
        self._max_tool_calls = max_tool_calls
        self._consumed = 0
        self._search_cache: dict[str, list[dict[str, Any]]] = {}
        self._fetch_cache: dict[str, dict[str, Any]] = {}
        self._calls: list[dict[str, Any]] = []
        self._sequence = 0

    @property
    def calls(self) -> list[dict[str, Any]]:
        return [dict(call) for call in self._calls]

    @property
    def live_calls(self) -> list[dict[str, Any]]:
        return self._inner.calls

    @property
    def consumed(self) -> int:
        return self._consumed

    @property
    def cache_hits(self) -> int:
        return sum(bool(call.get("cache_hit")) for call in self._calls)

    def metadata(self) -> dict[str, Any]:
        return self._inner.metadata()

    def search(self, *, query: str) -> list[dict[str, Any]]:
        self._reserve()
        key = _normalize_query(query)
        started = time.perf_counter()
        if key in self._search_cache:
            result = [dict(item) for item in self._search_cache[key]]
            self._record("search", query=query, docids=result, started=started, cache_hit=True)
            return result
        result = self._inner.search(query=query)
        self._search_cache[key] = [dict(item) for item in result]
        self._record("search", query=query, docids=result, started=started, cache_hit=False)
        return result

    def fetch(self, *, docid: str) -> dict[str, Any]:
        self._reserve()
        key = str(docid)
        started = time.perf_counter()
        if key in self._fetch_cache:
            result = dict(self._fetch_cache[key])
            self._record("fetch", docid=key, docids=[key], started=started, cache_hit=True)
            return result
        result = self._inner.fetch(docid=key)
        self._fetch_cache[key] = dict(result)
        self._record("fetch", docid=key, docids=[key], started=started, cache_hit=False)
        return result

    def _record(
        self,
        operation: str,
        *,
        started: float,
        cache_hit: bool,
        query: str | None = None,
        docid: str | None = None,
        docids: list[dict[str, Any]] | list[str],
    ) -> None:
        self._sequence += 1
        ids = [
            str(item.get("docid")) if isinstance(item, dict) else str(item)
            for item in docids
        ]
        self._calls.append(
            {
                "sequence": self._sequence,
                "operation": operation,
                "item_count": 1,
                "duration_ms": (time.perf_counter() - started) * 1_000,
                "success": True,
                "query": query,
                "docid": docid,
                "docids": ids,
                "output_chars": 0,
                "error": None,
                "cache_hit": cache_hit,
            }
        )

    def _reserve(self) -> None:
        self._consumed += 1
        if self._consumed > self._max_tool_calls:
            raise ToolBudgetExceeded(
                f"Tool budget exceeded: {self._consumed} > {self._max_tool_calls}"
            )


def _normalize_query(value: str) -> str:
    return re.sub(r"\s+", " ", str(value).strip().lower())
