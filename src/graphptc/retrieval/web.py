from __future__ import annotations

import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from typing import Any, Iterable

from tavily import TavilyClient


@dataclass(frozen=True)
class SearchCall:
    operation: str
    item_count: int
    duration_ms: float
    success: bool
    error: str | None = None


class ToolBudgetExceeded(RuntimeError):
    pass


class TavilySearchTools:
    """Metered search functions exposed inside a PTC program."""

    def __init__(
        self,
        api_key: str,
        *,
        search_depth: str = "advanced",
        default_max_results: int = 10,
        max_tool_calls: int = 1_000,
        timeout_seconds: float = 60.0,
    ) -> None:
        self._client = TavilyClient(api_key=api_key)
        self._search_depth = search_depth
        self._default_max_results = default_max_results
        self._max_tool_calls = max_tool_calls
        self._timeout_seconds = timeout_seconds
        self._calls: list[SearchCall] = []
        self._consumed = 0
        self._lock = threading.Lock()

    @property
    def calls(self) -> list[dict[str, Any]]:
        with self._lock:
            return [asdict(call) for call in self._calls]

    @property
    def consumed(self) -> int:
        with self._lock:
            return self._consumed

    def search_web(self, query: str, max_results: int | None = None) -> list[dict[str, Any]]:
        """Search the web and return title, URL, relevant snippets, and score."""
        self._reserve(1)
        limit = self._validate_max_results(max_results)
        started = time.perf_counter()
        try:
            response = self._client.search(
                query=query,
                search_depth=self._search_depth,
                max_results=limit,
                include_answer=False,
                include_raw_content=False,
                chunks_per_source=3,
                timeout=self._timeout_seconds,
            )
            results = [
                {
                    "title": item.get("title", ""),
                    "url": item.get("url", ""),
                    "content": item.get("content", ""),
                    "score": item.get("score"),
                }
                for item in response.get("results", [])
            ]
            self._record("search", 1, started, True)
            return results
        except Exception as exc:
            self._record("search", 1, started, False, str(exc))
            raise

    def search_web_batch(
        self, queries: list[str], max_results: int | None = None
    ) -> dict[str, list[dict[str, Any]]]:
        """Run independent searches concurrently and map each query to its results."""
        clean_queries = _deduplicate(queries, maximum=20)
        self._reserve(len(clean_queries))
        limit = self._validate_max_results(max_results)

        def search(query: str) -> tuple[str, list[dict[str, Any]]]:
            started = time.perf_counter()
            try:
                response = self._client.search(
                    query=query,
                    search_depth=self._search_depth,
                    max_results=limit,
                    include_answer=False,
                    include_raw_content=False,
                    chunks_per_source=3,
                    timeout=self._timeout_seconds,
                )
                results = [
                    {
                        "title": item.get("title", ""),
                        "url": item.get("url", ""),
                        "content": item.get("content", ""),
                        "score": item.get("score"),
                    }
                    for item in response.get("results", [])
                ]
                self._record("search", 1, started, True)
                return query, results
            except Exception as exc:
                self._record("search", 1, started, False, str(exc))
                return query, [{"error": str(exc)}]

        with ThreadPoolExecutor(max_workers=min(8, len(clean_queries))) as pool:
            return dict(pool.map(search, clean_queries))

    def fetch_url(
        self, url: str, query: str = "", max_chars: int = 1_000_000
    ) -> dict[str, Any]:
        """Extract one web page, optionally reranking chunks for a query."""
        return self.fetch_urls([url], query=query, max_chars=max_chars)[0]

    def fetch_urls(
        self, urls: list[str], query: str = "", max_chars: int = 1_000_000
    ) -> list[dict[str, Any]]:
        """Extract up to 20 URLs in one provider request."""
        clean_urls = _deduplicate(urls, maximum=20)
        self._reserve(len(clean_urls))
        max_chars = max(1_000, min(max_chars, 1_000_000))
        started = time.perf_counter()
        try:
            kwargs: dict[str, Any] = {
                "urls": clean_urls,
                "extract_depth": "advanced",
                "format": "markdown",
                "timeout": self._timeout_seconds,
            }
            if query:
                kwargs.update(query=query, chunks_per_source=5)
            response = self._client.extract(**kwargs)
            results = [
                {
                    "url": item.get("url", ""),
                    "content": (item.get("raw_content") or "")[:max_chars],
                    "truncated": len(item.get("raw_content") or "") > max_chars,
                }
                for item in response.get("results", [])
            ]
            results.extend(
                {"url": item.get("url", ""), "error": item.get("error", "extract failed")}
                for item in response.get("failed_results", [])
            )
            self._record("fetch", len(clean_urls), started, True)
            return results
        except Exception as exc:
            self._record("fetch", len(clean_urls), started, False, str(exc))
            raise

    def _reserve(self, amount: int) -> None:
        if amount <= 0:
            raise ValueError("At least one query or URL is required")
        with self._lock:
            if self._consumed + amount > self._max_tool_calls:
                raise ToolBudgetExceeded(
                    f"Tool budget exceeded: {self._consumed + amount} > {self._max_tool_calls}"
                )
            self._consumed += amount

    def _record(
        self,
        operation: str,
        item_count: int,
        started: float,
        success: bool,
        error: str | None = None,
    ) -> None:
        call = SearchCall(
            operation=operation,
            item_count=item_count,
            duration_ms=(time.perf_counter() - started) * 1_000,
            success=success,
            error=error,
        )
        with self._lock:
            self._calls.append(call)

    def _validate_max_results(self, value: int | None) -> int:
        return max(1, min(value or self._default_max_results, 20))


def _deduplicate(values: Iterable[str], *, maximum: int) -> list[str]:
    result = list(dict.fromkeys(value.strip() for value in values if value.strip()))
    if not result:
        raise ValueError("At least one non-empty value is required")
    if len(result) > maximum:
        raise ValueError(f"At most {maximum} unique values are allowed per batch")
    return result
