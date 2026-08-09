from __future__ import annotations

import re
from collections.abc import Mapping
from typing import Any


class GraphProgressView:
    """Bounded, read-only progress view over the current task's tool ledger."""

    def __init__(self, tools: Any, *, mode: str, max_tool_calls: int, target_chars: int = 512) -> None:
        if mode not in {"placebo", "graph"}:
            raise ValueError(f"unsupported graph progress mode: {mode!r}")
        self._tools = tools
        self._mode = mode
        self._max_tool_calls = max_tool_calls
        self._target_chars = target_chars
        self._snapshot_calls = 0

    def graph_progress(self) -> dict[str, Any]:
        return _pad_payload(self._snapshot(), self._target_chars)

    def capsule(self) -> str:
        return _capsule(self._snapshot(), self._target_chars)

    def telemetry(self) -> dict[str, Any]:
        return {
            "mode": self._mode,
            "snapshot_calls": self._snapshot_calls,
            "target_chars": self._target_chars,
        }

    def _snapshot(self) -> dict[str, Any]:
        self._snapshot_calls += 1
        calls = list(self._tools.calls)
        if self._mode == "graph":
            return self._graph_payload(calls)
        return {
            "search_calls": 0,
            "fetch_calls": 0,
            "unique_queries": 0,
            "unique_docids": 0,
            "repeated_queries": 0,
            "repeated_docids": 0,
            "zero_novelty_searches": 0,
            "unfetched_docids": 0,
            "remaining_tool_calls": self._max_tool_calls,
        }

    def _graph_payload(self, calls: list[Mapping[str, Any]]) -> dict[str, Any]:
        queries: list[str] = []
        seen_docids: set[str] = set()
        fetched_docids: set[str] = set()
        repeated_docids = 0
        zero_novelty = 0
        search_calls = 0
        fetch_calls = 0
        for call in calls:
            operation = call.get("operation")
            docids = {str(value) for value in call.get("docids", ())}
            if operation == "search":
                search_calls += 1
                query = _normalize(call.get("query"))
                if query in queries:
                    pass
                elif docids and not (docids - seen_docids):
                    zero_novelty += 1
                queries.append(query)
                repeated_docids += len(docids & seen_docids)
                seen_docids.update(docids)
            elif operation == "fetch":
                fetch_calls += 1
                fetched_docids.update(docids or {str(call.get("docid", ""))})
        consumed = int(getattr(self._tools, "consumed", len(calls)))
        return {
            "search_calls": search_calls,
            "fetch_calls": fetch_calls,
            "unique_queries": len(set(queries)),
            "unique_docids": len(seen_docids),
            "repeated_queries": len(queries) - len(set(queries)),
            "repeated_docids": repeated_docids,
            "zero_novelty_searches": zero_novelty,
            "unfetched_docids": len(seen_docids - fetched_docids),
            "remaining_tool_calls": max(0, self._max_tool_calls - consumed),
        }


def _pad_payload(payload: dict[str, Any], target_chars: int) -> dict[str, Any]:
    payload = dict(payload)
    payload["padding"] = ""
    while len(repr(payload)) < target_chars:
        payload["padding"] += "x"
    while len(repr(payload)) > target_chars:
        payload["padding"] = payload["padding"][:-1]
    return payload


def _capsule(payload: dict[str, Any], target_chars: int) -> str:
    prefix = "GRAPH_PROGRESS_SNAPSHOT "
    value = dict(payload)
    value["padding"] = ""
    rendered = prefix + _compact_json(value)
    while len(rendered) < target_chars:
        value["padding"] += "x"
        rendered = prefix + _compact_json(value)
    while len(rendered) > target_chars:
        value["padding"] = value["padding"][:-1]
        rendered = prefix + _compact_json(value)
    return rendered


def _compact_json(value: dict[str, Any]) -> str:
    import json

    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def _normalize(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "").strip().lower())
