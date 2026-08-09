from __future__ import annotations

import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable


CAPSULE_FIELDS = (
    "search_calls",
    "fetch_calls",
    "unique_queries",
    "unique_docids",
    "repeated_queries",
    "repeated_docids",
    "zero_novelty_searches",
    "unfetched_docids",
    "remaining_tool_calls",
)


def project_capsule_consumption(
    events: Iterable[dict[str, Any]], *, max_tool_calls: int
) -> dict[str, Any]:
    states: dict[str, dict[str, Any]] = {}
    transitions: list[dict[str, Any]] = []
    episode_ids: set[str] = set()

    for event in events:
        task_id = str(event["task_id"])
        event_type = str(event["type"])
        state = states.setdefault(task_id, _new_state())
        if event_type == "episode.started":
            episode_ids.add(task_id)
        elif event_type == "block.started":
            state["current"] = {
                "block_id": str(event["block_id"]),
                "turn": int(event["data"]["turn"]),
                "code": str(event["data"].get("code", "")),
                "calls": [],
            }
        elif event_type == "tool.called":
            if state["current"] is None:
                raise ValueError(f"tool event without active block for task {task_id}")
            state["current"]["calls"].append(_call_from_event(event))
        elif event_type == "block.finished":
            block = state["current"]
            if block is None or block["block_id"] != str(event["block_id"]):
                raise ValueError(f"block finish mismatch for task {task_id}")
            action = _action(block, state)
            pending = state["pending"]
            if pending is not None:
                transitions.append(_transition(task_id, pending, action))
            _apply_calls(state, block["calls"])
            if event["data"].get("success") is True:
                state["successful_blocks"] += 1
                state["pending"] = {
                    "source_block_id": block["block_id"],
                    "source_turn": block["turn"],
                    "snapshot": _snapshot(state, max_tool_calls),
                }
            else:
                state["pending"] = None
            state["current"] = None
        elif event_type == "episode.finished":
            pending = state["pending"]
            if pending is not None:
                transitions.append(_transition(task_id, pending, _terminal_action()))
                state["pending"] = None

    return {
        "episode_count": len(episode_ids),
        "successful_blocks": sum(state["successful_blocks"] for state in states.values()),
        "transition_count": len(transitions),
        "transitions": transitions,
        "summary": summarize_transitions(transitions),
    }


def summarize_transitions(transitions: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "all": _summarize_slice(transitions),
        "signals": {
            "repeated_queries_positive": _summarize_slice(
                [item for item in transitions if item["snapshot"]["repeated_queries"] > 0]
            ),
            "zero_novelty_positive": _summarize_slice(
                [item for item in transitions if item["snapshot"]["zero_novelty_searches"] > 0]
            ),
            "unfetched_docids_positive": _summarize_slice(
                [item for item in transitions if item["snapshot"]["unfetched_docids"] > 0]
            ),
        },
    }


def load_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def _new_state() -> dict[str, Any]:
    return {
        "current": None,
        "pending": None,
        "queries": [],
        "seen_docids": set(),
        "fetched_docids": set(),
        "repeated_docids": 0,
        "zero_novelty": 0,
        "calls": 0,
        "search_calls": 0,
        "fetch_calls": 0,
        "successful_blocks": 0,
    }


def _call_from_event(event: dict[str, Any]) -> dict[str, Any]:
    data = event.get("data", {})
    tool = str(data.get("tool", ""))
    arguments = data.get("arguments") or {}
    result = data.get("result")
    if tool == "search":
        docids = [
            str(item["docid"])
            for item in (result if isinstance(result, list) else ())
            if isinstance(item, dict) and item.get("docid") is not None
        ]
        return {
            "operation": "search",
            "query": _normalize(arguments.get("query")),
            "docids": docids,
        }
    if tool == "fetch":
        return {
            "operation": "fetch",
            "docid": str(arguments.get("docid", "")),
            "docids": [str(arguments.get("docid", ""))],
        }
    return {"operation": tool, "docids": []}


def _action(block: dict[str, Any], state: dict[str, Any]) -> dict[str, Any]:
    known_queries = set(state["queries"])
    seen_docids = set(state["seen_docids"])
    fetched_docids = set(state["fetched_docids"])
    search_calls = 0
    fetch_calls = 0
    repeated_searches = 0
    zero_novelty = 0
    repeated_fetches = 0
    known_unfetched_fetches = 0
    for call in block["calls"]:
        if call["operation"] == "search":
            search_calls += 1
            repeated_searches += call["query"] in known_queries
            docids = set(call["docids"])
            zero_novelty += not (docids - seen_docids)
            known_queries.add(call["query"])
            seen_docids.update(docids)
        elif call["operation"] == "fetch":
            fetch_calls += 1
            repeated_fetches += call["docid"] in fetched_docids
            known_unfetched_fetches += (
                call["docid"] in seen_docids and call["docid"] not in fetched_docids
            )
            fetched_docids.add(call["docid"])
    lowered_code = block["code"].lower()
    return {
        "terminal": False,
        "next_block_id": block["block_id"],
        "next_turn": block["turn"],
        "search_calls": search_calls,
        "fetch_calls": fetch_calls,
        "exact_repeat_searches": repeated_searches,
        "zero_novelty_searches": zero_novelty,
        "repeated_fetches": repeated_fetches,
        "known_unfetched_fetches": known_unfetched_fetches,
        "capsule_field_mentions": [field for field in CAPSULE_FIELDS if field in lowered_code],
    }


def _apply_calls(state: dict[str, Any], calls: list[dict[str, Any]]) -> None:
    for call in calls:
        state["calls"] += 1
        if call["operation"] == "search":
            query = call["query"]
            docids = set(call["docids"])
            if query not in state["queries"] and docids and not (docids - state["seen_docids"]):
                state["zero_novelty"] += 1
            state["queries"].append(query)
            state["repeated_docids"] += len(docids & state["seen_docids"])
            state["seen_docids"].update(docids)
            state["search_calls"] += 1
        elif call["operation"] == "fetch":
            state["fetched_docids"].add(call["docid"])
            state["fetch_calls"] += 1


def _snapshot(state: dict[str, Any], max_tool_calls: int) -> dict[str, int]:
    queries = state["queries"]
    return {
        "search_calls": state["search_calls"],
        "fetch_calls": state["fetch_calls"],
        "unique_queries": len(set(queries)),
        "unique_docids": len(state["seen_docids"]),
        "repeated_queries": len(queries) - len(set(queries)),
        "repeated_docids": state["repeated_docids"],
        "zero_novelty_searches": state["zero_novelty"],
        "unfetched_docids": len(state["seen_docids"] - state["fetched_docids"]),
        "remaining_tool_calls": max(0, max_tool_calls - state["calls"]),
    }


def _transition(task_id: str, pending: dict[str, Any], action: dict[str, Any]) -> dict[str, Any]:
    return {"task_id": task_id, **pending, "next_action": action}


def _terminal_action() -> dict[str, Any]:
    return {
        "terminal": True,
        "next_block_id": None,
        "next_turn": None,
        "search_calls": 0,
        "fetch_calls": 0,
        "exact_repeat_searches": 0,
        "zero_novelty_searches": 0,
        "repeated_fetches": 0,
        "known_unfetched_fetches": 0,
        "capsule_field_mentions": [],
    }


def _summarize_slice(values: list[dict[str, Any]]) -> dict[str, Any]:
    actions = [item["next_action"] for item in values]
    searches = sum(item["search_calls"] for item in actions)
    fetches = sum(item["fetch_calls"] for item in actions)
    count = len(actions)
    return {
        "opportunities": count,
        "terminal_rate": _ratio(sum(item["terminal"] for item in actions), count),
        "mean_next_search_calls": _ratio(searches, count),
        "mean_next_fetch_calls": _ratio(fetches, count),
        "next_exact_repeat_search_rate": _ratio(
            sum(item["exact_repeat_searches"] for item in actions), searches
        ),
        "next_zero_novelty_search_rate": _ratio(
            sum(item["zero_novelty_searches"] for item in actions), searches
        ),
        "next_repeated_fetch_rate": _ratio(
            sum(item["repeated_fetches"] for item in actions), fetches
        ),
        "next_known_unfetched_fetch_rate": _ratio(
            sum(item["known_unfetched_fetches"] for item in actions), fetches
        ),
        "explicit_capsule_field_mention_rate": _ratio(
            sum(bool(item["capsule_field_mentions"]) for item in actions), count
        ),
    }


def _ratio(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def _normalize(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "").strip().lower())
