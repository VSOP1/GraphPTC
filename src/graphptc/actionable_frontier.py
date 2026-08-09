from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path
from typing import Any, Iterable


def project_actionable_frontier(
    events: Iterable[dict[str, Any]], *, max_items: int
) -> dict[str, Any]:
    states: dict[str, dict[str, Any]] = {}
    opportunities: list[dict[str, Any]] = []
    episode_ids: set[str] = set()
    successful_blocks = 0
    triggered_blocks = 0

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
                "calls": [],
            }
        elif event_type == "tool.called":
            if state["current"] is None:
                raise ValueError(f"tool event without active block for {task_id}")
            state["current"]["calls"].append(_call(event))
        elif event_type == "block.finished":
            block = state["current"]
            if block is None or block["block_id"] != str(event["block_id"]):
                raise ValueError(f"block finish mismatch for {task_id}")
            if state["pending"] is not None:
                opportunities.append(
                    {**state["pending"], "next_action": _next_action(block, state)}
                )
            reasons = _apply_block(block, state)
            if event["data"].get("success") is True:
                successful_blocks += 1
                if reasons:
                    triggered_blocks += 1
                frontiers = _ranked_frontiers(state, max_items)
                state["pending"] = (
                    {
                        "task_id": task_id,
                        "source_block_id": block["block_id"],
                        "source_turn": block["turn"],
                        "trigger_reasons": reasons,
                        "frontiers": frontiers,
                    }
                    if reasons and frontiers["graph"]
                    else None
                )
            else:
                state["pending"] = None
            state["current"] = None
        elif event_type == "episode.finished" and state["pending"] is not None:
            opportunities.append(
                {**state["pending"], "next_action": _terminal_action()}
            )
            state["pending"] = None

    return {
        "episode_count": len(episode_ids),
        "successful_blocks": successful_blocks,
        "triggered_blocks": triggered_blocks,
        "trigger_rate": _ratio(triggered_blocks, successful_blocks),
        "actionable_opportunities": len(opportunities),
        "opportunities": opportunities,
        "summary": summarize_frontier(opportunities),
    }


def summarize_frontier(opportunities: list[dict[str, Any]]) -> dict[str, Any]:
    target_opportunities = [
        item for item in opportunities
        if item["next_action"]["eligible_first_time_fetches"]
    ]
    rankings = ("graph", "recency", "first_seen")
    return {
        "target_opportunities": len(target_opportunities),
        "terminal_opportunities": sum(item["next_action"]["terminal"] for item in opportunities),
        "ranking": {
            ranking: _ranking_summary(target_opportunities, ranking)
            for ranking in rankings
        },
        "trigger_reasons": dict(Counter(
            reason for item in opportunities for reason in item["trigger_reasons"]
        )),
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
        "queries": Counter(),
        "documents": {},
        "fetched": set(),
        "call_ordinal": 0,
    }


def _call(event: dict[str, Any]) -> dict[str, Any]:
    data = event.get("data", {})
    tool = str(data.get("tool", ""))
    arguments = data.get("arguments") or {}
    result = data.get("result")
    if tool == "search":
        return {
            "operation": "search",
            "query": _normalize(arguments.get("query")),
            "docids": [
                str(item["docid"])
                for item in (result if isinstance(result, list) else ())
                if isinstance(item, dict) and item.get("docid") is not None
            ],
        }
    if tool == "fetch":
        return {"operation": "fetch", "docid": str(arguments.get("docid", ""))}
    return {"operation": tool}


def _apply_block(block: dict[str, Any], state: dict[str, Any]) -> list[str]:
    reasons: list[str] = []
    for call in block["calls"]:
        state["call_ordinal"] += 1
        if call["operation"] == "search":
            query = call["query"]
            docids = call["docids"]
            if state["queries"][query] > 0:
                reasons.append("exact_query_repeat")
            if not (set(docids) - set(state["documents"])):
                reasons.append("zero_novelty_search")
            state["queries"][query] += 1
            for rank, docid in enumerate(docids, start=1):
                item = state["documents"].setdefault(
                    docid,
                    {
                        "docid": docid,
                        "retrieval_count": 0,
                        "first_seen_call": state["call_ordinal"],
                        "last_seen_call": state["call_ordinal"],
                        "best_rank": rank,
                        "source_query": query,
                        "source_turn": block["turn"],
                    },
                )
                item["retrieval_count"] += 1
                item["last_seen_call"] = state["call_ordinal"]
                if rank < item["best_rank"]:
                    item["best_rank"] = rank
                    item["source_query"] = query
                    item["source_turn"] = block["turn"]
        elif call["operation"] == "fetch":
            if call["docid"] in state["fetched"]:
                reasons.append("repeated_fetch")
            state["fetched"].add(call["docid"])
    return list(dict.fromkeys(reasons))


def _ranked_frontiers(state: dict[str, Any], max_items: int) -> dict[str, list[dict[str, Any]]]:
    eligible = [
        dict(item)
        for docid, item in state["documents"].items()
        if docid not in state["fetched"]
    ]
    keys = {
        "graph": lambda item: (
            -item["retrieval_count"],
            -item["last_seen_call"],
            item["best_rank"],
            item["docid"],
        ),
        "recency": lambda item: (-item["last_seen_call"], item["best_rank"], item["docid"]),
        "first_seen": lambda item: (item["first_seen_call"], item["best_rank"], item["docid"]),
    }
    return {
        name: sorted(eligible, key=key)[:max_items]
        for name, key in keys.items()
    }


def _next_action(block: dict[str, Any], state: dict[str, Any]) -> dict[str, Any]:
    fetched = set(state["fetched"])
    eligible = set(state["documents"]) - fetched
    first_time_fetches: list[str] = []
    repeated_fetches: list[str] = []
    for call in block["calls"]:
        if call["operation"] != "fetch":
            continue
        docid = call["docid"]
        if docid in fetched:
            repeated_fetches.append(docid)
        else:
            first_time_fetches.append(docid)
            fetched.add(docid)
    return {
        "terminal": False,
        "first_time_fetches": first_time_fetches,
        "eligible_first_time_fetches": [docid for docid in first_time_fetches if docid in eligible],
        "repeated_fetches": repeated_fetches,
    }


def _terminal_action() -> dict[str, Any]:
    return {
        "terminal": True,
        "first_time_fetches": [],
        "eligible_first_time_fetches": [],
        "repeated_fetches": [],
    }


def _ranking_summary(values: list[dict[str, Any]], ranking: str) -> dict[str, Any]:
    hits = 0
    covered_docids = 0
    total_docids = 0
    for item in values:
        targets = set(item["next_action"]["eligible_first_time_fetches"])
        frontier = {entry["docid"] for entry in item["frontiers"][ranking]}
        intersection = targets & frontier
        hits += bool(intersection)
        covered_docids += len(intersection)
        total_docids += len(targets)
    return {
        "opportunity_hits": hits,
        "opportunity_coverage": _ratio(hits, len(values)),
        "docid_coverage": _ratio(covered_docids, total_docids),
    }


def _ratio(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


def _normalize(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "").strip().lower())
