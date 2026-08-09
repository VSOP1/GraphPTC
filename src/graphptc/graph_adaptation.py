from __future__ import annotations

import json
import re
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping


@dataclass(frozen=True)
class BlockAssessment:
    search_calls: int
    all_queries_repeated: bool
    all_searches_zero_novelty: bool
    repeated_fetch_docids: tuple[str, ...]
    new_docids: tuple[str, ...]


@dataclass(frozen=True)
class GraphTrigger:
    reasons: tuple[str, ...]
    repeated_fetch_docids: tuple[str, ...]


@dataclass(frozen=True)
class FrontierItem:
    docid: str
    source_query: str
    source_turn: int
    source_call: int
    result_rank: int
    retrieval_count: int


@dataclass(frozen=True)
class GraphActionProposal:
    action: str
    reason_codes: tuple[str, ...]
    target_docids: tuple[str, ...]
    model_visible: bool = False
    action_taken: None = None


class GraphTriggerDetector:
    def detect(self, assessment: BlockAssessment) -> GraphTrigger | None:
        reasons = []
        if assessment.repeated_fetch_docids:
            reasons.append("repeated_fetch")
        if assessment.all_queries_repeated:
            reasons.append("all_queries_repeated")
        if assessment.all_searches_zero_novelty:
            reasons.append("all_searches_zero_novelty")
        if not reasons:
            return None
        return GraphTrigger(tuple(reasons), assessment.repeated_fetch_docids)


class ActionableFrontierBuilder:
    def __init__(self, *, max_items: int = 3) -> None:
        if max_items < 1:
            raise ValueError("max_items must be positive")
        self.max_items = max_items

    def build(
        self,
        documents: Mapping[str, Mapping[str, Any]],
        fetched_docids: set[str],
    ) -> tuple[FrontierItem, ...]:
        eligible = [
            item for docid, item in documents.items() if docid not in fetched_docids
        ]
        ranked = sorted(
            eligible,
            key=lambda item: (
                -int(item["source_call"]),
                int(item["result_rank"]),
                str(item["docid"]),
            ),
        )
        return tuple(
            FrontierItem(
                docid=str(item["docid"]),
                source_query=str(item["source_query"]),
                source_turn=int(item["source_turn"]),
                source_call=int(item["source_call"]),
                result_rank=int(item["result_rank"]),
                retrieval_count=int(item["retrieval_count"]),
            )
            for item in ranked[: self.max_items]
        )


class GraphActionPolicy:
    def propose(
        self,
        trigger: GraphTrigger,
        frontier: tuple[FrontierItem, ...],
    ) -> GraphActionProposal:
        if "repeated_fetch" in trigger.reasons:
            return GraphActionProposal(
                action="REUSE_FETCHED_ARTIFACT",
                reason_codes=trigger.reasons,
                target_docids=trigger.repeated_fetch_docids,
            )
        if "all_queries_repeated" in trigger.reasons:
            return GraphActionProposal(
                action="CHANGE_QUERY_DIRECTION",
                reason_codes=trigger.reasons,
                target_docids=(),
            )
        return GraphActionProposal(
            action="INSPECT_RECENT_LINEAGE",
            reason_codes=trigger.reasons,
            target_docids=tuple(item.docid for item in frontier),
        )


def project_shadow_adaptation(
    events: Iterable[dict[str, Any]], *, max_frontier_items: int = 3
) -> dict[str, Any]:
    detector = GraphTriggerDetector()
    builder = ActionableFrontierBuilder(max_items=max_frontier_items)
    policy = GraphActionPolicy()
    states: dict[str, dict[str, Any]] = {}
    proposals: list[dict[str, Any]] = []
    episode_ids: set[str] = set()
    successful_blocks = 0

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
            assessment = _assess_and_apply(block, state)
            if event["data"].get("success") is True:
                successful_blocks += 1
                trigger = detector.detect(assessment)
                if trigger is not None:
                    frontier = builder.build(state["documents"], state["fetched"])
                    proposal = policy.propose(trigger, frontier)
                    proposals.append(
                        {
                            "task_id": task_id,
                            "source_block_id": block["block_id"],
                            "source_turn": block["turn"],
                            "assessment": asdict(assessment),
                            "trigger": asdict(trigger),
                            "frontier": [asdict(item) for item in frontier],
                            "proposed_action": asdict(proposal),
                        }
                    )
            state["current"] = None

    actions = Counter(item["proposed_action"]["action"] for item in proposals)
    return {
        "schema_version": 1,
        "mode": "shadow-graph-guided-adaptation",
        "episode_count": len(episode_ids),
        "successful_blocks": successful_blocks,
        "triggered_blocks": len(proposals),
        "trigger_rate": len(proposals) / successful_blocks if successful_blocks else 0.0,
        "model_visible": False,
        "action_taken": None,
        "action_distribution": dict(actions),
        "proposals": proposals,
    }


def load_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def _new_state() -> dict[str, Any]:
    return {
        "current": None,
        "queries": Counter(),
        "documents": {},
        "fetched": set(),
        "call_ordinal": 0,
    }


def _call(event: dict[str, Any]) -> dict[str, Any]:
    data = event.get("data", {})
    arguments = data.get("arguments") or {}
    result = data.get("result")
    tool = str(data.get("tool", ""))
    if tool == "search":
        return {
            "operation": "search",
            "success": data.get("success") is True,
            "query": _normalize(arguments.get("query")),
            "docids": [
                str(item["docid"])
                for item in (result if isinstance(result, list) else ())
                if isinstance(item, dict) and item.get("docid") is not None
            ],
        }
    if tool == "fetch":
        return {
            "operation": "fetch",
            "success": data.get("success") is True,
            "docid": str(arguments.get("docid", "")),
        }
    return {"operation": tool, "success": data.get("success") is True}


def _assess_and_apply(block: dict[str, Any], state: dict[str, Any]) -> BlockAssessment:
    search_outcomes: list[tuple[bool, bool]] = []
    repeated_fetches: list[str] = []
    new_docids: list[str] = []
    for call in block["calls"]:
        if not call["success"]:
            continue
        state["call_ordinal"] += 1
        if call["operation"] == "search":
            query = call["query"]
            repeated = state["queries"][query] > 0
            unseen = [docid for docid in call["docids"] if docid not in state["documents"]]
            search_outcomes.append((repeated, not unseen))
            new_docids.extend(unseen)
            state["queries"][query] += 1
            for rank, docid in enumerate(call["docids"], start=1):
                previous = state["documents"].get(docid)
                state["documents"][docid] = {
                    "docid": docid,
                    "source_query": query,
                    "source_turn": block["turn"],
                    "source_call": state["call_ordinal"],
                    "result_rank": rank,
                    "retrieval_count": 1 + (0 if previous is None else previous["retrieval_count"]),
                }
        elif call["operation"] == "fetch":
            docid = call["docid"]
            if docid in state["fetched"]:
                repeated_fetches.append(docid)
            state["fetched"].add(docid)
    return BlockAssessment(
        search_calls=len(search_outcomes),
        all_queries_repeated=bool(search_outcomes) and all(item[0] for item in search_outcomes),
        all_searches_zero_novelty=bool(search_outcomes) and all(item[1] for item in search_outcomes),
        repeated_fetch_docids=tuple(dict.fromkeys(repeated_fetches)),
        new_docids=tuple(dict.fromkeys(new_docids)),
    )


def _normalize(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "").strip().lower())
