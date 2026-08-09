from __future__ import annotations

import hashlib
import json
import re
from collections import Counter, defaultdict
from typing import Any, Iterable

from graphptc.stage2_graph import DependencyGraph, build_dependency_graph


FETCH_CLASSES = (
    "answer_lexical_support",
    "stdout_lineage",
    "later_state_load",
    "unresolved",
)

QUERY_CLASSES = (
    "answer_support_followup",
    "fetch_followup",
    "new_result_content",
    "model_visible_only",
    "unresolved",
)


def project_evidence_consumption(events: Iterable[dict[str, Any]]) -> dict[str, Any]:
    values = list(events)
    graph = build_dependency_graph(values)
    tool_nodes = {
        int(node.data["event_sequence"]): node
        for node in graph.nodes
        if node.type == "TOOL"
    }
    terminal = values[-1]["data"]
    answer = _extract_result(str(terminal.get("answer", "")))
    normalized_answer = _normalize(answer)

    fetches: list[dict[str, Any]] = []
    fetches_by_docid: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for event in values:
        data = event.get("data", {})
        if (
            event.get("type") != "tool.called"
            or data.get("tool") != "fetch"
            or data.get("success") is not True
        ):
            continue
        sequence = int(event["sequence"])
        node = tool_nodes[sequence]
        result = data.get("result")
        content = str(result.get("content", "")) if isinstance(result, dict) else ""
        docid = str(data.get("arguments", {}).get("docid", ""))
        stdout_lineage = _reaches_block_output(graph, node.id)
        later_state_load = _has_later_state_load(graph, node.id, sequence)
        answer_support = bool(
            normalized_answer
            and len(normalized_answer) >= 4
            and normalized_answer in _normalize(content)
        )
        classification = (
            "answer_lexical_support"
            if answer_support
            else "stdout_lineage"
            if stdout_lineage
            else "later_state_load"
            if later_state_load
            else "unresolved"
        )
        item = {
            "sequence": sequence,
            "block_id": event.get("block_id"),
            "docid": docid,
            "content_sha256": _sha256(content),
            "content_chars": len(content),
            "stdout_lineage": stdout_lineage,
            "later_state_load": later_state_load,
            "answer_lexical_support": answer_support,
            "classification": classification,
        }
        fetches.append(item)
        fetches_by_docid[docid].append(item)

    zero_novelty_queries = _zero_novelty_queries(
        values, graph, tool_nodes, fetches_by_docid
    )
    fetch_counts = Counter(item["classification"] for item in fetches)
    query_counts = Counter(item["classification"] for item in zero_novelty_queries)
    return {
        "schema_version": 1,
        "episode_id": graph.episode_id,
        "example_id": graph.task_id,
        "model_visible": False,
        "action_taken": None,
        "answer_fingerprint": {
            "sha256": _sha256(answer),
            "chars": len(answer),
        },
        "metrics": {
            "successful_fetches": len(fetches),
            "fetch_classifications": {
                name: fetch_counts[name] for name in FETCH_CLASSES
            },
            "zero_novelty_queries": len(zero_novelty_queries),
            "query_classifications": {
                name: query_counts[name] for name in QUERY_CLASSES
            },
        },
        "fetches": fetches,
        "zero_novelty_query_frontier": zero_novelty_queries,
    }


def _zero_novelty_queries(
    events: list[dict[str, Any]],
    graph: DependencyGraph,
    tool_nodes: dict[int, Any],
    fetches_by_docid: dict[str, list[dict[str, Any]]],
) -> list[dict[str, Any]]:
    seen_queries: set[str] = set()
    seen_docids: set[str] = set()
    seen_result_content: set[tuple[str, str]] = set()
    frontier = []
    for event in events:
        data = event.get("data", {})
        if event.get("type") != "tool.called" or data.get("tool") != "search":
            continue
        sequence = int(event["sequence"])
        query = _normalize(data.get("arguments", {}).get("query"))
        result = data.get("result")
        rows = [item for item in result if isinstance(item, dict)] if isinstance(result, list) else []
        docids = {
            str(item["docid"])
            for item in rows
            if item.get("docid") is not None
        }
        content_keys = {
            (str(item["docid"]), _sha256(_normalize(item.get("snippet", ""))))
            for item in rows
            if item.get("docid") is not None
        }
        is_zero_novelty = query not in seen_queries and bool(docids) and not (docids - seen_docids)
        if is_zero_novelty:
            later_fetches = [
                fetch
                for docid in docids
                for fetch in fetches_by_docid.get(docid, ())
                if fetch["sequence"] > sequence
            ]
            new_content_count = len(content_keys - seen_result_content)
            stdout_lineage = _reaches_block_output(graph, tool_nodes[sequence].id)
            supporting = sorted(
                {fetch["docid"] for fetch in later_fetches if fetch["answer_lexical_support"]}
            )
            later_docids = sorted({fetch["docid"] for fetch in later_fetches})
            classification = (
                "answer_support_followup"
                if supporting
                else "fetch_followup"
                if later_docids
                else "new_result_content"
                if new_content_count
                else "model_visible_only"
                if stdout_lineage
                else "unresolved"
            )
            frontier.append(
                {
                    "sequence": sequence,
                    "block_id": event.get("block_id"),
                    "query_sha256": _sha256(query),
                    "result_docids": sorted(docids),
                    "new_result_content_count": new_content_count,
                    "stdout_lineage": stdout_lineage,
                    "later_fetch_docids": later_docids,
                    "answer_supporting_later_fetch_docids": supporting,
                    "classification": classification,
                }
            )
        seen_queries.add(query)
        seen_docids.update(docids)
        seen_result_content.update(content_keys)
    return frontier


def _reaches_block_output(graph: DependencyGraph, source: str) -> bool:
    targets: dict[str, list[str]] = defaultdict(list)
    for edge in graph.edges:
        if edge.type == "DATA":
            targets[edge.source].append(edge.target)
    pending = list(targets[source])
    visited = set(pending)
    while pending:
        node_id = pending.pop()
        node = graph.node(node_id)
        if node.type == "OUTPUT" and node.data.get("scope") == "block":
            return True
        for target in targets[node_id]:
            if target not in visited:
                visited.add(target)
                pending.append(target)
    return False


def _has_later_state_load(graph: DependencyGraph, source: str, sequence: int) -> bool:
    state_ids = {
        edge.target
        for edge in graph.edges
        if edge.type == "DATA"
        and edge.source == source
        and graph.node(edge.target).type == "STATE"
    }
    return any(
        edge.type == "STATE"
        and edge.source in state_ids
        and int(graph.node(edge.target).data.get("started_sequence", 0)) > sequence
        for edge in graph.edges
    )


def _extract_result(value: str) -> str:
    matches = re.findall(r"<result>(.*?)</result>", value, flags=re.IGNORECASE | re.DOTALL)
    return matches[-1].strip() if matches else value.strip()


def _normalize(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "").strip().casefold())


def _sha256(value: Any) -> str:
    if isinstance(value, str):
        payload = value.encode("utf-8")
    else:
        payload = json.dumps(value, ensure_ascii=False, sort_keys=True).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()
