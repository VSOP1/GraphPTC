from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from typing import Any, Iterable


def project_research_graph(events: Iterable[dict[str, Any]]) -> dict[str, Any]:
    values = list(events)
    episode_ids = {str(event["episode_id"]) for event in values}
    if len(episode_ids) != 1:
        raise ValueError("research projection requires one episode")
    episode_id = next(iter(episode_ids))
    nodes: list[dict[str, Any]] = []
    edges: list[dict[str, Any]] = []
    queries: Counter[str] = Counter()
    fetched: Counter[str] = Counter()
    seen_docids: set[str] = set()
    repeated_docids = 0
    search_count = 0
    fetch_count = 0
    query_ordinal = 0
    block_ids: set[str] = set()
    document_ids: set[str] = set()

    def add_node(node_id: str, kind: str, data: dict[str, Any]) -> None:
        nodes.append({"id": node_id, "kind": kind, "data": data})

    def add_edge(edge_type: str, source: str, target: str) -> None:
        edges.append({"type": edge_type, "source": source, "target": target})

    add_node(f"research:{episode_id}", "RESEARCH_EPISODE", {})
    for event in values:
        if event.get("type") != "tool.called":
            continue
        data = event.get("data", {})
        tool = str(data.get("tool", ""))
        arguments = data.get("arguments") or {}
        block_id = str(event.get("block_id"))
        sequence = int(event.get("sequence", 0))
        if block_id not in block_ids:
            block_ids.add(block_id)
            add_node(f"block:{block_id}", "BLOCK", {})
        if tool == "search":
            query = _normalize_query(arguments.get("query"))
            query_ordinal += 1
            search_count += 1
            queries[query] += 1
            query_id = f"query:{sequence}"
            result_id = f"result-set:{sequence}"
            add_node(query_id, "QUERY", {"text": query, "block_id": block_id})
            add_node(result_id, "RESULT_SET", {"block_id": block_id})
            add_edge("RETRIEVES", query_id, result_id)
            add_edge("OBSERVED_IN", result_id, f"block:{block_id}")
            result = data.get("result")
            for item in result if isinstance(result, list) else ():
                if not isinstance(item, dict) or item.get("docid") is None:
                    continue
                docid = str(item["docid"])
                doc_node = f"document:{docid}"
                if docid not in seen_docids:
                    seen_docids.add(docid)
                    document_ids.add(docid)
                    add_node(doc_node, "DOCUMENT", {"docid": docid})
                else:
                    repeated_docids += 1
                add_edge("CONTAINS", result_id, doc_node)
        elif tool == "fetch":
            fetch_count += 1
            docid = str(arguments.get("docid", ""))
            fetched[docid] += 1
            doc_node = f"document:{docid}"
            if docid not in document_ids:
                document_ids.add(docid)
                add_node(doc_node, "DOCUMENT", {"docid": docid})
            evidence_id = f"evidence:{sequence}"
            result = data.get("result")
            content = result.get("content", "") if isinstance(result, dict) else ""
            add_node(
                evidence_id,
                "EVIDENCE",
                {
                    "docid": docid,
                    "chars": len(str(content)),
                    "sha256": _sha256(content),
                    "block_id": block_id,
                },
            )
            add_edge("CONTAINS", doc_node, evidence_id)
            add_edge("OBSERVED_IN", evidence_id, f"block:{block_id}")

    repeated_queries = sum(count - 1 for count in queries.values() if count > 1)
    repeated_fetches = sum(count - 1 for count in fetched.values() if count > 1)
    return {
        "schema_version": 1,
        "episode_id": episode_id,
        "nodes": nodes,
        "edges": edges,
        "metrics": {
            "search_count": search_count,
            "unique_queries": len(queries),
            "repeated_queries": repeated_queries,
            "fetch_count": fetch_count,
            "unique_fetched_docids": len(fetched),
            "repeated_fetches": repeated_fetches,
            "unique_docids": len(seen_docids),
            "repeated_result_docids": repeated_docids,
        },
    }


def _normalize_query(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "").strip().lower())


def _sha256(value: Any) -> str:
    if isinstance(value, str):
        payload = value.encode("utf-8")
    else:
        payload = json.dumps(
            value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()
