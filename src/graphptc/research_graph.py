from __future__ import annotations

import copy
import re
from collections import Counter
from typing import Any, Mapping


_ID_RE = re.compile(r"^[A-Za-z0-9_.:-]{1,80}$")
_RELATIONS = {"supports", "refutes"}


class ResearchGraphState:
    """Agent-authored, runtime-verified research state for one episode."""

    def __init__(self, tools: Any, *, task: str, max_items: int = 4) -> None:
        if max_items < 1:
            raise ValueError("max_items must be positive")
        self._tools = tools
        self._max_items = max_items
        self._nodes: dict[str, dict[str, Any]] = {}
        self._edges: list[dict[str, str]] = []
        self._artifacts: dict[str, Any] = {}
        self._fetched_content: dict[str, str] = {}
        self._query_count = 0
        self._fetch_count = 0
        self._action_count = 0
        self._node_cursor = 0
        self._edge_cursor = 0
        self._node_order: list[str] = []
        self._interface_calls: Counter[str] = Counter()
        self._reuse_hits = 0
        self._add_node("task", "TASK", {"question": str(task)[:1_000]})

    def search(self, *, query: str) -> list[dict[str, Any]]:
        """Search and record query, document, intent, and reusable artifact nodes."""
        target = self._current_action_target()
        prior_document_ids = self._target_document_ids(target)
        results = self._tools.search(query=query)
        self._query_count += 1
        query_id = f"query:{self._query_count}"
        artifact_id = f"artifact:search:{self._query_count}"
        result_docids = {
            self._document_id(str(item.get("docid", "")))
            for item in results
            if item.get("docid")
        }
        self._add_node(
            query_id,
            "QUERY",
            {
                "text": str(query)[:500],
                "result_count": len(result_docids),
                "new_documents_for_target": len(result_docids - prior_document_ids),
                "repeated_documents_for_target": len(result_docids & prior_document_ids),
            },
        )
        if target is not None:
            self._add_edge("intends_to_resolve", query_id, target)
        for rank, item in enumerate(results, start=1):
            docid = str(item.get("docid", ""))
            if not docid:
                continue
            document_id = self._document_id(docid)
            self._add_node(
                document_id,
                "DOCUMENT",
                {"docid": docid, "snippet": str(item.get("snippet", ""))[:240]},
            )
            self._add_edge("retrieves", query_id, document_id)
        self._artifacts[artifact_id] = copy.deepcopy(results)
        self._add_node(
            artifact_id,
            "ARTIFACT",
            {"operation": "search", "query": str(query)[:500]},
        )
        self._add_edge("produces", query_id, artifact_id)
        return results

    def fetch(self, *, docid: str) -> dict[str, Any]:
        """Fetch and retain a source artifact for verified evidence and later reuse."""
        result = self._tools.fetch(docid=docid)
        key = str(result.get("docid", docid))
        content = str(result.get("content", ""))
        self._fetch_count += 1
        artifact_id = f"artifact:fetch:{self._fetch_count}"
        document_id = self._document_id(key)
        self._add_node(
            document_id,
            "DOCUMENT",
            {"docid": key, "fetched": True},
        )
        self._fetched_content[key] = content
        self._artifacts[artifact_id] = copy.deepcopy(result)
        self._add_node(
            artifact_id,
            "ARTIFACT",
            {"operation": "fetch", "docid": key},
        )
        self._add_edge("materializes", document_id, artifact_id)
        target = self._current_action_target()
        if target is not None:
            self._add_edge("investigates", document_id, target)
        return result

    def graph_add_constraint(self, *, constraint_id: str, description: str) -> dict[str, Any]:
        """Declare a task constraint; identifiers are stable within the episode."""
        self._count_interface("graph_add_constraint")
        node_id = self._prefixed_id("constraint", constraint_id)
        text = str(description).strip()
        if not text:
            raise ValueError("constraint description must not be empty")
        self._add_declared_node(node_id, "CONSTRAINT", {"description": text[:500]})
        self._add_edge("requires", "task", node_id)
        return {"node_id": node_id, "verified": True}

    def graph_add_candidate(self, *, candidate_id: str, label: str) -> dict[str, Any]:
        """Declare a candidate answer without asserting that it is correct."""
        self._count_interface("graph_add_candidate")
        node_id = self._prefixed_id("candidate", candidate_id)
        text = str(label).strip()
        if not text:
            raise ValueError("candidate label must not be empty")
        self._add_declared_node(node_id, "CANDIDATE", {"label": text[:500]})
        self._add_edge("candidate_for", node_id, "task")
        return {"node_id": node_id, "verified": True}

    def graph_add_evidence(
        self,
        *,
        evidence_id: str,
        docid: str,
        quote: str,
        relation: str,
        target_id: str,
        constraint_id: str = "",
    ) -> dict[str, Any]:
        """Add evidence only when its quoted text exists in a fetched document."""
        self._count_interface("graph_add_evidence")
        key = str(docid).strip()
        quoted = str(quote).strip()
        if key not in self._fetched_content:
            raise ValueError(f"document {key!r} has not been fetched")
        if not quoted:
            raise ValueError("evidence quote must not be empty")
        if quoted not in self._fetched_content[key]:
            raise ValueError("evidence quote is not an exact span of the fetched document")
        edge_type = str(relation).strip().lower()
        if edge_type not in _RELATIONS:
            raise ValueError("evidence relation must be supports or refutes")
        target = self._resolve_id(target_id, {"CANDIDATE", "CONSTRAINT"})
        constraint = (
            self._resolve_id(constraint_id, {"CONSTRAINT"}) if constraint_id else None
        )
        node_id = self._prefixed_id("evidence", evidence_id)
        self._add_declared_node(
            node_id,
            "EVIDENCE",
            {"docid": key, "quote": quoted[:1_000], "verified": True},
        )
        self._add_edge("contains", self._document_id(key), node_id)
        self._add_edge(edge_type, node_id, target)
        if constraint is not None and constraint != target:
            self._add_edge("addresses", node_id, constraint)
        return {
            "node_id": node_id,
            "verified": True,
            "relation": edge_type,
            "target_id": target,
            "constraint_id": constraint,
        }

    def graph_frontier(self) -> dict[str, Any]:
        """Return unresolved constraints, conflicts, and retrieved-but-unfetched documents."""
        self._count_interface("graph_frontier")
        return self.frontier()

    def graph_trace(self, *, node_id: str) -> dict[str, Any]:
        """Return a bounded provenance neighborhood for an existing graph node."""
        self._count_interface("graph_trace")
        resolved = self._resolve_id(node_id)
        edges = [
            edge
            for edge in self._edges
            if edge["source"] == resolved or edge["target"] == resolved
        ][: self._max_items * 2]
        neighbor_ids = {resolved}
        for edge in edges:
            neighbor_ids.update((edge["source"], edge["target"]))
        return {
            "node": copy.deepcopy(self._nodes[resolved]),
            "neighbors": [
                self._compact_node(self._nodes[value])
                for value in neighbor_ids
                if value != resolved and value in self._nodes
            ][: self._max_items],
            "edges": copy.deepcopy(edges),
        }

    def graph_load_artifact(self, *, artifact_id: str) -> Any:
        """Load a persisted search or fetch result without a new external tool call."""
        self._count_interface("graph_load_artifact")
        key = self._resolve_id(artifact_id, {"ARTIFACT"})
        self._reuse_hits += 1
        return copy.deepcopy(self._artifacts[key])

    def graph_alternatives(self, *, target_id: str) -> dict[str, Any]:
        """List alternative candidates and their verified support/refutation counts."""
        self._count_interface("graph_alternatives")
        target = self._resolve_id(target_id, {"CANDIDATE"})
        return {
            "target_id": target,
            "alternatives": [
                self._candidate_summary(node_id)
                for node_id, node in self._nodes.items()
                if node["kind"] == "CANDIDATE" and node_id != target
            ][: self._max_items],
        }

    def set_action_target(self, target_id: str | None) -> None:
        self._action_target = target_id

    def record_action(
        self,
        *,
        action: str,
        target: str,
        expected_change: str,
        target_valid: bool,
        source: str,
    ) -> str:
        self._action_count += 1
        node_id = f"action:{self._action_count}"
        self._add_node(
            node_id,
            "ACTION",
            {
                "action": action,
                "target": target or None,
                "expected_change": expected_change[:500] or None,
                "target_valid": target_valid,
                "source": source,
            },
        )
        if target_valid:
            self._add_edge("targets", node_id, self._resolve_id(target))
        return node_id

    def has_node(self, node_id: str) -> bool:
        try:
            self._resolve_id(node_id)
        except ValueError:
            return False
        return True

    def frontier(self) -> dict[str, Any]:
        constraints = []
        for node_id, node in self._nodes.items():
            if node["kind"] != "CONSTRAINT":
                continue
            incoming = [edge["type"] for edge in self._edges if edge["target"] == node_id]
            evidence_relations = [
                value for value in incoming if value in {"supports", "refutes", "addresses"}
            ]
            if not evidence_relations:
                constraints.append(self._compact_node(node))
        conflicts = [
            self._candidate_summary(node_id)
            for node_id, node in self._nodes.items()
            if node["kind"] == "CANDIDATE"
            and self._candidate_counts(node_id)["supports"]
            and self._candidate_counts(node_id)["refutes"]
        ]
        unfetched = [
            self._compact_node(node)
            for node_id, node in self._nodes.items()
            if node["kind"] == "DOCUMENT"
            and str(node["data"].get("docid", "")) not in self._fetched_content
        ]
        return {
            "unresolved_constraints": constraints[: self._max_items],
            "conflicted_candidates": conflicts[: self._max_items],
            "unfetched_documents": unfetched[: self._max_items],
            "reusable_artifacts": list(self._artifacts)[-self._max_items :],
        }

    def delta(self) -> dict[str, Any]:
        new_ids = self._node_order[self._node_cursor :]
        new_edges = self._edges[self._edge_cursor :]
        self._node_cursor = len(self._node_order)
        self._edge_cursor = len(self._edges)
        return {
            "new_nodes": [self._compact_node(self._nodes[node_id]) for node_id in new_ids],
            "new_edges": copy.deepcopy(new_edges),
            "frontier": self.frontier(),
        }

    def telemetry(self) -> dict[str, Any]:
        kinds = Counter(node["kind"] for node in self._nodes.values())
        return {
            "node_count": len(self._nodes),
            "edge_count": len(self._edges),
            "node_kinds": dict(kinds),
            "interface_calls": dict(self._interface_calls),
            "artifact_count": len(self._artifacts),
            "artifact_reuse_hits": self._reuse_hits,
            "verified_evidence": kinds["EVIDENCE"],
            "frontier": self.frontier(),
        }

    def interface_counts(self) -> dict[str, int]:
        return dict(self._interface_calls)

    def action_targets(self) -> dict[str, list[str]]:
        frontier = self.frontier()
        inspect = [
            node_id
            for node_id, node in self._nodes.items()
            if node["kind"] in {"CONSTRAINT", "CANDIDATE"}
        ][-self._max_items :]
        continue_targets = [item["id"] for item in frontier["unresolved_constraints"]]
        current = self._current_action_target()
        if current in continue_targets:
            continue_targets = [current, *[item for item in continue_targets if item != current]]
        if not continue_targets:
            continue_targets = ["task"]
        return {
            "ANSWER": ["task"],
            "CONTINUE": continue_targets,
            "INSPECT": inspect or ["task"],
            "PATCH": [self._current_action_target() or "task"],
            "REUSE_REPLAY": list(frontier["reusable_artifacts"]),
        }

    def answer_context(self) -> dict[str, Any]:
        frontier = self.frontier()
        candidates = [
            {
                "id": node_id,
                **self._candidate_counts(node_id),
            }
            for node_id, node in self._nodes.items()
            if node["kind"] == "CANDIDATE"
        ][-self._max_items :]
        return {
            "candidates": candidates,
            "unresolved_constraints": [
                item["id"] for item in frontier["unresolved_constraints"]
            ],
        }


    def target_context(self, target_id: str) -> dict[str, Any]:
        """Return bounded research lineage for the selected action target."""
        target = self._resolve_id(target_id)
        all_query_ids = [
            edge["source"]
            for edge in self._edges
            if edge["type"] == "intends_to_resolve" and edge["target"] == target
        ]
        query_ids = all_query_ids[-self._max_items :]
        query_id_set = set(query_ids)
        source_queries_by_document: dict[str, list[str]] = {}
        for edge in self._edges:
            if edge["source"] not in query_id_set:
                continue
            if edge["type"] == "retrieves":
                source_queries_by_document.setdefault(edge["target"], []).append(
                    edge["source"]
                )
        for edge in self._edges:
            if edge["type"] == "investigates" and edge["target"] == target:
                source_queries_by_document.setdefault(edge["source"], [])
        unfetched_documents = []
        for document_id, source_query_ids in source_queries_by_document.items():
            node = self._nodes.get(document_id)
            if node is None:
                continue
            docid = str(node["data"].get("docid", ""))
            item = self._compact_node(node)
            item["source_query_ids"] = source_query_ids[-self._max_items :]
            if docid not in self._fetched_content:
                unfetched_documents.append(item)
        evidence_ids = [
            edge["source"]
            for edge in self._edges
            if edge["type"] in {"supports", "refutes", "addresses"}
            and edge["target"] == target
            and self._nodes.get(edge["source"], {}).get("kind") == "EVIDENCE"
        ]
        all_document_ids = self._target_document_ids(target)
        attempts = []
        for query_id in query_ids:
            query_node = self._nodes.get(query_id)
            if query_node is None:
                continue
            artifacts = [
                edge["target"]
                for edge in self._edges
                if edge["type"] == "produces" and edge["source"] == query_id
            ]
            attempts.append(
                {
                    "query_id": query_id,
                    "query": query_node["data"].get("text", ""),
                    "result_count": query_node["data"].get("result_count", 0),
                    "new_documents": query_node["data"].get(
                        "new_documents_for_target", 0
                    ),
                    "repeated_documents": query_node["data"].get(
                        "repeated_documents_for_target", 0
                    ),
                    "artifact_id": artifacts[-1] if artifacts else None,
                }
            )
        fetched_count = sum(
            str(self._nodes[node_id]["data"].get("docid", "")) in self._fetched_content
            for node_id in all_document_ids
            if node_id in self._nodes
        )
        return {
            "target": self._compact_node(self._nodes[target]),
            "retrieval_memory": {
                "attempt_count": len(all_query_ids),
                "recent_attempts": attempts,
                "coverage": {
                    "retrieved_documents": len(all_document_ids),
                    "fetched_documents": fetched_count,
                    "unfetched_documents": len(all_document_ids) - fetched_count,
                },
            },
            "unfetched_documents": unfetched_documents[: self._max_items],
            "evidence": [
                self._compact_node(self._nodes[evidence_id])
                for evidence_id in dict.fromkeys(evidence_ids)
            ][: self._max_items],
        }

    def _target_document_ids(self, target_id: str | None) -> set[str]:
        if target_id is None:
            return set()
        query_ids = {
            edge["source"]
            for edge in self._edges
            if edge["type"] == "intends_to_resolve" and edge["target"] == target_id
        }
        document_ids = {
            edge["target"]
            for edge in self._edges
            if edge["type"] == "retrieves" and edge["source"] in query_ids
        }
        document_ids.update(
            edge["source"]
            for edge in self._edges
            if edge["type"] == "investigates" and edge["target"] == target_id
        )
        return document_ids

    def _current_action_target(self) -> str | None:
        value = getattr(self, "_action_target", None)
        return value if value in self._nodes else None

    def _candidate_counts(self, node_id: str) -> dict[str, int]:
        values = Counter(
            edge["type"]
            for edge in self._edges
            if edge["target"] == node_id and edge["type"] in _RELATIONS
        )
        return {"supports": values["supports"], "refutes": values["refutes"]}

    def _candidate_summary(self, node_id: str) -> dict[str, Any]:
        return {**self._compact_node(self._nodes[node_id]), **self._candidate_counts(node_id)}

    def _add_node(self, node_id: str, kind: str, data: Mapping[str, Any]) -> None:
        existing = self._nodes.get(node_id)
        if existing is None:
            self._nodes[node_id] = {"id": node_id, "kind": kind, "data": dict(data)}
            self._node_order.append(node_id)
            return
        if existing["kind"] != kind:
            raise ValueError(f"graph node {node_id!r} already has another kind")
        existing["data"].update({key: value for key, value in data.items() if value is not None})

    def _add_declared_node(
        self, node_id: str, kind: str, data: Mapping[str, Any]
    ) -> None:
        existing = self._nodes.get(node_id)
        expected = dict(data)
        if existing is None:
            self._add_node(node_id, kind, expected)
            return
        if existing["kind"] == kind and existing["data"] == expected:
            return
        if existing["kind"] != kind or existing["data"] != expected:
            raise ValueError(f"graph node {node_id!r} cannot be redefined")

    def _add_edge(self, edge_type: str, source: str, target: str) -> None:
        edge = {"type": edge_type, "source": source, "target": target}
        if edge not in self._edges:
            self._edges.append(edge)

    def _resolve_id(self, value: str, kinds: set[str] | None = None) -> str:
        key = str(value).strip()
        candidates = [key]
        if ":" not in key:
            candidates.extend((f"constraint:{key}", f"candidate:{key}", f"evidence:{key}"))
        for candidate in candidates:
            node = self._nodes.get(candidate)
            if node is not None and (kinds is None or node["kind"] in kinds):
                return candidate
        raise ValueError(f"unknown or incompatible graph node {key!r}")

    @staticmethod
    def _prefixed_id(prefix: str, value: str) -> str:
        key = str(value).strip()
        if not _ID_RE.fullmatch(key):
            raise ValueError("graph identifiers must be 1-80 ASCII letters, digits, or ._:-")
        return key if key.startswith(f"{prefix}:") else f"{prefix}:{key}"

    @staticmethod
    def _document_id(docid: str) -> str:
        return f"document:{docid}"

    @staticmethod
    def _compact_node(node: Mapping[str, Any]) -> dict[str, Any]:
        data = copy.deepcopy(node["data"])
        if node["kind"] == "EVIDENCE":
            data["quote"] = str(data.get("quote", ""))[:240]
        elif node["kind"] == "DOCUMENT":
            data["excerpt"] = str(data.get("excerpt", ""))[:240]
        return {
            "id": node["id"],
            "kind": node["kind"],
            "data": data,
        }

    def _count_interface(self, name: str) -> None:
        self._interface_calls[name] += 1
