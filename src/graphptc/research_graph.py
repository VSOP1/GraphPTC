from __future__ import annotations

import copy
import re
from collections import Counter
from typing import Any, Mapping

from .episode_graph import EpisodeGraph
from .tool_effects import ToolEffectContract, ToolGraphRuntime


_ID_RE = re.compile(r"^[A-Za-z0-9_.:-]{1,80}$")
_RELATIONS = {"supports", "refutes"}


class RetrievalGraphProjection:
    """Retrieval semantics projected onto the domain-neutral episode graph."""

    def __init__(
        self,
        tools: Any,
        *,
        task: str,
        max_items: int = 4,
        graph: EpisodeGraph | None = None,
    ) -> None:
        if max_items < 1:
            raise ValueError("max_items must be positive")
        self._tools = tools
        self._max_items = max_items
        self._episode_graph = graph or EpisodeGraph(task=task)
        self._nodes = self._episode_graph.nodes
        self._edges = self._episode_graph.edges
        self._artifacts = self._episode_graph.artifacts
        self._tool_runtime = ToolGraphRuntime(self._episode_graph)
        self._tool_runtime.register(
            tools.search,
            ToolEffectContract(
                name="search",
                effect="read",
                deterministic=False,
                cacheable=False,
                artifact_kind="search_result",
            ),
        )
        self._tool_runtime.register(
            tools.fetch,
            ToolEffectContract(
                name="fetch",
                effect="read",
                deterministic=True,
                cacheable=True,
                artifact_kind="fetched_resource",
            ),
        )
        self._fetched_content: dict[str, str] = {}
        self._fetched_results: dict[str, dict[str, Any]] = {}
        self._query_count = 0
        self._fetch_count = 0
        self._action_count = 0
        self._interface_calls: Counter[str] = Counter()
        self._reuse_hits = 0
        self._task_graph_initialized = False
        self._add_node("task", "TASK", {"question": str(task)[:1_000]})

    @property
    def episode_graph(self) -> EpisodeGraph:
        return self._episode_graph

    def initialize_task_graph(
        self, requirements: list[Mapping[str, Any]]
    ) -> dict[str, Any]:
        """Install the model's one-time requirement decomposition."""
        if self._task_graph_initialized:
            raise ValueError("task graph has already been initialized")
        if not requirements:
            raise ValueError("task graph must contain at least one requirement")

        normalized: list[tuple[str, str, list[str]]] = []
        ids: set[str] = set()
        for item in requirements:
            node_id = self._prefixed_id("constraint", str(item.get("id", "")))
            description = str(item.get("description", "")).strip()
            if not description:
                raise ValueError("requirement description must not be empty")
            if node_id in ids:
                raise ValueError(f"duplicate requirement {node_id!r}")
            ids.add(node_id)
            dependencies = [
                self._prefixed_id("constraint", str(value))
                for value in item.get("depends_on", ())
            ]
            normalized.append((node_id, description, dependencies))

        for node_id, _description, dependencies in normalized:
            unknown = [value for value in dependencies if value not in ids]
            if unknown:
                raise ValueError(
                    f"requirement {node_id!r} has unknown dependencies {unknown!r}"
                )
            if node_id in dependencies:
                raise ValueError(f"requirement {node_id!r} cannot depend on itself")

        dependency_map = {node_id: dependencies for node_id, _, dependencies in normalized}
        visiting: set[str] = set()
        visited: set[str] = set()

        def visit(node_id: str) -> None:
            if node_id in visiting:
                raise ValueError("task graph dependencies must be acyclic")
            if node_id in visited:
                return
            visiting.add(node_id)
            for dependency in dependency_map[node_id]:
                visit(dependency)
            visiting.remove(node_id)
            visited.add(node_id)

        for node_id in dependency_map:
            visit(node_id)

        for node_id, description, _dependencies in normalized:
            self._add_declared_node(
                node_id,
                "CONSTRAINT",
                {"description": description[:500], "origin": "initial_decomposition"},
            )
            self._add_edge("requires", "task", node_id)
        for node_id, _description, dependencies in normalized:
            for dependency in dependencies:
                self._add_edge("depends_on", node_id, dependency)
        self._task_graph_initialized = True
        self._count_interface("initialize_task_graph")
        return {
            "initialized": True,
            "requirements": [self.requirement_state(node_id) for node_id in dependency_map],
        }

    def search(self, *, query: str) -> list[dict[str, Any]]:
        """Search and record query, document, intent, and reusable artifact nodes."""
        target = self._current_action_target()
        prior_document_ids = self._target_document_ids(target)
        invocation = self._tool_runtime.invoke(
            "search",
            target=target,
            query=query,
        )
        results = invocation.value
        self._query_count += 1
        query_id = f"query:{self._query_count}"
        assert invocation.artifact_id is not None
        artifact_id = invocation.artifact_id
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
        self._add_node(
            artifact_id,
            "ARTIFACT",
            {"operation": "search", "query": str(query)[:500]},
        )
        self._add_edge("produces", query_id, artifact_id)
        return results

    def fetch(self, *, docid: str) -> dict[str, Any]:
        """Fetch and retain a source artifact for verified evidence and later reuse."""
        requested = str(docid)
        invocation = self._tool_runtime.invoke(
            "fetch",
            target=self._current_action_target(),
            docid=requested,
        )
        if invocation.reused:
            self._count_interface("reuse_fetch_artifact")
            return copy.deepcopy(invocation.value)
        result = invocation.value
        key = str(result.get("docid", docid))
        content = str(result.get("content", ""))
        self._fetch_count += 1
        assert invocation.artifact_id is not None
        artifact_id = invocation.artifact_id
        document_id = self._document_id(key)
        self._add_node(
            document_id,
            "DOCUMENT",
            {"docid": key, "fetched": True},
        )
        self._fetched_content[key] = content
        self._fetched_results[key] = copy.deepcopy(result)
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
        existing = self._nodes.get(node_id)
        if (
            existing is not None
            and existing["kind"] == "CONSTRAINT"
            and existing["data"].get("description") == text[:500]
        ):
            return {"node_id": node_id, "verified": True}
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
        return self._episode_graph.load_artifact(key)

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
            state = self.requirement_state(node_id)
            if state["status"] != "SATISFIED":
                constraints.append({**self._compact_node(node), **state})
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
            "reusable_artifacts": [
                artifact_id
                for artifact_id in self._artifacts
                if self._nodes.get(artifact_id, {}).get("data", {}).get("operation")
                in {"search", "fetch"}
            ][-self._max_items :],
        }

    def delta(self) -> dict[str, Any]:
        delta = self._episode_graph.delta()
        visible_nodes = [
            node
            for node in delta.nodes
            if node["kind"]
            in {
                "TASK",
                "CONSTRAINT",
                "CANDIDATE",
                "EVIDENCE",
                "QUERY",
                "DOCUMENT",
                "ACTION",
            }
            or (
                node["kind"] == "ARTIFACT"
                and node.get("data", {}).get("operation") in {"search", "fetch"}
            )
        ]
        visible_ids = {node["id"] for node in visible_nodes}
        return {
            "new_nodes": [self._compact_node(node) for node in visible_nodes],
            "new_edges": [
                edge
                for edge in delta.edges
                if edge["source"] in visible_ids and edge["target"] in visible_ids
            ],
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
            "artifact_reuse_hits": self._reuse_hits + self._tool_runtime.reuse_hits,
            "verified_evidence": kinds["EVIDENCE"],
            "task_graph_initialized": self._task_graph_initialized,
            "requirement_states": dict(
                Counter(
                    self.requirement_state(node_id)["status"]
                    for node_id, node in self._nodes.items()
                    if node["kind"] == "CONSTRAINT"
                )
            ),
            "frontier": self.frontier(),
        }

    def interface_counts(self) -> dict[str, int]:
        return dict(self._interface_calls)

    @property
    def task_graph_initialized(self) -> bool:
        return self._task_graph_initialized

    def action_snapshot(self, target_id: str) -> dict[str, Any]:
        """Return deterministic state used to verify an action's graph delta."""
        target = self._resolve_id(target_id)
        snapshot = {
            "target": target,
            "node_count": len(self._nodes),
            "edge_count": len(self._edges),
            "artifact_reuse_hits": self._reuse_hits,
        }
        if self._nodes[target]["kind"] == "CONSTRAINT":
            snapshot.update(self.requirement_state(target))
        else:
            snapshot.update(
                {
                    "status": "TASK_ONLY" if not self._task_graph_initialized else "ACTIVE",
                    "queries": self._query_count,
                    "retrieved_documents": sum(
                        node["kind"] == "DOCUMENT" for node in self._nodes.values()
                    ),
                    "fetched_documents": len(self._fetched_content),
                    "evidence": sum(
                        node["kind"] == "EVIDENCE" for node in self._nodes.values()
                    ),
                }
            )
        return snapshot

    def action_targets(self) -> dict[str, list[str]]:
        frontier = self.frontier()
        inspect = [
            node_id
            for node_id, node in self._nodes.items()
            if node["kind"] in {"CONSTRAINT", "CANDIDATE"}
        ][-self._max_items :]
        continue_targets = [item["id"] for item in frontier["unresolved_constraints"]]
        continue_targets.sort(key=self._target_priority)
        current = self._current_action_target()
        if current in continue_targets:
            others = [item for item in continue_targets if item != current]
            continue_targets = [current, *others]
        if not continue_targets:
            continue_targets = ["task"]
        return {
            "ANSWER": ["task"],
            "CONTINUE": continue_targets,
            "INSPECT": inspect or ["task"],
            "PATCH": [self._current_action_target() or "task"],
            "REUSE_REPLAY": list(frontier["reusable_artifacts"]),
        }

    def control_view(
        self,
        execution: Mapping[str, Any] | None,
    ) -> dict[str, Any]:
        """Build the retrieval projection's targets and semantic opportunities."""
        targets = self.action_targets()
        opportunities: list[dict[str, Any]] = []
        for index, continue_target in enumerate(targets["CONTINUE"]):
            full_context = self.target_context(continue_target)
            item = {
                "action": "CONTINUE",
                "target": continue_target,
                **_diagnose_retrieval_target(
                    full_context,
                    task_graph_initialized=self.task_graph_initialized,
                ),
                "target_context": (
                    full_context
                    if index == 0
                    else {
                        "target": full_context["target"],
                        "requirement_state": full_context.get("requirement_state"),
                    }
                ),
            }
            opportunities.append(item)
        if targets["INSPECT"] != ["task"]:
            opportunities.append(
                {
                    "action": "INSPECT",
                    "target": targets["INSPECT"][0],
                    "reason": "inspect provenance or competing evidence before choosing a branch",
                }
            )
        opportunities.append(
            {
                "action": "ANSWER",
                "target": "task",
                "reason": "finish only when the graph-backed evidence is sufficient",
            }
        )
        signals: dict[str, Any] = {}
        if execution is not None:
            calls = execution["block_calls"]
            if not execution["success"]:
                opportunities.insert(
                    0,
                    {
                        "action": "PATCH",
                        "target": targets["PATCH"][0],
                        "reason": "the previous executable block failed",
                    },
                )
            if targets["REUSE_REPLAY"] and (
                calls["repeated_queries"] or calls["repeated_fetches"]
            ):
                opportunities.insert(
                    1,
                    {
                        "action": "REUSE_REPLAY",
                        "target": targets["REUSE_REPLAY"][0],
                        "reason": "an equivalent dependency artifact is already materialized",
                    },
                )
            signals = {
                "last_block_success": execution["success"],
                "repeated_queries": calls["repeated_queries"],
                "zero_novelty_searches": calls["zero_novelty_searches"],
                "new_docids": calls["new_docids"],
                "new_fetches": calls["new_fetches"],
                "failed_tool_calls": calls["failed_tool_calls"],
            }
        return {
            "targets": targets,
            "opportunities": opportunities,
            "signals": signals,
            "answer_context": self.answer_context(),
        }

    @staticmethod
    def expected_delta_realized(
        expected: Mapping[str, Any] | None,
        before: Mapping[str, Any],
        after: Mapping[str, Any],
        *,
        initialized: bool,
    ) -> bool:
        """Interpret a generic graph delta using retrieval-domain semantics."""
        operation = str((expected or {}).get("operation", ""))
        if initialized:
            return True
        if operation == "SEARCH_TARGET":
            return int(after.get("queries", 0)) > int(before.get("queries", 0))
        if operation == "FETCH_DOCUMENT":
            return int(after.get("fetched_documents", 0)) > int(
                before.get("fetched_documents", 0)
            )
        if operation in {"ADD_EVIDENCE", "VERIFY_CANDIDATE", "RESOLVE_CONFLICT"}:
            return (
                int(after.get("evidence", 0)) > int(before.get("evidence", 0))
                or after.get("status") != before.get("status")
            )
        if operation == "REUSE_ARTIFACT":
            return int(after.get("artifact_reuse_hits", 0)) > int(
                before.get("artifact_reuse_hits", 0)
            )
        return any(
            after.get(key) != before.get(key)
            for key in (
                "status",
                "queries",
                "retrieved_documents",
                "fetched_documents",
                "evidence",
                "artifact_reuse_hits",
            )
        )

    def answer_context(self) -> dict[str, Any]:
        frontier = self.frontier()
        candidates = [
            {
                "id": node_id,
                "label": node["data"].get("label"),
                **self._candidate_counts(node_id),
            }
            for node_id, node in self._nodes.items()
            if node["kind"] == "CANDIDATE"
        ][-self._max_items :]
        return {
            "candidates": candidates,
            "unresolved_constraints": [
                {
                    "id": item["id"],
                    "status": item["status"],
                    "description": item["data"].get("description"),
                }
                for item in frontier["unresolved_constraints"]
            ],
        }

    def answer_review_context(self) -> dict[str, Any]:
        """Return the compact verified subgraph used for conservative answer review."""
        constraints = [
            {
                "id": node_id,
                "description": node["data"].get("description"),
                "status": self.requirement_state(node_id)["status"],
            }
            for node_id, node in self._nodes.items()
            if node["kind"] == "CONSTRAINT"
        ]
        candidates = [
            {
                "id": node_id,
                "label": node["data"].get("label"),
                **self._candidate_counts(node_id),
            }
            for node_id, node in self._nodes.items()
            if node["kind"] == "CANDIDATE"
        ]
        evidence = []
        for node_id, node in self._nodes.items():
            if node["kind"] != "EVIDENCE" or not node["data"].get("verified"):
                continue
            relations = [
                {
                    "relation": edge["type"],
                    "target": edge["target"],
                }
                for edge in self._edges
                if edge["source"] == node_id
                and edge["type"] in {"supports", "refutes", "addresses"}
            ]
            evidence.append(
                {
                    "id": node_id,
                    "docid": node["data"].get("docid"),
                    "quote": node["data"].get("quote"),
                    "relations": relations,
                }
            )
        return {
            "task": self._nodes["task"]["data"].get("question"),
            "constraints": constraints[:12],
            "candidates": candidates[:12],
            "verified_evidence": evidence[-16:],
        }

    def requirement_state(self, node_id: str) -> dict[str, Any]:
        target = self._resolve_id(node_id, {"CONSTRAINT"})
        dependencies = [
            edge["target"]
            for edge in self._edges
            if edge["type"] == "depends_on" and edge["source"] == target
        ]
        dependency_states = [self._requirement_status(value, set()) for value in dependencies]
        status = self._requirement_status(target, set())
        query_ids = {
            edge["source"]
            for edge in self._edges
            if edge["type"] == "intends_to_resolve" and edge["target"] == target
        }
        document_ids = self._target_document_ids(target)
        fetched = sum(
            str(self._nodes[value]["data"].get("docid", "")) in self._fetched_content
            for value in document_ids
            if value in self._nodes
        )
        evidence_ids = self._constraint_evidence_ids(target)
        return {
            "id": target,
            "status": status,
            "dependencies": dependencies,
            "dependency_states": dependency_states,
            "queries": len(query_ids),
            "retrieved_documents": len(document_ids),
            "fetched_documents": fetched,
            "evidence": len(evidence_ids),
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
        result = {
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
        if target == "task":
            result["child_requirements"] = [
                {
                    "id": edge["target"],
                    "description": self._nodes[edge["target"]]["data"].get(
                        "description"
                    ),
                    "status": self.requirement_state(edge["target"])["status"],
                }
                for edge in self._edges
                if edge["type"] == "requires"
                and edge["source"] == "task"
                and self._nodes.get(edge["target"], {}).get("kind") == "CONSTRAINT"
            ][: self._max_items]
        if self._nodes[target]["kind"] == "CONSTRAINT":
            result["requirement_state"] = self.requirement_state(target)
        return result

    def _constraint_evidence_ids(self, node_id: str) -> set[str]:
        return {
            edge["source"]
            for edge in self._edges
            if edge["target"] == node_id
            and edge["type"] in {"supports", "refutes", "addresses"}
            and self._nodes.get(edge["source"], {}).get("kind") == "EVIDENCE"
        }

    def _requirement_status(self, node_id: str, path: set[str]) -> str:
        if node_id in path:
            return "UNEXPLORED"
        path = {*path, node_id}
        evidence_ids = self._constraint_evidence_ids(node_id)
        polarities = {
            edge["type"]
            for edge in self._edges
            if edge["source"] in evidence_ids and edge["type"] in _RELATIONS
        }
        dependencies = [
            edge["target"]
            for edge in self._edges
            if edge["type"] == "depends_on" and edge["source"] == node_id
        ]
        dependencies_ready = all(
            self._requirement_status(value, path) == "SATISFIED"
            for value in dependencies
        )
        if polarities == {"supports", "refutes"}:
            return "CONFLICTED"
        if "refutes" in polarities:
            return "REFUTED"
        if "supports" in polarities:
            return "SATISFIED" if dependencies_ready else "SUPPORTED"
        if evidence_ids:
            return "EVIDENCE_FOUND"
        if any(
            edge["type"] == "intends_to_resolve" and edge["target"] == node_id
            for edge in self._edges
        ):
            return "SEARCHED"
        return "UNEXPLORED"

    def _target_priority(self, node_id: str) -> tuple[int, int, int]:
        state = self.requirement_state(node_id)
        dependencies_ready = all(
            value == "SATISFIED" for value in state["dependency_states"]
        )
        status_rank = {
            "CONFLICTED": 0,
            "REFUTED": 1,
            "SUPPORTED": 1,
            "EVIDENCE_FOUND": 2,
            "SEARCHED": 3,
            "UNEXPLORED": 4,
        }.get(state["status"], 5)
        downstream = sum(
            edge["type"] == "depends_on" and edge["target"] == node_id
            for edge in self._edges
        )
        return (0 if dependencies_ready else 1, status_rank, -downstream)

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
        self._episode_graph.add_node(node_id, kind, data)

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
        self._episode_graph.add_edge(edge_type, source, target)

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
        elif node["kind"] == "ARTIFACT":
            data = {
                key: data[key]
                for key in ("operation", "query", "docid")
                if key in data
            }
        elif node["kind"] == "TASK" and "question" in data:
            data = {"question": data["question"]}
        return {
            "id": node["id"],
            "kind": node["kind"],
            "data": data,
        }

    def _count_interface(self, name: str) -> None:
        self._interface_calls[name] += 1


def _diagnose_retrieval_target(
    context: Mapping[str, Any],
    *,
    task_graph_initialized: bool,
) -> dict[str, Any]:
    state = context.get("requirement_state")
    if not isinstance(state, Mapping):
        if not task_graph_initialized:
            return {
                "diagnosis": "TASK_DECOMPOSITION_REQUIRED",
                "reason": "decompose the task into stable, independently verifiable requirements",
                "expected_graph_delta": {
                    "operation": "INITIALIZE_TASK_GRAPH",
                    "from": "TASK_ONLY",
                    "to": "REQUIREMENTS_DECLARED",
                },
            }
        if context.get("child_requirements"):
            return {
                "diagnosis": "COMPOSITE_GOAL_COVERAGE",
                "reason": (
                    "pursue one action that can identify a shared candidate or advance multiple "
                    "ready child requirements"
                ),
                "expected_graph_delta": {"operation": "SEARCH_TARGET"},
            }
        return {
            "diagnosis": "NO_RETRIEVAL_COVERAGE",
            "reason": "extend the task-level research state",
            "expected_graph_delta": {"operation": "SEARCH_TARGET"},
        }

    status = str(state.get("status", "UNEXPLORED"))
    dependencies_ready = all(
        value == "SATISFIED" for value in state.get("dependency_states", ())
    )
    if not dependencies_ready:
        return {
            "diagnosis": "DEPENDENCY_NOT_SATISFIED",
            "reason": "resolve prerequisite requirements before advancing this target",
            "expected_graph_delta": {"operation": "SATISFY_DEPENDENCY"},
        }
    if status == "UNEXPLORED":
        return {
            "diagnosis": "NO_RETRIEVAL_COVERAGE",
            "reason": "search specifically for evidence that resolves this requirement",
            "expected_graph_delta": {
                "operation": "SEARCH_TARGET",
                "from": "UNEXPLORED",
                "to": "SEARCHED",
            },
        }
    if status == "SEARCHED":
        if context.get("unfetched_documents"):
            document = context["unfetched_documents"][0]
            docid = str((document.get("data") or {}).get("docid", ""))
            return {
                "diagnosis": "UNFETCHED_CANDIDATE_DOCUMENT",
                "reason": "fetch a document already retrieved for this requirement",
                "expected_graph_delta": {"operation": "FETCH_DOCUMENT"},
                "suggested_operations": (
                    [{"operation": "fetch", "docid": docid}] if docid else []
                ),
            }
        return {
            "diagnosis": "STALLED_RETRIEVAL_BRANCH",
            "reason": "use the query lineage to pursue a distinct evidence path",
            "expected_graph_delta": {"operation": "ADD_EVIDENCE"},
        }
    if status == "EVIDENCE_FOUND":
        return {
            "diagnosis": "UNSUPPORTED_CANDIDATE",
            "reason": "connect verified evidence to a candidate or requirement",
            "expected_graph_delta": {"operation": "VERIFY_CANDIDATE"},
        }
    if status == "CONFLICTED":
        return {
            "diagnosis": "CONFLICTING_EVIDENCE",
            "reason": "inspect the competing evidence and distinguish the alternatives",
            "expected_graph_delta": {"operation": "RESOLVE_CONFLICT"},
        }
    if status == "REFUTED":
        return {
            "diagnosis": "REFUTED_REQUIREMENT_PATH",
            "reason": "change candidate or research branch using the refuting evidence",
            "expected_graph_delta": {"operation": "ADD_EVIDENCE"},
        }
    return {
        "diagnosis": "DEPENDENCY_NOT_SATISFIED",
        "reason": "resolve remaining prerequisite coverage",
        "expected_graph_delta": {"operation": "SATISFY_DEPENDENCY"},
    }


# Backward-compatible name for the retrieval projection used by the current
# online controller. New domains should implement their own projection over
# EpisodeGraph rather than extending retrieval-specific node semantics.
ResearchGraphState = RetrievalGraphProjection
