from __future__ import annotations

import json
import re
from collections import Counter
from typing import Any, Mapping

from .research_graph import ResearchGraphState


ADAPT_ACTIONS = ("ANSWER", "CONTINUE", "INSPECT", "PATCH", "REUSE_REPLAY")
_MEMORY_METRIC_KEYS = (
    "visible_blocks",
    "target_matched_blocks",
    "target_mismatch_blocks",
    "searches_after_exposure",
    "differentiated_queries",
    "exposed_query_repeats",
    "known_document_fetches",
    "exposed_artifact_reuses",
    "aligned_blocks",
    "repeat_only_blocks",
)


class OnlineGraphAdaptation:
    """Typed research graph plus explicit, target-bound online Adapt decisions."""

    def __init__(
        self,
        tools: Any,
        *,
        max_tool_calls: int,
        task: str = "",
        max_graph_items: int = 4,
        max_observation_chars: int = 3_200,
    ) -> None:
        if max_tool_calls < 1:
            raise ValueError("max_tool_calls must be positive")
        self._tools = tools
        self._max_tool_calls = max_tool_calls
        self._max_observation_chars = max_observation_chars
        self._graph = ResearchGraphState(tools, task=task, max_items=max_graph_items)
        self._observation_calls = 0
        self._previous_call_count = 0
        self._actions: Counter[str] = Counter()
        self._action_history: list[dict[str, Any]] = []
        self._invalid_action_targets = 0
        self._rejected_research_updates = 0
        self._current_action: dict[str, Any] | None = None
        self._aligned_actions = 0
        self._misaligned_actions = 0
        self._last_execution: dict[str, Any] | None = None
        self._interface_counts_before_block: dict[str, int] = {}
        self._selection_mismatches = 0
        self._unavailable_action_requests = 0
        self._next_action_contract: dict[str, Any] | None = None
        self._current_memory_exposure: dict[str, Any] | None = None
        self._loaded_artifacts_since_observation: list[str] = []
        self._memory_metrics: Counter[str] = Counter(
            {key: 0 for key in _MEMORY_METRIC_KEYS}
        )

    def search(self, *, query: str) -> list[dict[str, Any]]:
        return self._graph.search(query=query)

    def fetch(self, *, docid: str) -> dict[str, Any]:
        return self._graph.fetch(docid=docid)

    def graph_add_constraint(self, *, constraint_id: str, description: str) -> dict[str, Any]:
        return self._graph.graph_add_constraint(
            constraint_id=constraint_id,
            description=description,
        )

    def graph_add_candidate(self, *, candidate_id: str, label: str) -> dict[str, Any]:
        return self._graph.graph_add_candidate(candidate_id=candidate_id, label=label)

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
        return self._graph.graph_add_evidence(
            evidence_id=evidence_id,
            docid=docid,
            quote=quote,
            relation=relation,
            target_id=target_id,
            constraint_id=constraint_id,
        )

    def graph_frontier(self) -> dict[str, Any]:
        return self._graph.graph_frontier()

    def graph_trace(self, *, node_id: str) -> dict[str, Any]:
        return self._graph.graph_trace(node_id=node_id)

    def graph_load_artifact(self, *, artifact_id: str) -> Any:
        value = self._graph.graph_load_artifact(artifact_id=artifact_id)
        self._loaded_artifacts_since_observation.append(str(artifact_id))
        return value

    def graph_alternatives(self, *, target_id: str) -> dict[str, Any]:
        return self._graph.graph_alternatives(target_id=target_id)

    def record_action(
        self,
        payload: Mapping[str, Any],
        *,
        source: str = "model_tool_call",
    ) -> dict[str, Any]:
        requested_action = str(payload.get("action", "")).strip().upper()
        action = requested_action
        target = str(payload.get("target", "")).strip()
        expected_change = str(payload.get("expected_change", "")).strip()[:500]
        accepted_updates = 0
        update_errors: list[str] = []
        flat_constraint_id = str(payload.get("constraint_id", "")).strip()
        flat_constraint = str(payload.get("constraint", "")).strip()
        if flat_constraint_id and flat_constraint:
            try:
                self.graph_add_constraint(
                    constraint_id=flat_constraint_id,
                    description=flat_constraint,
                )
                accepted_updates += 1
            except ValueError as exc:
                update_errors.append(str(exc))
        updates = payload.get("research_updates")
        if isinstance(updates, Mapping):
            for item in updates.get("constraints", ()):
                try:
                    self.graph_add_constraint(
                        constraint_id=str(item["id"]),
                        description=str(item["description"]),
                    )
                    accepted_updates += 1
                except (KeyError, TypeError, ValueError) as exc:
                    update_errors.append(str(exc))
            for item in updates.get("candidates", ()):
                try:
                    self.graph_add_candidate(
                        candidate_id=str(item["id"]),
                        label=str(item["label"]),
                    )
                    accepted_updates += 1
                except (KeyError, TypeError, ValueError) as exc:
                    update_errors.append(str(exc))
            for item in updates.get("evidence", ()):
                try:
                    self.graph_add_evidence(
                        evidence_id=str(item["id"]),
                        docid=str(item["docid"]),
                        quote=str(item["quote"]),
                        relation=str(item["relation"]),
                        target_id=str(item["target_id"]),
                        constraint_id=str(item.get("constraint_id", "")),
                    )
                    accepted_updates += 1
                except (KeyError, TypeError, ValueError) as exc:
                    update_errors.append(str(exc))
        action_available = True
        if self._next_action_contract is not None:
            available = self._next_action_contract["available_actions"]
            if action not in available:
                action_available = False
                self._unavailable_action_requests += 1
        self._rejected_research_updates += len(update_errors)
        action_valid = action in ADAPT_ACTIONS and action_available
        target_valid = self._graph.has_node(target)
        if not target_valid:
            self._invalid_action_targets += 1
        record = {
            "before_block": self._observation_calls + 1,
            "action": action or "MISSING",
            "requested_action": requested_action or "MISSING",
            "action_available": action_available,
            "target": target or None,
            "expected_change": expected_change or None,
            "action_valid": action_valid,
            "target_valid": target_valid,
            "source": source,
            "accepted_research_updates": accepted_updates,
            "rejected_research_updates": len(update_errors),
            "research_update_errors": update_errors[:3],
        }
        self._action_history.append(record)
        if action_valid:
            self._actions[action] += 1
        self._current_action = record
        self._interface_counts_before_block = self._graph.interface_counts()
        self._graph.set_action_target(target if target_valid else None)
        record["action_node_id"] = self._graph.record_action(
            action=record["action"],
            target=target,
            expected_change=expected_change,
            target_valid=target_valid,
            source=source,
        )

        return {
            "action": record["action"],
            "target": record["target"],
            "expected_change": record["expected_change"],
            "target_valid": record["target_valid"],
            "action_node_id": record["action_node_id"],
        }

    def initial_observation(self) -> str:
        assessment = self._assessment(execution=None)
        self._next_action_contract = assessment
        return _bounded_payload(
            "GRAPH_ASSESSMENT ",
            assessment,
            self._max_observation_chars,
        )

    def prepare_program_action(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        requested_action = str(payload.get("action", "")).strip().upper()
        requested_target = str(payload.get("target", "")).strip()
        exposure = _memory_exposure(
            self._next_action_contract,
            requested_target=requested_target,
        )
        pending_action = (
            self._current_action is not None
            and self._current_action.get("before_block") == self._observation_calls + 1
            and "execution_aligned" not in self._current_action
        )
        if not pending_action:
            self.record_action(payload, source="model_tool_call")
        assert self._current_action is not None
        self._current_memory_exposure = exposure
        self._current_action["requested_action"] = requested_action or "MISSING"
        self._current_action["requested_target"] = requested_target or None
        self._current_action["program_metadata_matches"] = True
        self._current_action["retrieval_memory_exposure"] = {
            "visible": exposure["visible"],
            "target": exposure["target"],
            "target_matched": exposure["target_matched"],
            "attempt_count": exposure["attempt_count"],
            "recent_query_count": len(exposure["queries"]),
            "unfetched_document_count": len(exposure["unfetched_docids"]),
            "reusable_artifact_count": len(exposure["artifact_ids"]),
        }
        return dict(payload)

    def observe(self, trace: Any) -> str:
        calls = list(self._tools.calls)
        prior_calls = calls[: self._previous_call_count]
        recent_calls = calls[self._previous_call_count :]
        self._previous_call_count = len(calls)
        self._observation_calls += 1
        consumed = int(getattr(self._tools, "consumed", len(calls)))
        execution = {
            "success": bool(getattr(trace, "success", False)),
            "block_calls": _call_summary(recent_calls, prior_calls=prior_calls),
            "remaining_tool_calls": max(0, self._max_tool_calls - consumed),
        }
        self._last_execution = execution
        if self._current_action is not None:
            memory_consumption = self._record_memory_consumption(recent_calls)
            self._current_action["retrieval_memory_consumption"] = memory_consumption
            aligned, reason = self._execution_alignment(
                self._current_action["action"], execution
            )
            self._current_action["execution_aligned"] = aligned
            self._current_action["alignment_reason"] = reason
            if aligned:
                self._aligned_actions += 1
            else:
                self._misaligned_actions += 1
        next_action_contract = self._assessment(execution=execution)
        self._next_action_contract = next_action_contract
        payload = {
            "schema_version": 1,
            "control_contract": "graph-adapt-v11",
            "block": self._observation_calls,
            "declared_action": _model_visible_action(self._current_action),
            "execution": execution,
            "research_graph_delta": self._graph.delta(),
            "next_action_contract": next_action_contract,
        }
        return _bounded_payload(
            "GRAPH_DELTA ", payload, self._max_observation_chars
        )

    def finish(self, *, answered: bool) -> None:
        if not answered:
            return
        if self._current_action is not None and self._current_action["action"] == "ANSWER":
            self._current_action["finalized"] = True
            return
        self._actions["ANSWER"] += 1
        record = {
            "after_block": self._observation_calls,
            "action": "ANSWER",
            "target": "task",
            "expected_change": "final_answer",
            "action_valid": True,
            "target_valid": True,
            "source": "final_result",
        }
        self._action_history.append(record)
        record["action_node_id"] = self._graph.record_action(
            action="ANSWER",
            target="task",
            expected_change="final_answer",
            target_valid=True,
            source="final_result",
        )

    def telemetry(self) -> dict[str, Any]:
        return {
            "mode": "online",
            "control_contract": "graph-adapt-v11",
            "observation_calls": self._observation_calls,
            "action_distribution": dict(self._actions),
            "action_history": list(self._action_history),
            "invalid_action_targets": self._invalid_action_targets,
            "rejected_research_updates": self._rejected_research_updates,
            "aligned_actions": self._aligned_actions,
            "misaligned_actions": self._misaligned_actions,
            "selection_mismatches": self._selection_mismatches,
            "unavailable_action_requests": self._unavailable_action_requests,
            "policy_overrides": 0,
            "program_overrides": 0,
            "max_observation_chars": self._max_observation_chars,
            "retrieval_memory_consumption": dict(self._memory_metrics),
            "research_graph": self._graph.telemetry(),
        }

    def _record_memory_consumption(
        self, recent_calls: list[Mapping[str, Any]]
    ) -> dict[str, Any]:
        exposure = self._current_memory_exposure or _empty_memory_exposure()
        visible = bool(exposure["visible"])
        target_matched = bool(exposure["target_matched"])
        if visible:
            self._memory_metrics["visible_blocks"] += 1
            if target_matched:
                self._memory_metrics["target_matched_blocks"] += 1
            else:
                self._memory_metrics["target_mismatch_blocks"] += 1

        searches = [
            call
            for call in recent_calls
            if call.get("operation") == "search" and call.get("success") is not False
        ]
        fetches = [
            call
            for call in recent_calls
            if call.get("operation") == "fetch" and call.get("success") is not False
        ]
        repeated_queries = sum(
            _normalize_query(call.get("query")) in exposure["queries"]
            for call in searches
        )
        differentiated_queries = len(searches) - repeated_queries
        fetched_known_documents = sum(
            bool(_call_docids(call) & exposure["unfetched_docids"])
            for call in fetches
        )
        reused_artifacts = sum(
            artifact_id in exposure["artifact_ids"]
            for artifact_id in self._loaded_artifacts_since_observation
        )
        aligned = bool(
            visible
            and target_matched
            and (
                differentiated_queries
                or fetched_known_documents
                or reused_artifacts
            )
        )
        if visible and target_matched:
            self._memory_metrics["searches_after_exposure"] += len(searches)
            self._memory_metrics["differentiated_queries"] += differentiated_queries
            self._memory_metrics["exposed_query_repeats"] += repeated_queries
            self._memory_metrics["known_document_fetches"] += fetched_known_documents
            self._memory_metrics["exposed_artifact_reuses"] += reused_artifacts
            if aligned:
                self._memory_metrics["aligned_blocks"] += 1
            if repeated_queries and not (
                differentiated_queries or fetched_known_documents or reused_artifacts
            ):
                self._memory_metrics["repeat_only_blocks"] += 1
        self._loaded_artifacts_since_observation.clear()
        self._current_memory_exposure = None
        return {
            "visible": visible,
            "target_matched": target_matched,
            "searches": len(searches),
            "differentiated_queries": differentiated_queries,
            "exposed_query_repeats": repeated_queries,
            "known_document_fetches": fetched_known_documents,
            "exposed_artifact_reuses": reused_artifacts,
            "aligned": aligned,
        }

    def _assessment(self, *, execution: dict[str, Any] | None) -> dict[str, Any]:
        targets = self._graph.action_targets()
        opportunities: list[dict[str, Any]] = []
        continue_target = targets["CONTINUE"][0]
        opportunities.append(
            {
                "action": "CONTINUE",
                "target": continue_target,
                "reason": (
                    "use target retrieval memory to resolve an evidence gap"
                    if continue_target != "task"
                    else "extend the current research state"
                ),
                "target_context": self._graph.target_context(continue_target),
            }
        )
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
        signals: dict[str, Any] = {}
        if execution is not None:
            calls = execution["block_calls"]
            signals = {
                "last_block_success": execution["success"],
                "repeated_queries": calls["repeated_queries"],
                "zero_novelty_searches": calls["zero_novelty_searches"],
                "new_docids": calls["new_docids"],
                "new_fetches": calls["new_fetches"],
                "failed_tool_calls": calls["failed_tool_calls"],
            }
        instruction = (
            "Choose the next action using the graph-grounded opportunities. The runtime does "
            "not choose or rewrite the program; action and target metadata must describe the "
            "program that you actually execute."
        )
        available = list(dict.fromkeys(item["action"] for item in opportunities))
        assessment = {
            "schema_version": 1,
            "control_contract": "graph-adapt-v11",
            "instruction": instruction,
            "available_actions": available,
            "valid_targets": {key: targets[key] for key in available},
            "action_opportunities": opportunities,
            "signals": signals,
            "answer_context": self._graph.answer_context(),
        }
        return assessment

    def _execution_alignment(
        self, action: str, execution: Mapping[str, Any]
    ) -> tuple[bool, str]:
        before = self._interface_counts_before_block
        after = self._graph.interface_counts()

        def used(name: str) -> bool:
            return after.get(name, 0) > before.get(name, 0)

        if action == "INSPECT":
            aligned = any(
                used(name)
                for name in ("graph_frontier", "graph_trace", "graph_alternatives")
            )
            return aligned, "graph inspection executed" if aligned else "no graph inspection"
        if action == "REUSE_REPLAY":
            aligned = used("graph_load_artifact")
            return aligned, "artifact loaded" if aligned else "artifact was not loaded"
        if action == "PATCH":
            aligned = bool(execution["success"])
            return aligned, "patch block succeeded" if aligned else "patch block failed"
        if action == "CONTINUE":
            return True, "research block executed"
        return False, "ANSWER must finalize without a PTC block"


def _call_summary(
    calls: list[Mapping[str, Any]],
    *,
    prior_calls: list[Mapping[str, Any]] | None = None,
) -> dict[str, int]:
    queries: Counter[str] = Counter()
    seen_docids: set[str] = set()
    fetched_docids: set[str] = set()
    for call in prior_calls or ():
        if call.get("success") is False:
            continue
        if call.get("operation") == "search":
            queries[_normalize_query(call.get("query"))] += 1
            seen_docids.update(str(value) for value in call.get("docids", ()))
        elif call.get("operation") == "fetch":
            fetched_docids.update(str(value) for value in call.get("docids", ()))
            if call.get("docid") is not None:
                fetched_docids.add(str(call["docid"]))
    current_queries: set[str] = set()
    repeated_queries = 0
    new_docids = 0
    repeated_docids = 0
    zero_novelty = 0
    repeated_fetches = 0
    new_fetches = 0
    search_calls = 0
    fetch_calls = 0
    failed_tool_calls = 0
    for call in calls:
        if call.get("success") is False:
            failed_tool_calls += 1
            continue
        operation = str(call.get("operation", ""))
        if operation == "search":
            search_calls += 1
            query = _normalize_query(call.get("query"))
            docids = {str(value) for value in call.get("docids", ())}
            unseen = docids - seen_docids
            new_docids += len(unseen)
            repeated_docids += len(docids & seen_docids)
            zero_novelty += bool(docids) and not unseen
            repeated_queries += queries[query] > 0
            queries[query] += 1
            current_queries.add(query)
            seen_docids.update(docids)
        elif operation == "fetch":
            fetch_calls += 1
            values = {str(value) for value in call.get("docids", ())}
            if call.get("docid") is not None:
                values.add(str(call["docid"]))
            repeated_fetches += len(values & fetched_docids)
            new_fetches += len(values - fetched_docids)
            fetched_docids.update(values)
    return {
        "search_calls": search_calls,
        "fetch_calls": fetch_calls,
        "unique_queries": len(current_queries),
        "repeated_queries": repeated_queries,
        "new_docids": new_docids,
        "repeated_docids": repeated_docids,
        "zero_novelty_searches": zero_novelty,
        "repeated_fetches": repeated_fetches,
        "new_fetches": new_fetches,
        "failed_tool_calls": failed_tool_calls,
    }


def _bounded_payload(prefix: str, payload: dict[str, Any], maximum: int) -> str:
    rendered = prefix + json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    if "research_graph_delta" not in payload:
        targets = payload.get("valid_targets", {})
        opportunities = payload.get("action_opportunities", [])
        contexts = [
            item.get("target_context", {})
            for item in opportunities
            if isinstance(item, dict)
        ]
        context_lists = [
            values
            for context in contexts
            for values in (
                context.get("evidence", []),
                context.get("unfetched_documents", []),
                context.get("fetched_documents", []),
                context.get("retrieval_memory", {}).get("recent_attempts", []),
            )
        ]
        answer_context = payload.get("answer_context", {})
        lists = context_lists + list(targets.values()) + [
            answer_context.get("candidates", []),
            answer_context.get("unresolved_constraints", []),
        ]
        while len(rendered) > maximum and any(lists):
            next(items for items in lists if items).pop()
            rendered = prefix + json.dumps(
                payload, ensure_ascii=False, separators=(",", ":")
            )
        if len(rendered) > maximum and contexts:
            for item in opportunities:
                item.pop("target_context", None)
            rendered = prefix + json.dumps(
                payload, ensure_ascii=False, separators=(",", ":")
            )
        if len(rendered) > maximum:
            raise ValueError("graph assessment exceeds configured bound")
        return rendered
    delta = payload["research_graph_delta"]
    contract = payload.get("next_action_contract", {})
    # Keep semantic claims visible longer than raw execution lineage when the
    # observation must be compacted to the configured bound.
    nodes = delta.get("new_nodes", [])
    priority = ("EVIDENCE", "CANDIDATE", "DOCUMENT", "QUERY", "TASK", "CONSTRAINT", "ACTION")
    delta["new_nodes"] = [
        *[item for kind in priority for item in nodes if item.get("kind") == kind],
        *[item for item in nodes if item.get("kind") not in priority],
    ]
    opportunities = contract.get("action_opportunities", [])
    contexts = [
        item.get("target_context", {})
        for item in opportunities
        if isinstance(item, dict)
    ]
    lists = [
        delta["frontier"]["unfetched_documents"],
        delta["frontier"]["reusable_artifacts"],
        delta["frontier"]["conflicted_candidates"],
        delta["frontier"]["unresolved_constraints"],
        delta["new_edges"],
        delta["new_nodes"],
        *(
            values
            for context in contexts
            for values in (
                context.get("evidence", []),
                context.get("unfetched_documents", []),
                context.get("fetched_documents", []),
                context.get("retrieval_memory", {}).get("recent_attempts", []),
            )
        ),
        *contract.get("valid_targets", {}).values(),
        contract.get("answer_context", {}).get("candidates", []),
        contract.get("answer_context", {}).get("unresolved_constraints", []),
    ]
    while len(rendered) > maximum and any(lists):
        next(items for items in lists if items).pop()
        rendered = prefix + json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    if len(rendered) > maximum and contexts:
        for item in opportunities:
            item.pop("target_context", None)
        rendered = prefix + json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    if len(rendered) > maximum:
        for item in opportunities:
            item.pop("reason", None)
        rendered = prefix + json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    if len(rendered) > maximum:
        raise ValueError("graph delta exceeds configured bound")
    return rendered


def _normalize_query(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "").strip().lower())


def _model_visible_action(action: Mapping[str, Any] | None) -> dict[str, Any] | None:
    if action is None:
        return None
    return {
        key: value
        for key, value in action.items()
        if key not in {"retrieval_memory_exposure", "retrieval_memory_consumption"}
    }


def _empty_memory_exposure() -> dict[str, Any]:
    return {
        "visible": False,
        "target": None,
        "target_matched": False,
        "attempt_count": 0,
        "queries": set(),
        "unfetched_docids": set(),
        "artifact_ids": set(),
    }


def _memory_exposure(
    contract: Mapping[str, Any] | None,
    *,
    requested_target: str,
) -> dict[str, Any]:
    exposure = _empty_memory_exposure()
    if not isinstance(contract, Mapping):
        return exposure
    opportunities = contract.get("action_opportunities", ())
    opportunity = next(
        (
            item
            for item in opportunities
            if isinstance(item, Mapping) and item.get("action") == "CONTINUE"
        ),
        None,
    )
    if not isinstance(opportunity, Mapping):
        return exposure
    target = str(opportunity.get("target", ""))
    context = opportunity.get("target_context")
    if not isinstance(context, Mapping):
        return exposure
    memory = context.get("retrieval_memory")
    if not isinstance(memory, Mapping):
        return exposure
    attempts = memory.get("recent_attempts", ())
    queries = {
        _normalize_query(item.get("query"))
        for item in attempts
        if isinstance(item, Mapping) and item.get("query")
    }
    artifact_ids = {
        str(item.get("artifact_id"))
        for item in attempts
        if isinstance(item, Mapping) and item.get("artifact_id")
    }
    unfetched_docids = {
        str((item.get("data") or {}).get("docid", ""))
        for item in context.get("unfetched_documents", ())
        if isinstance(item, Mapping) and (item.get("data") or {}).get("docid")
    }
    exposure.update(
        {
            "visible": bool(attempts or unfetched_docids),
            "target": target or None,
            "target_matched": bool(target and target == requested_target),
            "attempt_count": int(memory.get("attempt_count", 0)),
            "queries": queries,
            "unfetched_docids": unfetched_docids,
            "artifact_ids": artifact_ids,
        }
    )
    return exposure


def _call_docids(call: Mapping[str, Any]) -> set[str]:
    values = {str(value) for value in (call.get("docids") or ())}
    if call.get("docid") is not None:
        values.add(str(call["docid"]))
    return values
