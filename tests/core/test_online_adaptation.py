from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from graphptc.online_adaptation import OnlineGraphAdaptation
from graphptc.research_graph import ResearchGraphState


class FakeTools:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    @property
    def consumed(self) -> int:
        return len(self.calls)

    def search(self, *, query: str) -> list[dict[str, object]]:
        result = [{"docid": "d1", "score": 1.0, "snippet": "Alpha was founded in 2001."}]
        self.calls.append(
            {
                "operation": "search",
                "success": True,
                "query": query,
                "docids": ["d1"],
            }
        )
        return result

    def fetch(self, *, docid: str) -> dict[str, str]:
        self.calls.append(
            {
                "operation": "fetch",
                "success": True,
                "docid": docid,
                "docids": [docid],
            }
        )
        return {"docid": docid, "content": "Alpha was founded in 2001 by Ada."}


def _adaptation() -> OnlineGraphAdaptation:
    return OnlineGraphAdaptation(FakeTools(), max_tool_calls=20, task="Who founded Alpha?")


def test_agent_authored_graph_is_source_verified_and_emitted_as_delta() -> None:
    adaptation = _adaptation()
    adaptation.record_action(
        {"action": "CONTINUE", "target": "task", "expected_change": "identify founder"}
    )
    constraint = adaptation.graph_add_constraint(
        constraint_id="founder", description="identify the founder"
    )
    candidate = adaptation.graph_add_candidate(candidate_id="ada", label="Ada")
    adaptation.search(query="Alpha founder")
    page = adaptation.fetch(docid="d1")
    evidence = adaptation.graph_add_evidence(
        evidence_id="e1",
        docid="d1",
        quote=page["content"],
        relation="supports",
        target_id=candidate["node_id"],
        constraint_id=constraint["node_id"],
    )

    message = adaptation.observe(SimpleNamespace(success=True))
    payload = json.loads(message.removeprefix("GRAPH_DELTA "))

    assert payload["control_contract"] == "graph-adapt-v7"
    assert payload["declared_action"]["target_valid"] is True
    assert payload["execution"]["block_calls"]["search_calls"] == 1
    assert evidence["verified"] is True
    assert payload["research_graph_delta"]["frontier"]["unresolved_constraints"] == []
    kinds = {item["kind"] for item in payload["research_graph_delta"]["new_nodes"]}
    assert {
        "TASK",
        "ACTION",
        "CONSTRAINT",
        "CANDIDATE",
        "QUERY",
        "DOCUMENT",
        "EVIDENCE",
    } <= kinds


def test_evidence_rejects_unfetched_or_unquoted_sources() -> None:
    adaptation = _adaptation()
    candidate = adaptation.graph_add_candidate(candidate_id="ada", label="Ada")

    with pytest.raises(ValueError, match="has not been fetched"):
        adaptation.graph_add_evidence(
            evidence_id="e1",
            docid="d1",
            quote="Ada",
            relation="supports",
            target_id=candidate["node_id"],
        )

    adaptation.fetch(docid="d1")
    with pytest.raises(ValueError, match="not an exact span"):
        adaptation.graph_add_evidence(
            evidence_id="e1",
            docid="d1",
            quote="Grace",
            relation="supports",
            target_id=candidate["node_id"],
        )


def test_artifact_load_reuses_exact_value_without_external_call() -> None:
    adaptation = _adaptation()
    original = adaptation.search(query="Alpha founder")
    calls_before = adaptation.telemetry()["research_graph"]["node_count"]

    reused = adaptation.graph_load_artifact(artifact_id="artifact:search:1")

    assert reused == original
    assert adaptation.telemetry()["research_graph"]["artifact_reuse_hits"] == 1
    assert adaptation.telemetry()["research_graph"]["node_count"] == calls_before


def test_frontier_trace_and_alternatives_are_targeted_graph_queries() -> None:
    adaptation = _adaptation()
    constraint = adaptation.graph_add_constraint(
        constraint_id="founder", description="identify the founder"
    )
    first = adaptation.graph_add_candidate(candidate_id="ada", label="Ada")
    second = adaptation.graph_add_candidate(candidate_id="grace", label="Grace")

    frontier = adaptation.graph_frontier()
    trace = adaptation.graph_trace(node_id=constraint["node_id"])
    alternatives = adaptation.graph_alternatives(target_id=first["node_id"])

    assert frontier["unresolved_constraints"][0]["id"] == constraint["node_id"]
    assert trace["node"]["kind"] == "CONSTRAINT"
    assert alternatives["alternatives"][0]["id"] == second["node_id"]


def test_explicit_action_target_validation_is_side_channel_telemetry() -> None:
    adaptation = _adaptation()
    adaptation.record_action(
        {
            "action": "CONTINUE",
            "target": "constraint:missing",
            "expected_change": "resolve it",
        }
    )
    adaptation.observe(SimpleNamespace(success=False))
    adaptation.finish(answered=True)
    telemetry = adaptation.telemetry()

    assert telemetry["invalid_action_targets"] == 1
    assert telemetry["action_distribution"] == {"CONTINUE": 1, "ANSWER": 1}
    assert telemetry["action_history"][0]["target_valid"] is False


def test_action_can_target_constraint_declared_in_same_metadata() -> None:
    adaptation = _adaptation()
    adaptation.record_action(
        {
            "action": "CONTINUE",
            "target": "constraint:founder",
            "expected_change": "identify founder",
            "research_updates": {
                "constraints": [{"id": "founder", "description": "identify founder"}],
                "candidates": [],
                "evidence": [],
            },
        }
    )

    action = adaptation.telemetry()["action_history"][0]

    assert action["target_valid"] is True
    assert action["accepted_research_updates"] == 1


def test_graph_delta_is_bounded() -> None:
    adaptation = OnlineGraphAdaptation(
        FakeTools(),
        max_tool_calls=20,
        task="x" * 2_000,
        max_observation_chars=1_500,
    )
    for index in range(8):
        adaptation.graph_add_constraint(
            constraint_id=f"c{index}", description="description " * 20
        )

    message = adaptation.observe(SimpleNamespace(success=True))

    assert len(message) <= 1_500
    assert json.loads(message.removeprefix("GRAPH_DELTA "))["control_contract"] == "graph-adapt-v7"


def test_initial_assessment_exposes_bounded_action_contract() -> None:
    adaptation = _adaptation()

    payload = json.loads(
        adaptation.initial_observation().removeprefix("GRAPH_ASSESSMENT ")
    )

    assert payload["control_contract"] == "graph-adapt-v7"
    assert payload["available_actions"] == ["CONTINUE", "INSPECT", "ANSWER"]
    assert payload["valid_targets"]["CONTINUE"] == ["task"]
    assert payload["selected_action"] == "CONTINUE"
    assert payload["selected_target"] == "task"
    assert payload["target_context"]["target"]["id"] == "task"


def test_continue_target_context_exposes_only_related_research_lineage() -> None:
    tools = FakeTools()
    graph = ResearchGraphState(tools, task="Who founded Alpha?")
    founder = graph.graph_add_constraint(
        constraint_id="founder", description="identify the founder"
    )
    graph.graph_add_constraint(constraint_id="year", description="identify the year")
    candidate = graph.graph_add_candidate(candidate_id="ada", label="Ada")
    graph.set_action_target(founder["node_id"])
    graph.search(query="Alpha founder")
    page = graph.fetch(docid="d1")
    graph.graph_add_evidence(
        evidence_id="e1",
        docid="d1",
        quote=page["content"],
        relation="supports",
        target_id=candidate["node_id"],
        constraint_id=founder["node_id"],
    )
    graph.search(query="Alpha founder biography")
    graph.set_action_target("constraint:year")
    graph.search(query="Alpha founding year")

    context = graph.target_context(founder["node_id"])

    assert context["target"]["id"] == founder["node_id"]
    assert [item["data"]["text"] for item in context["query_history"]] == [
        "Alpha founder",
        "Alpha founder biography",
    ]
    assert context["reusable_artifact_ids"] == [
        "artifact:search:1",
        "artifact:search:2",
    ]
    assert context["unfetched_documents"] == []
    assert context["evidence"][0]["id"] == "evidence:e1"


def test_continue_target_context_is_pruned_to_observation_bound() -> None:
    adaptation = OnlineGraphAdaptation(
        FakeTools(),
        max_tool_calls=20,
        task="Who founded Alpha?",
        max_graph_items=8,
        max_observation_chars=1_500,
    )
    constraint = adaptation.graph_add_constraint(
        constraint_id="founder", description="identify the founder"
    )
    adaptation.record_action(
        {
            "action": "CONTINUE",
            "target": constraint["node_id"],
            "expected_change": "identify founder",
        }
    )
    for index in range(8):
        adaptation.search(query=f"Alpha founder query {index} " + "detail " * 20)

    message = adaptation.initial_observation()

    assert len(message) <= 1_500
    payload = json.loads(message.removeprefix("GRAPH_ASSESSMENT "))
    assert payload["target_context"]["target"]["id"] == constraint["node_id"]
    assert len(payload["target_context"]["query_history"]) < 8


def test_selected_inspect_must_be_implemented_by_graph_read() -> None:
    adaptation = _adaptation()
    adaptation.graph_add_constraint(
        constraint_id="founder", description="identify the founder"
    )
    adaptation.record_action(
        {
            "action": "INSPECT",
            "target": "constraint:founder",
            "expected_change": "inspect its provenance",
        }
    )
    adaptation.graph_trace(node_id="constraint:founder")

    adaptation.observe(SimpleNamespace(success=True))

    telemetry = adaptation.telemetry()
    assert telemetry["aligned_actions"] == 1
    assert telemetry["misaligned_actions"] == 0


def test_identical_semantic_declaration_is_idempotent() -> None:
    adaptation = _adaptation()

    first = adaptation.graph_add_constraint(
        constraint_id="founder", description="identify the founder"
    )
    second = adaptation.graph_add_constraint(
        constraint_id="founder", description="identify the founder"
    )

    assert second == first
    assert adaptation.telemetry()["research_graph"]["node_kinds"]["CONSTRAINT"] == 1


def test_stagnation_contract_overrides_another_continue_with_inspect() -> None:
    adaptation = _adaptation()
    adaptation.initial_observation()
    for _ in range(2):
        adaptation.record_action(
            {"action": "CONTINUE", "target": "task", "expected_change": "search"}
        )
        adaptation.search(query="Alpha founder")
        adaptation.observe(SimpleNamespace(success=True))

    prepared = adaptation.prepare_program_action(
        {
            "code": "print(search(query='again'))",
            "action": "CONTINUE",
            "target": "task",
            "expected_change": "search again",
        }
    )

    assert prepared["action"] == "INSPECT"
    assert "graph_frontier" in prepared["code"]
    assert adaptation.telemetry()["policy_overrides"] == 1


def test_selected_inspect_replaces_unaligned_retrieval_program() -> None:
    adaptation = _adaptation()
    adaptation.graph_add_constraint(
        constraint_id="founder", description="identify the founder"
    )
    adaptation.record_action(
        {
            "action": "INSPECT",
            "target": "constraint:founder",
            "expected_change": "inspect provenance",
        }
    )

    prepared = adaptation.prepare_program_action(
        {
            "code": "print(search(query='more'))",
            "action": "CONTINUE",
            "target": "constraint:founder",
        }
    )

    assert prepared["action"] == "INSPECT"
    assert prepared["code"] == (
        "print(graph_trace(node_id='constraint:founder'))"
    )
    telemetry = adaptation.telemetry()
    assert telemetry["selection_mismatches"] == 1
    assert telemetry["program_overrides"] == 1
