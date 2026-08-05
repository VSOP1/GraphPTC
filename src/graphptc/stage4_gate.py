from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any, Callable

from .failure_attribution import build_failure_contexts
from .patch_controller import (
    GRAPHPTC_REPAIR_PROMPT_VARIANT,
    LocalPatchProposal,
    apply_local_patch,
    build_repair_context,
)
from .stage2_graph import DependencyGraph, load_dependency_graph_report
from .stage4_repair import RepairModel, request_local_patch, reexecute_patch_prefix


def write_stage4_gate_report(
    graph_path: str | Path,
    expectations_path: str | Path,
    output_path: str | Path,
) -> dict[str, Any]:
    expectation_bytes = Path(expectations_path).read_bytes()
    expectations = json.loads(expectation_bytes)
    if not isinstance(expectations, dict) or expectations.get("schema_version") != 1:
        raise ValueError("Unsupported Stage 4 promotion gate expectations")
    if expectations.get("prompt_variant") != GRAPHPTC_REPAIR_PROMPT_VARIANT:
        raise ValueError("Stage 4 gate requires prompt_variant='fewshot-ptc-v1'")
    timeout_seconds = expectations.get("timeout_seconds")
    if not isinstance(timeout_seconds, (int, float)) or timeout_seconds <= 0:
        raise ValueError("Stage 4 gate timeout_seconds must be positive")
    positive_cases = _case_list(expectations, "positive_cases")
    negative_cases = _case_list(expectations, "negative_cases")

    graphs = load_dependency_graph_report(graph_path)
    graphs_by_episode = {graph.episode_id: graph for graph in graphs}
    if len(graphs_by_episode) != len(graphs):
        raise ValueError("Stage 4 gate graph contains duplicate episode IDs")

    positive_results = [
        _run_positive_case(graphs_by_episode, case, float(timeout_seconds))
        for case in positive_cases
    ]
    negative_results = [
        _run_negative_case(graphs_by_episode, case) for case in negative_cases
    ]
    positive_count = len(positive_results)
    negative_count = len(negative_results)
    valid_count = sum(result["checks"]["patch_valid"] for result in positive_results)
    location_count = sum(
        result["checks"]["patch_location"] for result in positive_results
    )
    reexecution_count = sum(
        result["checks"]["reexecution_success"] for result in positive_results
    )
    rejected_count = sum(result["rejected"] for result in negative_results)
    out_of_bounds_acceptance_count = sum(
        result["out_of_bounds"] and not result["rejected"]
        for result in negative_results
    )
    report = {
        "schema_version": 1,
        "expectations_sha256": hashlib.sha256(expectation_bytes).hexdigest(),
        "prompt_variant": GRAPHPTC_REPAIR_PROMPT_VARIANT,
        "positive_case_count": positive_count,
        "negative_case_count": negative_count,
        "patch_valid_rate": _rate(valid_count, positive_count),
        "patch_location_match_rate": _rate(location_count, positive_count),
        "reexecution_success_rate": _rate(reexecution_count, positive_count),
        "negative_rejection_rate": _rate(rejected_count, negative_count),
        "out_of_bounds_acceptance_count": out_of_bounds_acceptance_count,
        "passed": (
            all(result["passed"] for result in positive_results)
            and all(result["passed"] for result in negative_results)
            and out_of_bounds_acceptance_count == 0
        ),
        "positive_cases": positive_results,
        "negative_cases": negative_results,
    }
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return report


def write_stage4_model_gate_report(
    model: RepairModel,
    graph_path: str | Path,
    expectations_path: str | Path,
    output_path: str | Path,
    *,
    runtime_tool_manifest: tuple[dict[str, Any], ...] = (),
) -> dict[str, Any]:
    expectation_bytes = Path(expectations_path).read_bytes()
    expectations = json.loads(expectation_bytes)
    if not isinstance(expectations, dict) or expectations.get("schema_version") != 1:
        raise ValueError("Unsupported Stage 4 model gate expectations")
    if expectations.get("prompt_variant") != GRAPHPTC_REPAIR_PROMPT_VARIANT:
        raise ValueError("Stage 4 model gate requires prompt_variant='fewshot-ptc-v1'")
    timeout_seconds = expectations.get("timeout_seconds")
    if not isinstance(timeout_seconds, (int, float)) or timeout_seconds <= 0:
        raise ValueError("Stage 4 model gate timeout_seconds must be positive")
    cases = _case_list(expectations, "positive_cases")
    graphs = load_dependency_graph_report(graph_path)
    graphs_by_episode = {graph.episode_id: graph for graph in graphs}

    results = []
    for case in cases:
        graph, repair = _repair_for_case(
            graphs_by_episode,
            case,
            runtime_tool_manifest=runtime_tool_manifest,
        )
        expected_proposal = _proposal(case)
        try:
            generated = request_local_patch(model, repair)
        except Exception as exc:
            results.append(
                {
                    "case_id": case["case_id"],
                    "episode_id": graph.episode_id,
                    "passed": False,
                    "checks": {
                        "location_match": False,
                        "patch_valid": False,
                        "reexecution_success": False,
                        "stdout": False,
                        "tool_calls": False,
                        "zero_reuse": False,
                    },
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                }
            )
            continue
        proposal = generated.proposal
        location_match = (
            proposal.block_id == expected_proposal.block_id
            and proposal.start_line == expected_proposal.start_line
            and proposal.end_line == expected_proposal.end_line
        )
        try:
            application = apply_local_patch(graph, repair, proposal)
        except ValueError as exc:
            results.append(
                {
                    "case_id": case["case_id"],
                    "episode_id": graph.episode_id,
                    "passed": False,
                    "checks": {
                        "location_match": location_match,
                        "patch_valid": False,
                        "reexecution_success": False,
                        "stdout": False,
                        "tool_calls": False,
                        "zero_reuse": False,
                    },
                    "generated_patch": asdict(generated),
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                }
            )
            continue
        namespace, expected_calls, observed_calls = _tool_namespace(
            case.get("tools", [])
        )
        reexecution = reexecute_patch_prefix(
            graph,
            application,
            namespace=namespace,
            timeout_seconds=float(timeout_seconds),
        )
        stdout = reexecution.blocks[-1].stdout if reexecution.blocks else ""
        checks = {
            "location_match": location_match,
            "patch_valid": True,
            "reexecution_success": reexecution.success,
            "stdout": stdout == case.get("expected_stdout"),
            "tool_calls": observed_calls == expected_calls,
            "zero_reuse": reexecution.reused_block_ids == (),
        }
        results.append(
            {
                "case_id": case["case_id"],
                "episode_id": graph.episode_id,
                "passed": all(checks.values()),
                "checks": checks,
                "generated_patch": asdict(generated),
                "application": application.to_dict(),
                "reexecution": asdict(reexecution),
            }
        )

    case_count = len(results)
    report = {
        "schema_version": 1,
        "expectations_sha256": hashlib.sha256(expectation_bytes).hexdigest(),
        "prompt_variant": GRAPHPTC_REPAIR_PROMPT_VARIANT,
        "case_count": case_count,
        "model_request_count": case_count,
        "location_match_rate": _rate(
            sum(result["checks"]["location_match"] for result in results),
            case_count,
        ),
        "patch_valid_rate": _rate(
            sum(result["checks"]["patch_valid"] for result in results),
            case_count,
        ),
        "reexecution_success_rate": _rate(
            sum(result["checks"]["reexecution_success"] for result in results),
            case_count,
        ),
        "passed": all(result["passed"] for result in results),
        "cases": results,
    }
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return report


def _run_positive_case(
    graphs: dict[str, DependencyGraph],
    case: dict[str, Any],
    timeout_seconds: float,
) -> dict[str, Any]:
    graph, repair = _repair_for_case(graphs, case)
    proposal = _proposal(case)
    try:
        application = apply_local_patch(graph, repair, proposal)
    except ValueError as exc:
        return {
            "case_id": case["case_id"],
            "episode_id": graph.episode_id,
            "passed": False,
            "checks": {
                "patch_valid": False,
                "patch_location": False,
                "code_sha256": False,
                "reexecution_success": False,
                "block_ids": False,
                "stdout": False,
                "tool_calls": False,
                "zero_reuse": False,
            },
            "error": str(exc),
        }
    namespace, expected_calls, observed_calls = _tool_namespace(case.get("tools", []))
    reexecution = reexecute_patch_prefix(
        graph,
        application,
        namespace=namespace,
        timeout_seconds=timeout_seconds,
    )
    block_ids = [block.block_id for block in reexecution.blocks]
    stdout = reexecution.blocks[-1].stdout if reexecution.blocks else ""
    checks = {
        "patch_valid": True,
        "patch_location": (
            application.patched.block_id == proposal.block_id
            and application.mapping.old_start_line == proposal.start_line
            and application.mapping.old_end_line == proposal.end_line
        ),
        "code_sha256": application.patched.code_sha256
        == case.get("expected_code_sha256"),
        "reexecution_success": reexecution.success,
        "block_ids": block_ids == case.get("expected_block_ids"),
        "stdout": stdout == case.get("expected_stdout"),
        "tool_calls": observed_calls == expected_calls,
        "zero_reuse": reexecution.reused_block_ids == (),
    }
    return {
        "case_id": case["case_id"],
        "episode_id": graph.episode_id,
        "source_events_sha256": graph.source_events_sha256,
        "passed": all(checks.values()),
        "checks": checks,
        "observed_tool_calls": observed_calls,
        "application": application.to_dict(),
        "reexecution": asdict(reexecution),
    }


def _run_negative_case(
    graphs: dict[str, DependencyGraph],
    case: dict[str, Any],
) -> dict[str, Any]:
    graph, repair = _repair_for_case(graphs, case)
    rejected = False
    error = None
    try:
        apply_local_patch(graph, repair, _proposal(case))
    except ValueError as exc:
        rejected = True
        error = str(exc)
    expected_error = case.get("expected_error")
    return {
        "case_id": case["case_id"],
        "episode_id": graph.episode_id,
        "passed": rejected and error == expected_error,
        "rejected": rejected,
        "out_of_bounds": case.get("out_of_bounds") is True,
        "expected_error": expected_error,
        "observed_error": error,
    }


def _repair_for_case(
    graphs: dict[str, DependencyGraph],
    case: dict[str, Any],
    *,
    runtime_tool_manifest: tuple[dict[str, Any], ...] = (),
) -> tuple[DependencyGraph, Any]:
    episode_id = str(case.get("episode_id") or "")
    if episode_id not in graphs:
        raise ValueError(f"Unknown Stage 4 gate episode: {episode_id}")
    graph = graphs[episode_id]
    anchor_node_id = str(case.get("anchor_node_id") or "")
    contexts = [
        context
        for context in build_failure_contexts(graph)
        if context.anchor.node_id == anchor_node_id
    ]
    if len(contexts) != 1:
        raise ValueError(f"Expected one Stage 4 gate anchor: {anchor_node_id}")
    return graph, build_repair_context(
        graph,
        contexts[0],
        runtime_tool_manifest=runtime_tool_manifest,
    )


def _proposal(case: dict[str, Any]) -> LocalPatchProposal:
    value = case.get("proposal")
    if not isinstance(value, dict):
        raise ValueError("Stage 4 gate case requires a proposal")
    try:
        return LocalPatchProposal(**value)
    except TypeError as exc:
        raise ValueError("Invalid Stage 4 gate proposal") from exc


def _tool_namespace(
    specifications: Any,
) -> tuple[
    dict[str, Callable[..., Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
]:
    if not isinstance(specifications, list):
        raise ValueError("Stage 4 gate tools must be a list")
    namespace: dict[str, Callable[..., Any]] = {}
    expected_calls: list[dict[str, Any]] = []
    observed_calls: list[dict[str, Any]] = []
    for specification in specifications:
        name = str(specification.get("name") or "")
        calls = specification.get("calls")
        if not name or not isinstance(calls, list):
            raise ValueError("Invalid Stage 4 gate tool fixture")
        expected_calls.extend(
            {"name": name, "arguments": call["arguments"]} for call in calls
        )
        namespace[name] = _fixture_tool(name, calls, observed_calls)
    return namespace, expected_calls, observed_calls


def _fixture_tool(
    name: str,
    calls: list[dict[str, Any]],
    observed: list[dict[str, Any]],
) -> Callable[..., Any]:
    index = 0

    def invoke(**kwargs: Any) -> Any:
        nonlocal index
        observed.append({"name": name, "arguments": kwargs})
        if index >= len(calls) or kwargs != calls[index].get("arguments"):
            raise ValueError(f"Unexpected {name} call: {kwargs}")
        result = copy.deepcopy(calls[index].get("result"))
        index += 1
        return result

    return invoke


def _case_list(values: dict[str, Any], name: str) -> list[dict[str, Any]]:
    cases = values.get(name)
    if not isinstance(cases, list) or not all(isinstance(case, dict) for case in cases):
        raise ValueError(f"Stage 4 gate {name} must be a list of objects")
    ids = [case.get("case_id") for case in cases]
    if len(set(ids)) != len(ids):
        raise ValueError(f"Stage 4 gate {name} contains duplicate case IDs")
    return cases


def _rate(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 1.0
