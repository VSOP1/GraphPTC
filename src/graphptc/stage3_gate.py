from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from .failure_attribution import FailureContext, build_failure_contexts
from .stage2_graph import load_dependency_graph_report


def write_stage3_precision_gate_report(
    graph_path: str | Path,
    expectations_path: str | Path,
    output_path: str | Path,
) -> dict[str, Any]:
    expectations_bytes = Path(expectations_path).read_bytes()
    expectations = json.loads(expectations_bytes)
    if not isinstance(expectations, dict) or expectations.get("schema_version") != 1:
        raise ValueError("Unsupported Stage 3 precision gate expectations")
    expected_cases = expectations.get("cases")
    if not isinstance(expected_cases, list):
        raise ValueError("Stage 3 precision gate requires a cases list")

    graphs = load_dependency_graph_report(graph_path)
    graphs_by_episode = {graph.episode_id: graph for graph in graphs}
    expected_ids = [case.get("episode_id") for case in expected_cases]
    if len(set(expected_ids)) != len(expected_ids):
        raise ValueError("Stage 3 precision gate contains duplicate episode IDs")
    if len(graphs_by_episode) != len(graphs):
        raise ValueError("Stage 3 precision gate graph contains duplicate episode IDs")
    if set(graphs_by_episode) != set(expected_ids):
        raise ValueError("Stage 3 precision gate graph episodes do not match expectations")

    max_nodes = _integer_limit(expectations, "max_nodes", positive=True)
    code_radius = _integer_limit(expectations, "code_radius")
    preview_chars = _integer_limit(expectations, "preview_chars")
    case_results = []
    exact_passed = 0
    exact_total = 0
    forbidden_leakage_count = 0
    context_count = 0
    for expected_case in expected_cases:
        if not isinstance(expected_case, dict):
            raise ValueError("Each Stage 3 precision gate case must be an object")
        episode_id = str(expected_case["episode_id"])
        expected_contexts = expected_case.get("contexts")
        if not isinstance(expected_contexts, list):
            raise ValueError(f"Gate case {episode_id} requires a contexts list")
        contexts = build_failure_contexts(
            graphs_by_episode[episode_id],
            max_nodes=max_nodes,
            code_radius=code_radius,
            preview_chars=preview_chars,
        )
        context_count += len(contexts)
        context_results = []
        contexts_match = len(contexts) == len(expected_contexts)
        for index, expected_context in enumerate(expected_contexts):
            if not isinstance(expected_context, dict):
                raise ValueError(
                    f"Gate context {episode_id}:{index} must be an object"
                )
            if index >= len(contexts):
                context_results.append(
                    {"index": index, "passed": False, "missing": True}
                )
                exact_total += 3
                continue
            result = _evaluate_context(contexts[index], expected_context, max_nodes)
            context_results.append(result)
            exact_passed += sum(
                result["checks"][name]
                for name in ("anchor", "node_ids", "edges")
            )
            exact_total += 3
            forbidden_leakage_count += result["forbidden_leakage_count"]
        case_passed = contexts_match and all(
            result["passed"] for result in context_results
        )
        case_results.append(
            {
                "episode_id": episode_id,
                "source_events_sha256": graphs_by_episode[
                    episode_id
                ].source_events_sha256,
                "passed": case_passed,
                "context_count_match": contexts_match,
                "contexts": context_results,
            }
        )

    exact_match_rate = exact_passed / exact_total if exact_total else 1.0
    passed = (
        all(case["passed"] for case in case_results)
        and exact_match_rate == 1.0
        and forbidden_leakage_count == 0
    )
    report = {
        "schema_version": 1,
        "expectations_sha256": hashlib.sha256(expectations_bytes).hexdigest(),
        "graph_count": len(graphs),
        "case_count": len(case_results),
        "context_count": context_count,
        "exact_match_rate": exact_match_rate,
        "forbidden_leakage_count": forbidden_leakage_count,
        "passed": passed,
        "limits": {
            "max_nodes": max_nodes,
            "code_radius": code_radius,
            "preview_chars": preview_chars,
        },
        "cases": case_results,
    }
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return report


def _evaluate_context(
    context: FailureContext,
    expected: dict[str, Any],
    max_nodes: int,
) -> dict[str, Any]:
    observed_anchor = {
        "kind": context.anchor.kind,
        "node_id": context.anchor.node_id,
        "block_id": context.anchor.block_id,
        "error_type": context.anchor.error_type,
        "location": context.anchor.location,
    }
    observed_nodes = [node.id for node in context.nodes]
    observed_edges = [
        {"type": edge.type, "source": edge.source, "target": edge.target}
        for edge in context.edges
    ]
    observed_regions = [
        {
            "block_id": region.block_id,
            "start_line": region.start_line,
            "end_line": region.end_line,
            "focus_lines": list(region.focus_lines),
        }
        for region in context.code_regions
    ]
    observed_artifacts = [artifact.id for artifact in context.artifacts]
    forbidden_nodes = set(expected.get("forbidden_node_ids", ()))
    forbidden_artifacts = set(expected.get("forbidden_artifact_ids", ()))
    leaked_nodes = [node_id for node_id in observed_nodes if node_id in forbidden_nodes]
    leaked_artifacts = [
        artifact_id
        for artifact_id in observed_artifacts
        if artifact_id in forbidden_artifacts
    ]
    checks = {
        "anchor": observed_anchor == expected.get("anchor"),
        "node_ids": observed_nodes == expected.get("node_ids"),
        "edges": observed_edges == expected.get("edges"),
        "code_regions": observed_regions == expected.get("code_regions"),
        "expandable_artifact_ids": observed_artifacts
        == expected.get("expandable_artifact_ids"),
        "forbidden_nodes": not leaked_nodes,
        "forbidden_artifacts": not leaked_artifacts,
        "node_budget": len(context.nodes) <= max_nodes and not context.truncated,
    }
    return {
        "anchor_node_id": context.anchor.node_id,
        "passed": all(checks.values()),
        "checks": checks,
        "forbidden_leakage_count": len(leaked_nodes) + len(leaked_artifacts),
        "leaked_node_ids": leaked_nodes,
        "leaked_artifact_ids": leaked_artifacts,
        "observed": {
            "anchor": observed_anchor,
            "node_ids": observed_nodes,
            "edges": observed_edges,
            "code_regions": observed_regions,
            "expandable_artifact_ids": observed_artifacts,
        },
        "context": context.to_dict(),
    }


def _integer_limit(
    values: dict[str, Any],
    name: str,
    *,
    positive: bool = False,
) -> int:
    value = values.get(name)
    minimum = 1 if positive else 0
    if not isinstance(value, int) or isinstance(value, bool) or value < minimum:
        qualifier = "positive" if positive else "non-negative"
        raise ValueError(f"{name} must be a {qualifier} integer")
    return value
