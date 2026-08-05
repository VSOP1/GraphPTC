from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from .failure_attribution import build_failure_contexts
from .stage2_graph import load_dependency_graph_report


def write_stage3_audit_report(
    graph_path: str | Path,
    expectations_path: str | Path,
    output_path: str | Path,
) -> dict[str, Any]:
    expectations_source = Path(expectations_path)
    expectations_bytes = expectations_source.read_bytes()
    expectations = json.loads(expectations_bytes)
    if not isinstance(expectations, dict) or expectations.get("schema_version") != 1:
        raise ValueError("Unsupported Stage 3 audit expectations")
    cases = expectations.get("cases")
    if not isinstance(cases, list):
        raise ValueError("Stage 3 audit expectations require a cases list")

    graphs = load_dependency_graph_report(graph_path)
    graphs_by_episode = {graph.episode_id: graph for graph in graphs}
    expected_ids = [case.get("episode_id") for case in cases]
    if len(graphs_by_episode) != len(graphs):
        raise ValueError("Stage 3 audit graph contains duplicate episode IDs")
    if set(graphs_by_episode) != set(expected_ids):
        raise ValueError("Stage 3 audit graph episodes do not match expectations")

    max_nodes = _non_negative_int(expectations, "max_nodes", positive=True)
    code_radius = _non_negative_int(expectations, "code_radius")
    preview_chars = _non_negative_int(expectations, "preview_chars")
    results = []
    failure_count = 0
    for expected in cases:
        if not isinstance(expected, dict):
            raise ValueError("Each Stage 3 audit case must be an object")
        episode_id = str(expected["episode_id"])
        graph = graphs_by_episode[episode_id]
        contexts = build_failure_contexts(
            graph,
            max_nodes=max_nodes,
            code_radius=code_radius,
            preview_chars=preview_chars,
        )
        failure_count += len(contexts)
        anchor_kinds = [context.anchor.kind for context in contexts]
        error_types = [context.anchor.error_type for context in contexts]
        node_types = sorted({node.type for context in contexts for node in context.nodes})
        required_node_types = expected.get("required_node_types", [])
        checks = {
            "anchor_kinds": anchor_kinds == expected.get("anchor_kinds"),
            "error_types": error_types == expected.get("error_types"),
            "required_node_types": set(required_node_types).issubset(node_types),
            "node_budget": all(len(context.nodes) <= max_nodes for context in contexts),
            "artifact_preview_bound": all(
                len(artifact.preview) <= preview_chars
                for context in contexts
                for artifact in context.artifacts
            ),
        }
        results.append(
            {
                "episode_id": episode_id,
                "task_id": graph.task_id,
                "source_events_sha256": graph.source_events_sha256,
                "passed": all(checks.values()),
                "checks": checks,
                "observed": {
                    "anchor_kinds": anchor_kinds,
                    "error_types": error_types,
                    "node_types": node_types,
                },
                "contexts": [context.to_dict() for context in contexts],
            }
        )

    passed_case_count = sum(case["passed"] for case in results)
    report = {
        "schema_version": 1,
        "expectations_sha256": hashlib.sha256(expectations_bytes).hexdigest(),
        "graph_count": len(graphs),
        "case_count": len(results),
        "passed_case_count": passed_case_count,
        "failure_count": failure_count,
        "passed": passed_case_count == len(results),
        "limits": {
            "max_nodes": max_nodes,
            "code_radius": code_radius,
            "preview_chars": preview_chars,
        },
        "cases": results,
    }
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return report


def _non_negative_int(
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
