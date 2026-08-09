from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit Stage 7.1 offline research projections.")
    parser.add_argument("projection_paths", nargs="+", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    reports = [json.loads(path.read_text(encoding="utf-8")) for path in args.projection_paths]
    checks: dict[str, bool] = {}
    checks["all_offline"] = all(
        report.get("mode") == "offline-research-layer-projection"
        and report.get("official_benchmark_result") is False
        for report in reports
    )
    checks["expected_episode_count"] = all(report.get("episode_count") == 20 for report in reports)
    checks["graph_node_ids_unique"] = all(
        len({node["id"] for node in graph["nodes"]}) == len(graph["nodes"])
        for report in reports
        for graph in report["graphs"]
    )
    checks["graph_edges_closed"] = all(
        _edges_closed(report) for report in reports
    )
    checks["no_model_visible_fields"] = all(
        _no_model_visible_fields(report) for report in reports
    )
    output = {
        "schema_version": 1,
        "stage": "7.1",
        "mode": "offline-research-layer-projection-gate",
        "official_benchmark_result": False,
        "passed": all(checks.values()),
        "checks": checks,
        "projection_paths": [str(path) for path in args.projection_paths],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(output, ensure_ascii=False))
    if not output["passed"]:
        raise SystemExit(1)


def _edges_closed(report: dict[str, Any]) -> bool:
    for graph in report.get("graphs", []):
        node_ids = {node["id"] for node in graph.get("nodes", [])}
        if any(
            edge.get("source") not in node_ids or edge.get("target") not in node_ids
            for edge in graph.get("edges", [])
        ):
            return False
    return True


def _no_model_visible_fields(report: dict[str, Any]) -> bool:
    forbidden = {"prompt", "messages", "tool_schema", "observation"}
    return not any(
        forbidden & set(node.get("data", {}))
        for graph in report.get("graphs", [])
        for node in graph.get("nodes", [])
    )


if __name__ == "__main__":
    main()
