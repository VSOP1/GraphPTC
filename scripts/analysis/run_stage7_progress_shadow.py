from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from graphptc.research_projection import project_research_graph


def main() -> None:
    parser = argparse.ArgumentParser(description="Build an offline Stage 7.2 progress shadow.")
    parser.add_argument("events_path", type=Path)
    parser.add_argument("output_path", type=Path)
    parser.add_argument("--max-tool-calls", type=int, default=1000)
    args = parser.parse_args()
    grouped: dict[str, list[dict[str, Any]]] = {}
    for line in args.events_path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            event = json.loads(line)
            grouped.setdefault(str(event["episode_id"]), []).append(event)

    capsules = []
    for episode_id, events in grouped.items():
        projection = project_research_graph(events)
        metrics = projection["metrics"]
        tool_calls = sum(event.get("type") == "tool.called" for event in events)
        capsules.append(
            {
                "episode_id": episode_id,
                "tool_calls": tool_calls,
                "budget_fraction": tool_calls / args.max_tool_calls,
                "metrics": metrics,
                "signals": {
                    "repeated_query_seen": metrics["repeated_queries"] > 0,
                    "repeated_fetch_seen": metrics["repeated_fetches"] > 0,
                    "low_novelty_candidate": (
                        metrics["repeated_result_docids"]
                        > metrics["unique_docids"]
                    ),
                    "budget_risk_candidate": tool_calls >= args.max_tool_calls * 0.8,
                },
            }
        )
    capsules.sort(
        key=lambda item: (
            item["metrics"]["repeated_queries"],
            item["metrics"]["repeated_fetches"],
            item["tool_calls"],
        ),
        reverse=True,
    )
    report = {
        "schema_version": 1,
        "stage": "7.2",
        "mode": "offline-progress-shadow",
        "official_benchmark_result": False,
        "source_events_sha256": hashlib.sha256(args.events_path.read_bytes()).hexdigest(),
        "episode_count": len(capsules),
        "model_visible": False,
        "action_taken": None,
        "capsules": capsules,
    }
    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    args.output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"stage": "7.2", "episode_count": len(capsules), "output": str(args.output_path)}))


if __name__ == "__main__":
    main()
