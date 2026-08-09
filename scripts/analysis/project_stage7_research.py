from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from graphptc.research_projection import project_research_graph


def main() -> None:
    parser = argparse.ArgumentParser(description="Build an offline Stage 7 research projection.")
    parser.add_argument("events_path", type=Path)
    parser.add_argument("output_path", type=Path)
    args = parser.parse_args()
    grouped: dict[str, list[dict[str, object]]] = {}
    for line in args.events_path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            event = json.loads(line)
            grouped.setdefault(str(event["episode_id"]), []).append(event)
    graphs = [project_research_graph(group) for group in grouped.values()]
    report = {
        "schema_version": 1,
        "stage": "7.1",
        "mode": "offline-research-layer-projection",
        "official_benchmark_result": False,
        "source_events_sha256": hashlib.sha256(args.events_path.read_bytes()).hexdigest(),
        "episode_count": len(graphs),
        "graphs": graphs,
    }
    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    args.output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"stage": "7.1", "episode_count": len(graphs), "output": str(args.output_path)}))


if __name__ == "__main__":
    main()
