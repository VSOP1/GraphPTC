from __future__ import annotations

import argparse
import json
from pathlib import Path

from graphptc.stage2_graph import (
    write_dependency_graph_bundle,
    write_dependency_graph_report,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build deterministic Stage 2 graphs from Stage 1 event JSONL."
    )
    parser.add_argument("events_path", type=Path)
    parser.add_argument("output_path", type=Path)
    parser.add_argument("--artifacts-path", type=Path)
    args = parser.parse_args()

    report = (
        write_dependency_graph_bundle(
            args.events_path,
            args.output_path,
            args.artifacts_path,
        )
        if args.artifacts_path is not None
        else write_dependency_graph_report(args.events_path, args.output_path)
    )
    summary = {
        "graph_count": report["graph_count"],
        "output_path": str(args.output_path),
        "artifacts_path": (
            str(args.artifacts_path) if args.artifacts_path is not None else None
        ),
    }
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
