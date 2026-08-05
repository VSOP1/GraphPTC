from __future__ import annotations

import argparse
import json
from pathlib import Path

from graphptc.failure_attribution import write_failure_attribution_report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build deterministic Stage 3 failure contexts from Stage 2 graphs."
    )
    parser.add_argument("graph_path", type=Path)
    parser.add_argument("output_path", type=Path)
    parser.add_argument("--max-nodes", type=int, default=64)
    parser.add_argument("--code-radius", type=int, default=2)
    parser.add_argument("--preview-chars", type=int, default=160)
    args = parser.parse_args()

    report = write_failure_attribution_report(
        args.graph_path,
        args.output_path,
        max_nodes=args.max_nodes,
        code_radius=args.code_radius,
        preview_chars=args.preview_chars,
    )
    print(
        json.dumps(
            {
                "episode_count": report["episode_count"],
                "failure_count": report["failure_count"],
                "output_path": str(args.output_path),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
