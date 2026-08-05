from __future__ import annotations

import argparse
import json
from pathlib import Path

from graphptc.stage5_commit_gate import write_stage5_commit_gate_report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the deterministic Stage 5.3 replay commit gate."
    )
    parser.add_argument("graph_path", type=Path)
    parser.add_argument("expectations_path", type=Path)
    parser.add_argument("output_path", type=Path)
    args = parser.parse_args()
    report = write_stage5_commit_gate_report(
        args.graph_path,
        args.expectations_path,
        args.output_path,
    )
    print(
        json.dumps(
            {
                "passed": report["passed"],
                "case_count": report["case_count"],
                "exact_match_rate": report["exact_match_rate"],
                "output_path": str(args.output_path),
            },
            ensure_ascii=False,
        )
    )
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
