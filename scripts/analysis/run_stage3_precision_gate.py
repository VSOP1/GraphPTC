from __future__ import annotations

import argparse
import json
from pathlib import Path

from graphptc.stage3_gate import write_stage3_precision_gate_report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the exact-match Stage 3 attribution promotion gate."
    )
    parser.add_argument("graph_path", type=Path)
    parser.add_argument("expectations_path", type=Path)
    parser.add_argument("output_path", type=Path)
    args = parser.parse_args()
    report = write_stage3_precision_gate_report(
        args.graph_path,
        args.expectations_path,
        args.output_path,
    )
    print(
        json.dumps(
            {
                "passed": report["passed"],
                "case_count": report["case_count"],
                "context_count": report["context_count"],
                "exact_match_rate": report["exact_match_rate"],
                "forbidden_leakage_count": report["forbidden_leakage_count"],
                "output_path": str(args.output_path),
            },
            ensure_ascii=False,
        )
    )
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
