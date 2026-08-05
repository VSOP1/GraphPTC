from __future__ import annotations

import argparse
import json
from pathlib import Path

from graphptc.stage4_gate import write_stage4_gate_report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the deterministic Stage 4 local-patch promotion gate."
    )
    parser.add_argument("graph_path", type=Path)
    parser.add_argument("expectations_path", type=Path)
    parser.add_argument("output_path", type=Path)
    args = parser.parse_args()
    report = write_stage4_gate_report(
        args.graph_path,
        args.expectations_path,
        args.output_path,
    )
    print(
        json.dumps(
            {
                "passed": report["passed"],
                "positive_case_count": report["positive_case_count"],
                "negative_case_count": report["negative_case_count"],
                "patch_valid_rate": report["patch_valid_rate"],
                "reexecution_success_rate": report["reexecution_success_rate"],
                "negative_rejection_rate": report["negative_rejection_rate"],
                "out_of_bounds_acceptance_count": report[
                    "out_of_bounds_acceptance_count"
                ],
                "output_path": str(args.output_path),
            },
            ensure_ascii=False,
        )
    )
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
