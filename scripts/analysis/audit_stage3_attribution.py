from __future__ import annotations

import argparse
import json
from pathlib import Path

from graphptc.stage3_audit import write_stage3_audit_report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Audit Stage 3 attribution against fixed expectations."
    )
    parser.add_argument("graph_path", type=Path)
    parser.add_argument("expectations_path", type=Path)
    parser.add_argument("output_path", type=Path)
    args = parser.parse_args()

    report = write_stage3_audit_report(
        args.graph_path,
        args.expectations_path,
        args.output_path,
    )
    print(
        json.dumps(
            {
                "passed": report["passed"],
                "passed_case_count": report["passed_case_count"],
                "case_count": report["case_count"],
                "failure_count": report["failure_count"],
                "output_path": str(args.output_path),
            },
            ensure_ascii=False,
        )
    )
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
