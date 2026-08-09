from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit Stage 7.2 progress shadow outputs.")
    parser.add_argument("shadow_paths", nargs="+", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    reports = [json.loads(path.read_text(encoding="utf-8")) for path in args.shadow_paths]
    checks = {
        "all_offline": all(
            report.get("mode") == "offline-progress-shadow"
            and report.get("official_benchmark_result") is False
            for report in reports
        ),
        "expected_episode_count": all(report.get("episode_count") == 20 for report in reports),
        "model_visible_false": all(report.get("model_visible") is False for report in reports),
        "no_action_taken": all(report.get("action_taken") is None for report in reports),
        "capsule_fields_complete": all(
            all(
                {"episode_id", "tool_calls", "budget_fraction", "metrics", "signals"}
                <= set(capsule)
                for capsule in report.get("capsules", [])
            )
            for report in reports
        ),
    }
    result = {
        "schema_version": 1,
        "stage": "7.2",
        "mode": "offline-progress-shadow-gate",
        "official_benchmark_result": False,
        "passed": all(checks.values()),
        "checks": checks,
        "shadow_paths": [str(path) for path in args.shadow_paths],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False))
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
