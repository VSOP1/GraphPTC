from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit the online Adapt micro gate.")
    parser.add_argument("preregistration", type=Path)
    parser.add_argument("control_responses", type=Path)
    parser.add_argument("adapt_responses", type=Path)
    parser.add_argument("control_grades", type=Path)
    parser.add_argument("adapt_grades", type=Path)
    parser.add_argument("output_path", type=Path)
    args = parser.parse_args()
    prereg = _json(args.preregistration)
    expected_ids = prereg["example_ids"]
    control = _index(_jsonl(args.control_responses))
    adapt = _index(_jsonl(args.adapt_responses))
    control_grades = _index(_jsonl(args.control_grades))
    adapt_grades = _index(_jsonl(args.adapt_grades))
    acceptance = prereg["acceptance"]
    control_calls = sum(len(item["agent"]["search_calls"]) for item in control.values())
    adapt_calls = sum(len(item["agent"]["search_calls"]) for item in adapt.values())
    call_increase = (adapt_calls - control_calls) / control_calls if control_calls else None
    allowed_actions = set(acceptance["allowed_actions"])
    control_correct = sum(bool(item["correct"]) for item in control_grades.values())
    adapt_correct = sum(bool(item["correct"]) for item in adapt_grades.values())
    checks = {
        "expected_ids_each": all(
            set(values) == set(expected_ids)
            for values in (control, adapt, control_grades, adapt_grades)
        ),
        "signed_responses": all(
            item["run_signature"] == prereg[f"{name}_run_signature"]
            for name, values in (("control", control), ("adapt", adapt))
            for item in values.values()
        ),
        "all_examples_succeeded": all(
            item["status"] == "success" for values in (control, adapt) for item in values.values()
        ),
        "control_adaptation_absent": all(
            item.get("graph_adaptation") is None for item in control.values()
        ),
        "adapt_observation_per_completed_block": all(
            item["graph_adaptation"]["observation_calls"] == item["agent"]["ptc_blocks"]
            for item in adapt.values()
        ),
        "adapt_action_history_complete": all(
            len(item["graph_adaptation"]["action_history"])
            == item["graph_adaptation"]["observation_calls"]
            for item in adapt.values()
        ),
        "adapt_actions_allowed": all(
            action["action"] in allowed_actions
            for item in adapt.values()
            for action in item["graph_adaptation"]["action_history"]
        ),
        "adapt_tool_calls_bounded": call_increase is not None
        and call_increase <= acceptance["maximum_adapt_tool_call_increase_fraction"],
        "accuracy_noninferior": adapt_correct
        >= control_correct - acceptance["accuracy_noninferiority_margin_questions"],
    }
    paired = {
        example_id: {
            "control_correct": bool(control_grades[example_id]["correct"]),
            "adapt_correct": bool(adapt_grades[example_id]["correct"]),
            "control_prediction": control[example_id]["prediction"],
            "adapt_prediction": adapt[example_id]["prediction"],
            "control_blocks": control[example_id]["agent"]["ptc_blocks"],
            "adapt_blocks": adapt[example_id]["agent"]["ptc_blocks"],
            "control_tool_calls": len(control[example_id]["agent"]["search_calls"]),
            "adapt_tool_calls": len(adapt[example_id]["agent"]["search_calls"]),
        }
        for example_id in expected_ids
    }
    actions = Counter(
        action["action"]
        for item in adapt.values()
        for action in item["graph_adaptation"]["action_history"]
    )
    unbacked_reuse = actions["REUSE_REPLAY"] if not any(
        call.get("cache_hit")
        for item in adapt.values()
        for call in item["agent"]["search_calls"]
    ) else 0
    report = {
        "schema_version": 1,
        "mode": prereg.get("gate_mode", "matched-online-graph-adaptation-micro-gate"),
        "official_benchmark_result": False,
        "development_subset": True,
        "passed": all(checks.values()),
        "pilot20_eligible": all(checks.values()),
        "checks": checks,
        "summary": {
            "control_correct": control_correct,
            "adapt_correct": adapt_correct,
            "control_tool_calls": control_calls,
            "adapt_tool_calls": adapt_calls,
            "adapt_tool_call_increase_fraction": call_increase,
            "adapt_action_distribution": dict(actions),
        },
        "paired": paired,
        "posthoc_diagnostics": {
            "unbacked_reuse_replay_labels": unbacked_reuse,
            "interpretation": (
                "REUSE_REPLAY labels without cache hits represent repeated live execution, not reuse; "
                "this diagnostic was not added to the preregistered pass criteria"
            ),
        },
        "inputs": {
            str(path).replace("\\", "/"): _sha256(path)
            for path in (
                args.preregistration,
                args.control_responses,
                args.adapt_responses,
                args.control_grades,
                args.adapt_grades,
            )
        },
        "boundary": prereg["boundary"],
    }
    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    args.output_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "passed": report["passed"],
        "checks": checks,
        "summary": report["summary"],
        "posthoc_diagnostics": report["posthoc_diagnostics"],
    }))
    if not report["passed"]:
        raise SystemExit(1)


def _index(values: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    result = {str(item["example_id"]): item for item in values}
    if len(result) != len(values):
        raise ValueError("duplicate example IDs")
    return result


def _json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


if __name__ == "__main__":
    main()
