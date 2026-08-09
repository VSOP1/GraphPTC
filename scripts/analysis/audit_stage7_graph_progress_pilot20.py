from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from graphptc.config import ExperimentConfig


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit the real Stage 7.4 control/placebo/graph pilot.")
    parser.add_argument("control_config", type=Path)
    parser.add_argument("control_events", type=Path)
    parser.add_argument("placebo_config", type=Path)
    parser.add_argument("placebo_events", type=Path)
    parser.add_argument("graph_config", type=Path)
    parser.add_argument("graph_events", type=Path)
    parser.add_argument("output_path", type=Path)
    args = parser.parse_args()
    inputs = {
        "control": (args.control_config, args.control_events),
        "placebo": (args.placebo_config, args.placebo_events),
        "graph": (args.graph_config, args.graph_events),
    }
    runs = {
        name: _load_run(ExperimentConfig.from_toml(config_path), events_path)
        for name, (config_path, events_path) in inputs.items()
    }
    expected_ids = set(runs["control"]["grades"])
    checks = {
        "twenty_unique_responses_each": all(run["response_count"] == 20 for run in runs.values()),
        "matched_example_ids": all(set(run["grades"]) == expected_ids for run in runs.values()),
        "valid_grades_each": all(run["valid_grade_count"] == 20 for run in runs.values()),
        "fewshot_prompt_each": all(run["prompt_variant"] == "fewshot-ptc-v1" for run in runs.values()),
        "placebo_graph_interface_matched": runs["placebo"]["graph_progress_mode"] == "placebo" and runs["graph"]["graph_progress_mode"] == "graph",
        "placebo_interface_exposed": runs["placebo"]["progress_calls"] > 0,
        "graph_interface_exposed": runs["graph"]["progress_calls"] > 0,
        "outcome_gate_preregistered": False,
    }
    pairs = {
        "control_vs_placebo": _paired(runs["control"]["grades"], runs["placebo"]["grades"]),
        "placebo_vs_graph": _paired(runs["placebo"]["grades"], runs["graph"]["grades"]),
        "control_vs_graph": _paired(runs["control"]["grades"], runs["graph"]["grades"]),
    }
    attribution_eligible = checks["placebo_interface_exposed"] and checks["graph_interface_exposed"]
    report = {
        "schema_version": 1,
        "stage": "7.4",
        "mode": "real-placebo-controlled-graph-progress-pilot20",
        "official_benchmark_result": False,
        "development_subset": True,
        "passed": all(checks.values()),
        "promotion_eligible": False,
        "graph_effect_attribution_eligible": attribution_eligible,
        "checks": checks,
        "runs": {
            name: {key: value for key, value in run.items() if key != "grades"}
            for name, run in runs.items()
        },
        "pairs": pairs,
        "artifacts": {
            str(path): _sha256(path)
            for pair in inputs.values()
            for path in pair
        },
        "interpretation": {
            "result": "the pilot is not attribution-valid because neither challenger arm called graph_progress",
            "accuracy_boundary": "independent trajectory scores and grader transitions are descriptive, not a causal graph effect",
            "next": "register a new exposure mechanism and gate before any further outcome pilot; do not rerun this challenger",
        },
    }
    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    args.output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"passed": report["passed"], "promotion_eligible": False, "graph_effect_attribution_eligible": attribution_eligible, "runs": report["runs"], "pairs": pairs}))
    if not report["passed"]:
        raise SystemExit(1)


def _load_run(config: ExperimentConfig, events_path: Path) -> dict[str, Any]:
    responses = _jsonl(config.benchmark.responses_path)
    grades = _jsonl(config.benchmark.grades_path)
    report = json.loads(config.benchmark.report_path.read_text(encoding="utf-8"))
    events = _jsonl(events_path)
    correct = {str(item["example_id"]): bool(item.get("correct")) for item in grades}
    progress = [event for event in events if event.get("type") == "tool.called" and event.get("data", {}).get("tool") == "graph_progress"]
    return {
        "prompt_variant": config.browsecomp_plus.prompt_variant,
        "graph_progress_mode": config.runtime.graph_progress_mode,
        "response_count": len(responses),
        "valid_grade_count": sum(item.get("correct") in {True, False} for item in grades),
        "correct": sum(correct.values()),
        "accuracy": report["summary"]["accuracy"],
        "candidate_retrieval_recall": report["summary"]["candidate_retrieval_recall"],
        "fetched_evidence_recall": report["summary"]["fetched_evidence_recall"],
        "tool_calls": report["generation"]["tool_calls"],
        "repeated_exact_search_queries": report["generation"]["repeated_exact_search_queries"],
        "repeated_fetches": report["generation"]["repeated_fetches"],
        "mean_duration_ms": report["generation"]["mean_duration_ms"],
        "progress_calls": len(progress),
        "episodes_with_progress_calls": len({str(event["task_id"]) for event in progress}),
        "grades": correct,
    }


def _paired(left: dict[str, bool], right: dict[str, bool]) -> dict[str, Any]:
    ids = sorted(set(left) & set(right), key=int)
    wrong_to_correct = [item for item in ids if not left[item] and right[item]]
    correct_to_wrong = [item for item in ids if left[item] and not right[item]]
    return {
        "matched_examples": len(ids),
        "left_correct": sum(left[item] for item in ids),
        "right_correct": sum(right[item] for item in ids),
        "delta_correct": sum(right[item] - left[item] for item in ids),
        "wrong_to_correct": wrong_to_correct,
        "correct_to_wrong": correct_to_wrong,
    }


def _jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


if __name__ == "__main__":
    main()
