from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

from graphptc.config import ExperimentConfig


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit the preregistered Stage 7.4c auto-progress pilot.")
    parser.add_argument("gate_path", type=Path)
    parser.add_argument("preregistration_path", type=Path)
    parser.add_argument("control_config", type=Path)
    parser.add_argument("placebo_config", type=Path)
    parser.add_argument("graph_config", type=Path)
    parser.add_argument("output_path", type=Path)
    args = parser.parse_args()

    config_paths = {
        "control": args.control_config,
        "placebo_auto": args.placebo_config,
        "graph_auto": args.graph_config,
    }
    gate = _json(args.gate_path)
    preregistration = _json(args.preregistration_path)
    acceptance = gate["acceptance"]
    configs = {
        name: ExperimentConfig.from_toml(path)
        for name, path in config_paths.items()
    }
    runs = {
        name: _load_run(config)
        for name, config in configs.items()
    }
    expected_ids = set(runs["control"]["grades"])
    placebo = runs["placebo_auto"]
    graph = runs["graph_auto"]
    control = runs["control"]

    exact_reduction = _relative_reduction(
        placebo["exact_search_repeat_rate"], graph["exact_search_repeat_rate"]
    )
    zero_novelty_reduction = _relative_reduction(
        placebo["zero_novelty_search_rate"], graph["zero_novelty_search_rate"]
    )
    best_stagnation_reduction = max(exact_reduction, zero_novelty_reduction)
    tool_call_increase = _relative_increase(placebo["tool_calls"], graph["tool_calls"])
    repeated_fetch_rate_increase = (
        graph["repeated_fetch_rate"] - placebo["repeated_fetch_rate"]
    )

    frozen_artifacts = _frozen_artifacts_match(
        preregistration, args.gate_path, config_paths
    )
    checks = {
        "preregistration_passed": preregistration.get("passed") is True,
        "frozen_artifacts_match_preregistration": frozen_artifacts,
        "twenty_unique_responses_each": all(
            run["response_count"] == acceptance["expected_examples_per_arm"]
            and run["unique_response_count"] == run["response_count"]
            for run in runs.values()
        ),
        "matched_example_ids": all(set(run["grades"]) == expected_ids for run in runs.values()),
        "complete_grade_records_each": all(
            run["grade_record_count"] == acceptance["expected_examples_per_arm"]
            and run["grade_integrity"]
            for run in runs.values()
        ),
        "fewshot_prompt_each": all(
            run["prompt_variant"] == acceptance["prompt_variant"]
            for run in runs.values()
        ),
        "stateful_tool_support_disabled": acceptance["stateful_tool_support"] is False,
        "matched_model_visible_tool_contract": _tool_contracts_match(runs),
        "control_exposure_count": control["snapshot_calls"] == acceptance["control_exposure_count"],
        "placebo_auto_exposure_exact": placebo["auto_exposure_rate"] == acceptance["auto_exposure_rate"]
        and placebo["per_episode_exposure_exact"],
        "graph_auto_exposure_exact": graph["auto_exposure_rate"] == acceptance["auto_exposure_rate"]
        and graph["per_episode_exposure_exact"],
        "placebo_accuracy_noninferior": placebo["correct"]
        >= control["correct"] - acceptance["placebo_accuracy_noninferiority_margin_questions"],
        "graph_accuracy_noninferior": graph["correct"]
        >= placebo["correct"] - acceptance["graph_accuracy_noninferiority_margin_questions"],
        "stagnation_rate_reduction": best_stagnation_reduction
        >= acceptance["minimum_relative_stagnation_rate_reduction"],
        "tool_call_increase_bounded": tool_call_increase
        <= acceptance["maximum_tool_call_increase_fraction"],
        "repeated_fetch_rate_increase_bounded": repeated_fetch_rate_increase
        <= acceptance["maximum_repeated_fetch_rate_increase"],
    }
    pairs = {
        "control_vs_placebo_auto": _paired(control["grades"], placebo["grades"]),
        "placebo_auto_vs_graph_auto": _paired(placebo["grades"], graph["grades"]),
        "control_vs_graph_auto": _paired(control["grades"], graph["grades"]),
    }
    passed = all(checks.values())
    report = {
        "schema_version": 1,
        "stage": "7.4c",
        "mode": "preregistered-matched-auto-progress-pilot",
        "official_benchmark_result": False,
        "development_subset": True,
        "passed": passed,
        "promotion_eligible": passed,
        "checks": checks,
        "acceptance": acceptance,
        "derived_gate_metrics": {
            "exact_search_repeat_rate_relative_reduction": exact_reduction,
            "zero_novelty_search_rate_relative_reduction": zero_novelty_reduction,
            "best_stagnation_rate_relative_reduction": best_stagnation_reduction,
            "tool_call_increase_fraction": tool_call_increase,
            "repeated_fetch_rate_increase": repeated_fetch_rate_increase,
        },
        "runs": {
            name: {key: value for key, value in run.items() if key not in {"grades", "tool_contract"}}
            for name, run in runs.items()
        },
        "pairs": pairs,
        "artifacts": {
            str(path): _sha256(path)
            for path in [args.gate_path, args.preregistration_path, *config_paths.values()]
        },
        "interpretation": {
            "result": "gate passed" if passed else "gate failed; automatic graph progress is not promoted",
            "accuracy_boundary": "grader transitions are descriptive results on a real fixed development subset, not an official benchmark result",
            "causal_boundary": "the matched placebo isolates graph-valued capsules from automatic capsule exposure, but independent model trajectories remain stochastic",
            "next": "do not rerun or tune this preregistered challenger; diagnose why the graph capsule increased search and fetch activity before registering another intervention",
        },
    }
    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    args.output_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({
        "passed": passed,
        "promotion_eligible": passed,
        "checks": checks,
        "derived_gate_metrics": report["derived_gate_metrics"],
        "scores": {name: run["correct"] for name, run in runs.items()},
        "pairs": pairs,
    }))
    if not passed:
        raise SystemExit(1)


def _load_run(config: ExperimentConfig) -> dict[str, Any]:
    responses = _jsonl(config.benchmark.responses_path)
    grades = _jsonl(config.benchmark.grades_path)
    report = _json(config.benchmark.report_path)
    event_path = config.benchmark.responses_path.with_name("events.jsonl")
    successful_blocks = _successful_blocks_by_task(event_path)
    response_status = {
        str(item["example_id"]): str(item.get("status")) for item in responses
    }
    grade_map = {
        str(item["example_id"]): item.get("correct") is True for item in grades
    }
    ungraded_ids = {
        str(item["example_id"])
        for item in grades
        if item.get("correct") not in {True, False}
    }
    snapshots = {
        str(item["example_id"]): int((item.get("graph_progress") or {}).get("snapshot_calls", 0))
        for item in responses
    }
    generation = report["generation"]
    search_calls = generation["search_calls"]
    fetch_calls = generation["fetch_calls"]
    snapshot_calls = sum(snapshots.values())
    successful_ptc_blocks = generation["successful_ptc_blocks"]
    return {
        "prompt_variant": config.browsecomp_plus.prompt_variant,
        "graph_progress_mode": config.runtime.graph_progress_mode,
        "response_count": len(responses),
        "unique_response_count": len({str(item["example_id"]) for item in responses}),
        "successful_responses": sum(item.get("status") == "success" for item in responses),
        "failed_responses": sum(item.get("status") != "success" for item in responses),
        "grade_record_count": len(grades),
        "valid_grade_count": sum(item.get("correct") in {True, False} for item in grades),
        "grade_integrity": ungraded_ids
        <= {example_id for example_id, status in response_status.items() if status != "success"}
        and report["summary"]["invalid_auto_rater_responses"] == 0
        and report["summary"]["judge_errors"] == 0,
        "correct": sum(grade_map.values()),
        "accuracy": report["summary"]["accuracy"],
        "candidate_retrieval_recall": report["summary"]["candidate_retrieval_recall"],
        "fetched_evidence_recall": report["summary"]["fetched_evidence_recall"],
        "successful_ptc_blocks": successful_ptc_blocks,
        "failed_ptc_blocks": generation["failed_ptc_blocks"],
        "snapshot_calls": snapshot_calls,
        "auto_exposure_rate": snapshot_calls / successful_ptc_blocks if successful_ptc_blocks else 0.0,
        "per_episode_exposure_exact": snapshots == successful_blocks,
        "tool_calls": generation["tool_calls"],
        "search_calls": search_calls,
        "fetch_calls": fetch_calls,
        "repeated_exact_search_queries": generation["repeated_exact_search_queries"],
        "searches_without_new_docids": generation["searches_without_new_docids"],
        "repeated_fetches": generation["repeated_fetches"],
        "exact_search_repeat_rate": generation["repeated_exact_search_queries"] / search_calls if search_calls else 0.0,
        "zero_novelty_search_rate": generation["searches_without_new_docids"] / search_calls if search_calls else 0.0,
        "repeated_fetch_rate": generation["repeated_fetches"] / fetch_calls if fetch_calls else 0.0,
        "mean_duration_ms": generation["mean_duration_ms"],
        "grades": grade_map,
        "tool_contract": {
            "system_prompt": report["run_configuration"]["system_prompt"],
            "runtime_tool_manifest": report["run_configuration"]["runtime_tool_manifest"],
            "ptc_tool_spec": report["run_configuration"]["ptc_tool_spec"],
            "direct_tool_specs": report["run_configuration"]["direct_tool_specs"],
        },
    }


def _successful_blocks_by_task(path: Path) -> dict[str, int]:
    counts: Counter[str] = Counter()
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            event = json.loads(line)
            if event.get("type") == "block.finished" and event.get("data", {}).get("success") is True:
                counts[str(event["task_id"])] += 1
    return dict(counts)


def _tool_contracts_match(runs: dict[str, dict[str, Any]]) -> bool:
    contracts = [
        json.dumps(run["tool_contract"], sort_keys=True, separators=(",", ":"))
        for run in runs.values()
    ]
    return len(set(contracts)) == 1


def _frozen_artifacts_match(
    preregistration: dict[str, Any], gate_path: Path, config_paths: dict[str, Path]
) -> bool:
    if preregistration.get("gate_sha256") != _sha256(gate_path):
        return False
    for name, path in config_paths.items():
        if preregistration["arms"][name]["config_sha256"] != _sha256(path):
            return False
    return all(
        _sha256(Path(path)) == digest
        for path, digest in preregistration["implementation"].items()
    )


def _relative_reduction(left: float, right: float) -> float:
    return (left - right) / left if left else 0.0


def _relative_increase(left: int, right: int) -> float:
    return (right - left) / left if left else 0.0


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


def _json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


if __name__ == "__main__":
    main()
