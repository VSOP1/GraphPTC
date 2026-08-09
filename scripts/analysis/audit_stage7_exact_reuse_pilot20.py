from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit Stage 7.3 exact reuse pilot20.")
    parser.add_argument("gate_path", type=Path)
    parser.add_argument("control_dir", type=Path)
    parser.add_argument("reuse_dir", type=Path)
    parser.add_argument("output_path", type=Path)
    args = parser.parse_args()
    gate = json.loads(args.gate_path.read_text(encoding="utf-8"))
    acceptance = gate["acceptance"]
    control = _run(args.control_dir)
    reuse = _run(args.reuse_dir)
    ids = set(control["responses"])
    paired = []
    for example_id in sorted(ids, key=int):
        control_correct = control["grades"][example_id]
        reuse_correct = reuse["grades"][example_id]
        paired.append(
            {
                "example_id": example_id,
                "control_correct": control_correct,
                "reuse_correct": reuse_correct,
                "transition": _transition(control_correct, reuse_correct),
                "control_logical_calls": control["metrics"][example_id]["logical_calls"],
                "reuse_logical_calls": reuse["metrics"][example_id]["logical_calls"],
                "reuse_cache_hits": reuse["metrics"][example_id]["cache_hits"],
                "reuse_live_calls": reuse["metrics"][example_id]["live_calls"],
            }
        )
    control_correct = sum(control["grades"].values())
    reuse_correct = sum(reuse["grades"].values())
    control_live = sum(value["live_calls"] for value in control["metrics"].values())
    reuse_live = sum(value["live_calls"] for value in reuse["metrics"].values())
    reuse_hits = sum(value["cache_hits"] for value in reuse["metrics"].values())
    live_reduction = (control_live - reuse_live) / control_live
    checks = {
        "matched_example_ids": ids == set(reuse["responses"])
        and ids == set(control["grades"])
        and ids == set(reuse["grades"]),
        "expected_examples": len(ids) == acceptance["expected_examples"],
        "valid_grades": len(control["grades"]) == len(reuse["grades"]) == len(ids),
        "accuracy_noninferiority": reuse_correct - control_correct
        >= -acceptance["accuracy_allowed_loss_count"],
        "live_call_reduction": live_reduction
        >= acceptance["minimum_live_call_reduction"],
        "reuse_cache_hits": reuse_hits >= acceptance["minimum_cache_hits"],
        "control_cache_hits": sum(
            value["cache_hits"] for value in control["metrics"].values()
        )
        == acceptance["control_cache_hits"],
        "cache_hits_successful": all(
            call.get("success") is True
            for response in reuse["responses"].values()
            for call in response["agent"].get("search_calls", [])
            if call.get("cache_hit")
        ),
    }
    report = {
        "schema_version": 1,
        "stage": "7.3",
        "variant": gate["variant"],
        "official_benchmark_result": False,
        "passed": all(checks.values()),
        "checks": checks,
        "scores": {
            "control_correct": control_correct,
            "reuse_correct": reuse_correct,
            "difference_count": reuse_correct - control_correct,
            "allowed_loss_count": acceptance["accuracy_allowed_loss_count"],
        },
        "efficiency": {
            "control_logical_calls": sum(
                value["logical_calls"] for value in control["metrics"].values()
            ),
            "reuse_logical_calls": sum(
                value["logical_calls"] for value in reuse["metrics"].values()
            ),
            "control_live_calls": control_live,
            "reuse_live_calls": reuse_live,
            "reuse_cache_hits": reuse_hits,
            "reuse_cache_hit_rate": reuse_hits
            / sum(value["logical_calls"] for value in reuse["metrics"].values()),
            "paired_live_call_reduction": live_reduction,
        },
        "transitions": {
            name: sum(pair["transition"] == name for pair in paired)
            for name in (
                "correct_to_correct",
                "correct_to_wrong",
                "wrong_to_correct",
                "wrong_to_wrong",
            )
        },
        "pairs": paired,
        "artifacts": {
            str(args.gate_path): _sha256(args.gate_path),
            str(args.control_dir / "responses.jsonl"): _sha256(
                args.control_dir / "responses.jsonl"
            ),
            str(args.control_dir / "grades.jsonl"): _sha256(
                args.control_dir / "grades.jsonl"
            ),
            str(args.reuse_dir / "responses.jsonl"): _sha256(
                args.reuse_dir / "responses.jsonl"
            ),
            str(args.reuse_dir / "grades.jsonl"): _sha256(
                args.reuse_dir / "grades.jsonl"
            ),
        },
        "interpretation": {
            "cache_savings": "cache hits are deterministic avoided retriever calls within reuse trajectories",
            "score_delta": "independent model trajectories prevent causal attribution to exact reuse",
        },
    }
    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    args.output_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({"passed": report["passed"], "checks": checks}))
    if not report["passed"]:
        raise SystemExit(1)


def _run(directory: Path) -> dict[str, Any]:
    responses = _jsonl_by_id(directory / "responses.jsonl")
    grade_rows = _jsonl_by_id(directory / "grades.jsonl")
    grades = {example_id: bool(row.get("correct")) for example_id, row in grade_rows.items()}
    metrics = {}
    for example_id, response in responses.items():
        calls = response.get("agent", {}).get("search_calls", [])
        hits = sum(bool(call.get("cache_hit")) for call in calls)
        metrics[example_id] = {
            "logical_calls": len(calls),
            "cache_hits": hits,
            "live_calls": len(calls) - hits,
        }
    return {"responses": responses, "grades": grades, "metrics": metrics}


def _jsonl_by_id(path: Path) -> dict[str, dict[str, Any]]:
    rows = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    result = {str(row["example_id"]): row for row in rows}
    if len(result) != len(rows):
        raise ValueError(f"Duplicate example_id in {path}")
    return result


def _transition(control: bool, reuse: bool) -> str:
    return ("correct" if control else "wrong") + "_to_" + (
        "correct" if reuse else "wrong"
    )


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


if __name__ == "__main__":
    main()
