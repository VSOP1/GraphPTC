from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
from pathlib import Path
from typing import Any

from graphptc.browsecomp_plus_benchmark import _retriever_metadata, _run_signature_payload
from graphptc.config import ExperimentConfig


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit matched GraphPTC pilot20.")
    parser.add_argument("gate_path", type=Path)
    parser.add_argument("control_config", type=Path)
    parser.add_argument("active_config", type=Path)
    parser.add_argument("control_dir", type=Path)
    parser.add_argument("active_dir", type=Path)
    parser.add_argument("freeze_manifest", type=Path)
    parser.add_argument("output_path", type=Path)
    args = parser.parse_args()

    gate = json.loads(args.gate_path.read_text(encoding="utf-8"))
    acceptance = gate["acceptance"]
    control_config = ExperimentConfig.from_toml(args.control_config)
    active_config = ExperimentConfig.from_toml(args.active_config)
    metadata = _retriever_metadata(control_config)
    payload_equal = _run_signature_payload(
        control_config, metadata
    ) == _run_signature_payload(active_config, metadata)

    control = _run(args.control_dir)
    active = _run(args.active_dir, include_active=True)
    expected = int(acceptance["expected_examples"])
    control_ids = set(control["responses"])
    active_ids = set(active["responses"])
    repaired_ids = {
        qid
        for qid, value in active["active"].items()
        if value.get("status") == "repaired_active"
    }
    attempts = {
        qid
        for qid, value in active["active"].items()
        if value.get("status") != "no_repairable_failure"
    }
    active_errors = [
        value
        for value in active["active"].values()
        if value.get("status") in {"active_repair_error", "replay_failed"}
    ]
    repair_events = [
        event for event in active["events"] if event.get("type") == "repair.finished"
    ]
    forbidden_tools = [
        event
        for event in active["events"]
        if event.get("type") == "tool.called"
        and event.get("data", {}).get("tool") not in {"search", "fetch"}
    ]
    control_correct = sum(
        grade.get("correct") is True for grade in control["grades"].values()
    )
    active_correct = sum(
        grade.get("correct") is True for grade in active["grades"].values()
    )
    allowed_loss = int(acceptance["active_accuracy_allowed_loss_count"])
    repair_success_rate = (
        sum(
            active["active"][qid].get("status") == "repaired_active"
            for qid in active["active"]
            if int(active["active"][qid].get("model_request_count", 0)) > 0
        )
        / sum(
            int(value.get("model_request_count", 0)) > 0
            for value in active["active"].values()
        )
    )
    checks = {
        "matched_payloads": payload_equal,
        "matched_response_ids": control_ids == active_ids
        and len(control_ids) == expected,
        "complete_responses": len(control["responses"]) == expected
        and len(active["responses"]) == expected,
        "valid_grades": len(control["grades"]) == expected
        and len(active["grades"]) == expected
        and all(row.get("status") == "valid" for row in control["grades"].values())
        and all(row.get("status") == "valid" for row in active["grades"].values()),
        "uniform_matched_signature": len(control["signatures"])
        == len(active["signatures"])
        == 1
        and control["signatures"] == active["signatures"],
        "active_accuracy_noninferiority": active_correct
        >= control_correct - allowed_loss,
        "active_repair_success_rate": repair_success_rate
        == acceptance["active_repair_success_rate"],
        "active_error_count": len(active_errors) == acceptance["active_error_count"],
        "repair_request_ceiling": all(
            int(value.get("model_request_count", 0))
            <= acceptance["max_repair_model_requests_per_example"]
            for value in active["active"].values()
        ),
        "noop_zero_repair_requests": all(
            int(value.get("model_request_count", 0))
            == acceptance["noop_repair_model_requests"]
            for value in active["active"].values()
            if value.get("status") != "repaired_active"
        ),
        "repair_event_record_rate": len(repair_events) == len(attempts),
        "forbidden_tool_count": len(forbidden_tools)
        == acceptance["forbidden_tool_count"],
        "repaired_observation_cap_preserved": _observation_cap_preserved(
            active, repaired_ids, active_config.runtime.max_stdout_chars
        ),
    }

    pairs = []
    for qid in sorted(control_ids, key=int):
        control_response = control["responses"][qid]
        active_response = active["responses"][qid]
        control_grade = control["grades"][qid]
        active_grade = active["grades"][qid]
        active_status = active["active"][qid].get("status")
        pairs.append(
            {
                "example_id": qid,
                "active_status": active_status,
                "control_correct": control_grade.get("correct"),
                "active_correct": active_grade.get("correct"),
                "transition": _transition(control_grade, active_grade),
                "control_prediction": control_response.get("prediction"),
                "active_prediction": active_response.get("prediction"),
                "control": _episode_metrics(control_response),
                "active": _episode_metrics(active_response),
                "replay": _replay_metrics(active["active"][qid]),
            }
        )

    report = {
        "schema_version": 1,
        "variant": "graphptc-active-v1",
        "benchmark": "browsecomp_plus",
        "scope": "matched_turn30_pilot20",
        "official_benchmark_result": False,
        "passed": all(checks.values()),
        "checks": checks,
        "scores": {
            "control_correct": control_correct,
            "active_correct": active_correct,
            "control_accuracy": control_correct / expected,
            "active_accuracy": active_correct / expected,
            "difference_count": active_correct - control_correct,
            "allowed_loss_count": allowed_loss,
        },
        "active_repair": {
            "attempt_count": len(attempts),
            "model_repair_count": sum(
                int(value.get("model_request_count", 0)) > 0
                for value in active["active"].values()
            ),
            "committed_count": len(repaired_ids),
            "not_repairable_count": sum(
                value.get("status") == "not_repairable"
                for value in active["active"].values()
            ),
            "error_count": len(active_errors),
            "reused_tool_calls": sum(
                int((value.get("replay") or {}).get("reused_tool_call_count", 0))
                for value in active["active"].values()
            ),
            "executed_tool_calls": sum(
                int((value.get("replay") or {}).get("executed_tool_call_count", 0))
                for value in active["active"].values()
            ),
            "repaired_control_correct": sum(
                control["grades"][qid].get("correct") is True for qid in repaired_ids
            ),
            "repaired_active_correct": sum(
                active["grades"][qid].get("correct") is True for qid in repaired_ids
            ),
        },
        "retrieval": {
            "control_candidate_recall": control["report"]["summary"].get(
                "candidate_retrieval_recall"
            ),
            "active_candidate_recall": active["report"]["summary"].get(
                "candidate_retrieval_recall"
            ),
            "control_fetched_evidence_recall": control["report"]["summary"].get(
                "fetched_evidence_recall"
            ),
            "active_fetched_evidence_recall": active["report"]["summary"].get(
                "fetched_evidence_recall"
            ),
        },
        "telemetry": {
            "control": _aggregate(control["responses"].values()),
            "active": _aggregate(active["responses"].values()),
        },
        "transitions": {
            name: sum(pair["transition"] == name for pair in pairs)
            for name in ("correct_to_correct", "correct_to_wrong", "wrong_to_correct", "wrong_to_wrong")
        },
        "pairs": pairs,
        "artifacts": {
            str(path): _sha256(path)
            for path in (
                args.gate_path,
                args.control_config,
                args.active_config,
                args.freeze_manifest,
                args.control_dir / "responses.jsonl",
                args.control_dir / "grades.jsonl",
                args.control_dir / "report.json",
                args.control_dir / "events.jsonl",
                args.active_dir / "responses.jsonl",
                args.active_dir / "grades.jsonl",
                args.active_dir / "report.json",
                args.active_dir / "events.jsonl",
                args.active_dir / "active.jsonl",
            )
        },
    }
    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    args.output_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({"passed": report["passed"], "checks": len(checks)}))
    if not report["passed"]:
        raise SystemExit(1)


def _run(directory: Path, *, include_active: bool = False) -> dict[str, Any]:
    responses = {row["example_id"]: row for row in _jsonl(directory / "responses.jsonl")}
    result = {
        "responses": responses,
        "grades": {row["example_id"]: row for row in _jsonl(directory / "grades.jsonl")},
        "report": json.loads((directory / "report.json").read_text(encoding="utf-8")),
        "events": _jsonl(directory / "events.jsonl"),
        "signatures": {row.get("run_signature") for row in responses.values()},
    }
    if include_active:
        result["active"] = {
            row["example_id"]: row["active"] for row in _jsonl(directory / "active.jsonl")
        }
    return result


def _episode_metrics(response: dict[str, Any]) -> dict[str, Any]:
    agent = response.get("agent") or {}
    calls = agent.get("search_calls", [])
    searches = [call for call in calls if call.get("operation") == "search"]
    queries = [str(call.get("query")) for call in searches]
    slots = [str(docid) for call in searches for docid in call.get("docids", [])]
    blocks = int(agent.get("ptc_blocks", 0))
    runtime_calls = int((agent.get("runtime_session") or {}).get("tool_calls", 0))
    return {
        "runtime_tool_calls": runtime_calls,
        "live_tool_calls": len(calls),
        "calls_per_block": runtime_calls / blocks if blocks else 0.0,
        "model_requests": agent.get("model_requests"),
        "ptc_blocks": blocks,
        "failed_blocks": sum(
            block.get("success") is False for block in agent.get("blocks", [])
        ),
        "duration_ms": agent.get("duration_ms"),
        "input_tokens": (agent.get("usage") or {}).get("input_tokens"),
        "output_tokens": (agent.get("usage") or {}).get("output_tokens"),
        "repeated_searches": len(queries) - len(set(queries)),
        "repeated_result_slots": len(slots) - len(set(slots)),
    }


def _aggregate(responses: Any) -> dict[str, Any]:
    metrics = [_episode_metrics(response) for response in responses]
    names = (
        "runtime_tool_calls",
        "live_tool_calls",
        "calls_per_block",
        "model_requests",
        "ptc_blocks",
        "failed_blocks",
        "duration_ms",
        "input_tokens",
        "output_tokens",
        "repeated_searches",
        "repeated_result_slots",
    )
    return {
        name: {
            "total": sum(float(metric[name] or 0) for metric in metrics),
            "median": statistics.median(float(metric[name] or 0) for metric in metrics),
            "p90": _p90([float(metric[name] or 0) for metric in metrics]),
        }
        for name in names
    }


def _p90(values: list[float]) -> float:
    ordered = sorted(values)
    return ordered[max(0, math.ceil(0.9 * len(ordered)) - 1)]


def _replay_metrics(active: dict[str, Any]) -> dict[str, Any] | None:
    replay = active.get("replay")
    if not isinstance(replay, dict):
        return None
    return {
        "reused_tool_calls": replay.get("reused_tool_call_count"),
        "executed_tool_calls": replay.get("executed_tool_call_count"),
    }


def _transition(control: dict[str, Any], active: dict[str, Any]) -> str:
    return ("correct" if control.get("correct") else "wrong") + "_to_" + (
        "correct" if active.get("correct") else "wrong"
    )


def _observation_cap_preserved(
    active: dict[str, Any], repaired_ids: set[str], maximum: int
) -> bool:
    for qid in repaired_ids:
        code = active["active"][qid].get("patched_code")
        traces = [
            block
            for block in (active["responses"][qid].get("agent") or {}).get("blocks", [])
            if block.get("code") == code and block.get("success") is True
        ]
        if len(traces) != 1 or len(str(traces[0].get("stdout", ""))) > maximum:
            return False
    return True


def _jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


if __name__ == "__main__":
    main()
