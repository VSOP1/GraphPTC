from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from graphptc.config import ExperimentConfig


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit the Stage 6.3 online gate.")
    parser.add_argument("config_path", type=Path)
    parser.add_argument("gate_path", type=Path)
    parser.add_argument("historical_responses_path", type=Path)
    parser.add_argument("historical_grades_path", type=Path)
    parser.add_argument("responses_path", type=Path)
    parser.add_argument("events_path", type=Path)
    parser.add_argument("active_path", type=Path)
    parser.add_argument("stage62_report_path", type=Path)
    parser.add_argument("output_path", type=Path)
    args = parser.parse_args()

    config = ExperimentConfig.from_toml(args.config_path)
    gate = json.loads(args.gate_path.read_text(encoding="utf-8"))
    historical = _jsonl(args.historical_responses_path)
    historical_grades = _jsonl(args.historical_grades_path)
    gold_rows = _jsonl(config.benchmark.dataset_path)
    responses = _jsonl(args.responses_path)
    events = _jsonl(args.events_path)
    active_rows = _jsonl(args.active_path)
    stage62 = json.loads(args.stage62_report_path.read_text(encoding="utf-8"))
    acceptance = gate["acceptance"]

    expected_primary = [str(value) for value in gate["primary_example_ids"]]
    selected_primary = _select_primary(historical, len(expected_primary))
    response_ids = [str(row["example_id"]) for row in responses]
    active_by_id = {str(row["example_id"]): row["active"] for row in active_rows}
    gold_by_id = {str(row["query_id"]): str(row["answer"]) for row in gold_rows}
    historical_correct = {
        str(row["example_id"]): bool(row.get("correct"))
        for row in historical_grades
    }
    repaired = {
        example_id: active
        for example_id, active in active_by_id.items()
        if active.get("status") == "repaired_active"
    }
    noops = [
        active
        for active in active_by_id.values()
        if active.get("status") == "no_repairable_failure"
    ]
    repair_events = [event for event in events if event.get("type") == "repair.finished"]
    failed_blocks = [
        event
        for event in events
        if event.get("type") == "block.finished"
        and event.get("data", {}).get("success") is False
    ]
    failed_block_ids = {event.get("block_id") for event in failed_blocks}
    repair_statuses = [event.get("data", {}).get("status") for event in repair_events]
    completed_rate = len(responses) / len(expected_primary)
    success_rate = (
        sum(row.get("status") == "success" for row in responses) / len(responses)
        if responses
        else 0.0
    )
    active_errors = [
        active
        for active in active_by_id.values()
        if active.get("status") in {"active_repair_error", "replay_failed"}
    ]
    exact_anchor = all(_exact_anchor(active, failed_blocks) for active in repaired.values())
    stateless_tools_only = all(
        event.get("data", {}).get("tool") in {"search", "fetch"}
        for event in events
        if event.get("type") == "tool.called"
    )
    repaired_traces = [
        block
        for response in responses
        if str(response["example_id"]) in repaired
        for block in (response.get("agent") or {}).get("blocks", [])
        if block.get("success") is True
        and block.get("code")
        == repaired[str(response["example_id"])].get("patched_code")
    ]
    checks = {
        "fewshot_prompt": config.browsecomp_plus.prompt_variant == gate["prompt_variant"],
        "historical_selection_exact": selected_primary == expected_primary,
        "primary_ids_exact": set(response_ids) == set(expected_primary)
        and len(response_ids) == len(expected_primary),
        "reserve_condition_obeyed": bool(repaired)
        and set(response_ids).isdisjoint(set(gate["reserve_example_ids"])),
        "one_active_row_per_response": set(active_by_id) == set(response_ids)
        and len(active_rows) == len(responses),
        "completed_response_rate": completed_rate
        >= acceptance["completed_response_rate_min"],
        "successful_response_rate": success_rate
        >= acceptance["successful_response_rate_min"],
        "fresh_active_repair_count": len(repaired)
        >= acceptance["fresh_active_repair_count_min"],
        "active_repair_success_rate": bool(repair_events)
        and all(status == "repaired_active" for status in repair_statuses),
        "active_error_count": len(active_errors) == acceptance["active_error_count"],
        "repair_request_ceiling": all(
            int(active.get("model_request_count", 0))
            <= acceptance["max_repair_model_requests_per_example"]
            for active in active_by_id.values()
        ),
        "noop_zero_repair_requests": all(
            active.get("model_request_count")
            == acceptance["noop_repair_model_requests"]
            for active in noops
        ),
        "repair_event_record_rate": len(repair_events) == len(repaired)
        and all(event.get("block_id") in failed_block_ids for event in repair_events),
        "exact_failure_patch_anchor": exact_anchor,
        "repaired_observation_cap_preserved": len(repaired_traces) == len(repaired)
        and all(
            len(str(trace.get("stdout", ""))) <= config.runtime.max_stdout_chars
            and (
                trace.get("stdout_truncated") is True
                if int(trace.get("stdout_chars", 0)) > config.runtime.max_stdout_chars
                else trace.get("stdout_truncated") is False
            )
            for trace in repaired_traces
        ),
        "read_only_tool_scope_only": stateless_tools_only,
        "stage62_prerequisite_passed": stage62.get("passed") is True,
    }

    metrics = []
    for response in responses:
        example_id = str(response["example_id"])
        agent = response.get("agent") or {}
        active = active_by_id[example_id]
        prediction = str(response.get("prediction", ""))
        reference_answer = gold_by_id[example_id]
        metrics.append(
            {
                "example_id": example_id,
                "response_status": response.get("status"),
                "prediction": prediction,
                "reference_answer": reference_answer,
                "exact_match": prediction.strip().casefold()
                == reference_answer.strip().casefold(),
                "historical_turn30_grader_correct": historical_correct.get(
                    example_id
                ),
                "active_status": active.get("status"),
                "repair_model_requests": active.get("model_request_count"),
                "model_requests": agent.get("model_requests"),
                "ptc_blocks": agent.get("ptc_blocks"),
                "failed_result_blocks": sum(
                    block.get("success") is False for block in agent.get("blocks", [])
                ),
                "runtime_tool_calls": (agent.get("runtime_session") or {}).get(
                    "tool_calls"
                ),
                "duration_ms": agent.get("duration_ms"),
                "reused_replay_calls": (active.get("replay") or {}).get(
                    "reused_tool_call_count"
                ),
                "executed_replay_calls": (active.get("replay") or {}).get(
                    "executed_tool_call_count"
                ),
            }
        )

    report = {
        "schema_version": 1,
        "stage": "6.3",
        "mode": "bounded-online-active-promotion-gate",
        "official_benchmark_result": False,
        "passed": all(checks.values()),
        "checks": checks,
        "selection": {
            "expected_primary": expected_primary,
            "recomputed_primary": selected_primary,
            "reserve_used": False,
        },
        "summary": {
            "response_count": len(responses),
            "successful_response_count": sum(
                row.get("status") == "success" for row in responses
            ),
            "fresh_active_repair_count": len(repaired),
            "noop_count": len(noops),
            "active_error_count": len(active_errors),
            "repair_event_count": len(repair_events),
            "failed_block_event_count": len(failed_blocks),
            "current_exact_match_count": sum(
                metric["exact_match"] for metric in metrics
            ),
            "historical_turn30_grader_correct_count": sum(
                metric["historical_turn30_grader_correct"] is True
                for metric in metrics
            ),
        },
        "repairs": [
            {
                "example_id": example_id,
                "proposal": active["generated_patch"]["proposal"],
                "reused_tool_call_count": active["replay"]["reused_tool_call_count"],
                "executed_tool_call_count": active["replay"][
                    "executed_tool_call_count"
                ],
                "actions": [
                    event["action"] for event in active["replay"]["tool_events"]
                ],
            }
            for example_id, active in repaired.items()
        ],
        "metrics": metrics,
        "artifacts": {
            path.name: _sha256(path)
            for path in (
                args.config_path,
                args.gate_path,
                args.historical_responses_path,
                args.historical_grades_path,
                args.responses_path,
                args.events_path,
                args.active_path,
                args.stage62_report_path,
            )
        },
    }
    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    args.output_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "passed": report["passed"],
                "check_count": len(checks),
                "output_path": str(args.output_path),
            },
            ensure_ascii=False,
        )
    )
    if not report["passed"]:
        raise SystemExit(1)


def _select_primary(rows: list[dict[str, Any]], count: int) -> list[str]:
    eligible = []
    for row in rows:
        failures = [
            block
            for block in (row.get("agent") or {}).get("blocks", [])
            if block.get("success") is False
        ]
        if len(failures) >= 3 and int(failures[0].get("turn", 999)) <= 2:
            eligible.append((str(row["example_id"]), len(failures)))
    eligible.sort(key=lambda item: (-item[1], int(item[0])))
    return [example_id for example_id, _ in eligible[:count]]


def _exact_anchor(
    active: dict[str, Any], failed_blocks: list[dict[str, Any]]
) -> bool:
    proposal = active.get("generated_patch", {}).get("proposal", {})
    block_id = proposal.get("block_id")
    matching = [event for event in failed_blocks if event.get("block_id") == block_id]
    if len(matching) != 1:
        return False
    location = matching[0].get("data", {}).get("runtime_trace", {}).get(
        "error_location"
    )
    return (
        isinstance(location, dict)
        and int(proposal.get("start_line", -1)) <= int(location.get("line", -2))
        <= int(proposal.get("end_line", -1))
    )


def _jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


if __name__ == "__main__":
    main()
