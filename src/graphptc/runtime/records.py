from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable


@dataclass(frozen=True)
class BenchmarkRunSummary:
    selected: int
    completed: int
    succeeded: int
    failed: int
    skipped_existing: int
    responses_path: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


ProgressCallback = Callable[[int, int, dict[str, Any]], None]


def load_records(
    path: Path, *, recover_truncated_tail: bool = False
) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    if recover_truncated_tail:
        _recover_truncated_jsonl_tail(path)
    records: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid response JSON on line {line_number}: {exc}"
                ) from exc
            example_id = record.get("example_id") if isinstance(record, dict) else None
            if not isinstance(example_id, str):
                raise ValueError(
                    f"Response line {line_number} has no string example_id"
                )
            if example_id in seen_ids:
                raise ValueError(f"Duplicate response example_id: {example_id}")
            seen_ids.add(example_id)
            records.append(record)
    return records


def _recover_truncated_jsonl_tail(path: Path) -> None:
    data = path.read_bytes()
    if not data or data.endswith(b"\n"):
        return
    last_newline = data.rfind(b"\n")
    tail = data[last_newline + 1 :]
    try:
        json.loads(tail.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        path.write_bytes(data[: last_newline + 1])


def record_succeeded(record: dict[str, Any]) -> bool:
    status = record.get("status")
    if status is not None:
        return status == "success"
    return bool(record.get("prediction"))


def summarize_generation(records: list[dict[str, Any]]) -> dict[str, Any]:
    agents = [
        record["agent"]
        for record in records
        if isinstance(record.get("agent"), dict)
    ]
    usage_fields = (
        "input_tokens",
        "output_tokens",
        "cached_input_tokens",
        "reasoning_tokens",
    )
    duration_total = sum(float(agent.get("duration_ms", 0.0)) for agent in agents)
    blocks = [
        block
        for agent in agents
        for block in agent.get("blocks", [])
        if isinstance(block, dict)
    ]
    search_calls = [
        call
        for agent in agents
        for call in agent.get("search_calls", [])
        if isinstance(call, dict)
    ]
    requests = [
        request
        for agent in agents
        for request in agent.get("requests", [])
        if isinstance(request, dict)
    ]
    runtime_calls = [int(block.get("runtime_calls", 0)) for block in blocks]
    repeated_search_queries = 0
    repeated_result_slots = 0
    total_result_slots = 0
    searches_without_new_docids = 0
    repeated_fetches = 0
    for agent in agents:
        seen_docids: set[str] = set()
        fetched_docids: set[str] = set()
        queries = [
            str(call.get("query", "")).strip().casefold()
            for call in agent.get("search_calls", [])
            if isinstance(call, dict)
            and call.get("operation") == "search"
            and str(call.get("query", "")).strip()
        ]
        repeated_search_queries += len(queries) - len(set(queries))
        for call in agent.get("search_calls", []):
            if not isinstance(call, dict) or not call.get("success", True):
                continue
            operation = call.get("operation")
            if operation == "search":
                docids = [str(value) for value in call.get("docids", [])]
                new_docids = [value for value in docids if value not in seen_docids]
                total_result_slots += len(docids)
                repeated_result_slots += len(docids) - len(new_docids)
                searches_without_new_docids += not new_docids
                seen_docids.update(docids)
            elif operation == "fetch":
                docid = str(call.get("docid", ""))
                if docid and docid in fetched_docids:
                    repeated_fetches += 1
                if docid:
                    fetched_docids.add(docid)
    analyses = [
        block.get("program_analysis") or {}
        for block in blocks
        if isinstance(block.get("program_analysis") or {}, dict)
    ]
    compactions = [
        item
        for agent in agents
        for item in agent.get("compactions", [])
        if isinstance(item, dict)
    ]
    attempts = [
        attempt
        for request in requests
        for attempt in request.get("attempts", [])
        if isinstance(attempt, dict)
    ]
    runtime_sessions = [
        agent.get("runtime_session")
        for agent in agents
        if isinstance(agent.get("runtime_session"), dict)
        and agent.get("runtime_session")
    ]
    persistent_sessions = [
        item for item in runtime_sessions if item.get("persistent") is True
    ]
    direct_sessions = [
        item for item in runtime_sessions if item.get("mode") == "direct_tool_calling"
    ]
    search_operations = [
        call for call in search_calls if call.get("operation") == "search"
    ]
    fetch_operations = [
        call for call in search_calls if call.get("operation") == "fetch"
    ]
    tool_output_chars = sum(int(call.get("output_chars", 0)) for call in search_calls)
    stdout_chars = sum(int(block.get("stdout_chars", 0)) for block in blocks)
    return {
        "total_records": len(records),
        "successful": sum(record.get("status") == "success" for record in records),
        "failed": sum(record.get("status") != "success" for record in records),
        "model_requests": sum(int(agent.get("model_requests", 0)) for agent in agents),
        "ptc_blocks": len(blocks),
        "successful_ptc_blocks": sum(bool(block.get("success")) for block in blocks),
        "failed_ptc_blocks": sum(not bool(block.get("success")) for block in blocks),
        "tool_calls": len(search_calls),
        "search_calls": len(search_operations),
        "fetch_calls": len(fetch_operations),
        "runtime_calls": sum(runtime_calls),
        "zero_call_ptc_blocks": sum(count == 0 for count in runtime_calls),
        "single_call_ptc_blocks": sum(count == 1 for count in runtime_calls),
        "multi_call_ptc_blocks": sum(count > 1 for count in runtime_calls),
        "mean_runtime_calls_per_ptc_block": (
            sum(runtime_calls) / len(runtime_calls) if runtime_calls else None
        ),
        "repeated_exact_search_queries": repeated_search_queries,
        "searches_without_new_docids": searches_without_new_docids,
        "repeated_result_slots": repeated_result_slots,
        "total_result_slots": total_result_slots,
        "repeated_result_slot_rate": (
            repeated_result_slots / total_result_slots if total_result_slots else None
        ),
        "repeated_fetches": repeated_fetches,
        "tool_output_chars": tool_output_chars,
        "ptc_stdout_chars": stdout_chars,
        "stdout_to_tool_output_ratio": (
            stdout_chars / tool_output_chars if tool_output_chars else None
        ),
        "blocks_with_tool_calls_in_loops": sum(
            int(analysis.get("tool_calls_in_loops", 0)) > 0 for analysis in analyses
        ),
        "blocks_with_conditional_tool_calls": sum(
            int(analysis.get("conditional_tool_calls", 0)) > 0
            for analysis in analyses
        ),
        "blocks_with_dedup": sum(bool(analysis.get("has_dedup")) for analysis in analyses),
        "blocks_with_filtering": sum(
            bool(analysis.get("has_filter")) for analysis in analyses
        ),
        "blocks_with_aggregation": sum(
            bool(analysis.get("has_aggregation")) for analysis in analyses
        ),
        "compaction_requests": sum(
            int(agent.get("compaction_requests", 0)) for agent in agents
        ),
        "successful_compactions": sum(bool(item.get("success")) for item in compactions),
        "direct_tool_sessions": len(direct_sessions),
        "direct_tool_rounds": sum(
            int(item.get("tool_rounds", 0)) for item in direct_sessions
        ),
        "direct_model_tool_calls": sum(
            int(item.get("direct_tool_calls", 0)) for item in direct_sessions
        ),
        "direct_tool_observation_chars": sum(
            int(item.get("tool_observation_chars", 0)) for item in direct_sessions
        ),
        "persistent_runtime_sessions": len(persistent_sessions),
        "persistent_runtime_process_starts": sum(
            int(item.get("process_starts", 0)) for item in persistent_sessions
        ),
        "persistent_runtime_restarts": sum(
            max(0, int(item.get("process_starts", 0)) - 1)
            for item in persistent_sessions
        ),
        "persistent_runtime_executions": sum(
            int(item.get("executions", 0)) for item in persistent_sessions
        ),
        "persistent_runtime_timeouts": sum(
            int(item.get("timeouts", 0)) for item in persistent_sessions
        ),
        "persistent_runtime_protocol_errors": sum(
            int(item.get("protocol_errors", 0)) for item in persistent_sessions
        ),
        "compaction_chars_before": sum(
            int(item.get("before_chars", 0)) for item in compactions
        ),
        "compaction_chars_after": sum(
            int(item.get("after_chars", 0)) for item in compactions
        ),
        "model_attempts": len(attempts),
        "failed_model_attempts": sum(
            attempt.get("status") == "failed" for attempt in attempts
        ),
        "duration_ms": duration_total,
        "mean_duration_ms": duration_total / len(agents) if agents else None,
        "ptc_duration_ms": sum(float(block.get("duration_ms", 0.0)) for block in blocks),
        "search_duration_ms": sum(
            float(call.get("duration_ms", 0.0)) for call in search_calls
        ),
        "model_request_duration_ms": sum(
            float(request.get("duration_ms", 0.0)) for request in requests
        ),
        "max_context_chars": max(
            (int(request.get("context_chars", 0)) for request in requests),
            default=0,
        ),
        "max_request_input_tokens": max(
            (
                int((request.get("usage") or {}).get("input_tokens", 0))
                for request in requests
            ),
            default=0,
        ),
        "usage": {
            field: sum(
                int((agent.get("usage") or {}).get(field, 0)) for agent in agents
            )
            for field in usage_fields
        },
    }


def write_records(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    content = "".join(
        json.dumps(record, ensure_ascii=False) + "\n" for record in records
    )
    temporary_path.write_text(content, encoding="utf-8")
    temporary_path.replace(path)
