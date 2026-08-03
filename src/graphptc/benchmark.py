from __future__ import annotations

import hashlib
import json
import platform
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any, Callable, Iterable

from .config import ExperimentConfig, GraderConfig
from .deepsearchqa import (
    DEEPSEARCHQA_DATASET_SHA256,
    DeepSearchQAExample,
    ExampleGrade,
    EvaluationResult,
    GeminiJudge,
    OpenAICompatibleJudge,
    Prediction,
    build_judge_prompt,
    evaluate_predictions,
    load_deepsearchqa,
    load_predictions,
    summarize_grades,
)
from .model import OpenAIChatModel
from .ptc import (
    PTC_TOOL_SPEC,
    SYSTEM_PROMPT,
    USER_PROMPT_TEMPLATE,
    OriginalPTCAgent,
    extract_result_tag,
)
from .search import TavilySearchTools


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


def run_benchmark(
    config: ExperimentConfig,
    *,
    limit: int | None = None,
    example_ids: Iterable[str] | None = None,
    resume: bool = True,
    progress: ProgressCallback | None = None,
) -> BenchmarkRunSummary:
    """Run independent DeepSearchQA tasks and append durable result records."""
    if limit is not None and limit < 1:
        raise ValueError("limit must be at least 1")
    if config.benchmark.workers < 1:
        raise ValueError("benchmark workers must be at least 1")

    examples = load_deepsearchqa(
        config.benchmark.dataset_path,
        download_if_missing=True,
        verify_checksum=True,
    )
    selected = _select_examples(examples, limit=limit, example_ids=example_ids)
    output_path = config.benchmark.responses_path
    output_path.parent.mkdir(parents=True, exist_ok=True)
    run_signature = _run_signature(config)

    selected_ids = {example.example_id for example in selected}
    existing_records = _load_records(output_path) if resume else []
    incompatible_ids = [
        record["example_id"]
        for record in existing_records
        if record.get("run_signature") != run_signature
    ]
    if incompatible_ids:
        raise ValueError(
            "Response file contains records from another or unknown run "
            f"configuration (examples: {incompatible_ids[:5]}). "
            "Use a matching config, a new responses_path, or --restart."
        )
    successful_ids = {
        record["example_id"]
        for record in existing_records
        if _record_succeeded(record)
    }
    pending = [
        example for example in selected if example.example_id not in successful_ids
    ]

    model_api_key = ""
    search_api_key = ""
    if pending:
        model_api_key = config.require_api_key(config.model.api_key_env)
        search_api_key = config.require_api_key(config.search.api_key_env)

    if not resume:
        output_path.write_text("", encoding="utf-8")
    else:
        retained = [
            record
            for record in existing_records
            if record["example_id"] not in selected_ids or _record_succeeded(record)
        ]
        if len(retained) != len(existing_records):
            _write_records(output_path, retained)

    def run_one(example: DeepSearchQAExample) -> dict[str, Any]:
        checkpoint_path = _checkpoint_path(output_path, example.example_id)
        try:
            model = OpenAIChatModel(config.model, model_api_key)
            search_tools = TavilySearchTools(
                search_api_key,
                search_depth=config.search.search_depth,
                default_max_results=config.search.max_results,
                max_tool_calls=config.search.max_tool_calls,
                timeout_seconds=config.search.timeout_seconds,
            )
            result = OriginalPTCAgent(
                model=model,
                search_tools=search_tools,
                runtime=config.runtime,
                checkpoint_callback=lambda snapshot: _write_checkpoint(
                    checkpoint_path,
                    {
                        "run_signature": run_signature,
                        "example_id": example.example_id,
                        "updated_at": datetime.now(UTC).isoformat(),
                        **snapshot,
                    },
                ),
            ).run(example.problem)
            prediction = (
                extract_result_tag(result.answer) if result.status == "success" else None
            )
            status = "success" if prediction is not None else "failed"
            error = result.error
            if result.status == "success" and prediction is None:
                error = "Final answer did not contain a non-empty <result> tag"
            record = {
                "schema_version": 1,
                "run_signature": run_signature,
                "example_id": example.example_id,
                "prediction": prediction or "",
                "status": status,
                "error": error,
                "problem_category": example.problem_category,
                "answer_type": example.answer_type,
                "model": config.model.model,
                "created_at": datetime.now(UTC).isoformat(),
                "agent": result.to_dict(),
            }
            checkpoint_path.unlink(missing_ok=True)
            return record
        except Exception as exc:
            return {
                "schema_version": 1,
                "run_signature": run_signature,
                "example_id": example.example_id,
                "prediction": "",
                "status": "failed",
                "error": f"{type(exc).__name__}: {exc}",
                "problem_category": example.problem_category,
                "answer_type": example.answer_type,
                "model": config.model.model,
                "created_at": datetime.now(UTC).isoformat(),
                "agent": None,
            }

    records: list[dict[str, Any]] = []
    workers = min(config.benchmark.workers, len(pending)) if pending else 0
    if workers:
        with output_path.open("a", encoding="utf-8", newline="\n") as handle:
            with ThreadPoolExecutor(max_workers=workers) as executor:
                futures = [executor.submit(run_one, example) for example in pending]
                for index, future in enumerate(as_completed(futures), start=1):
                    record = future.result()
                    handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                    handle.flush()
                    records.append(record)
                    if progress is not None:
                        progress(index, len(pending), record)

    succeeded = sum(record["status"] == "success" for record in records)
    return BenchmarkRunSummary(
        selected=len(selected),
        completed=len(records),
        succeeded=succeeded,
        failed=len(records) - succeeded,
        skipped_existing=len(selected) - len(pending),
        responses_path=str(output_path),
    )


def evaluate_benchmark(config: ExperimentConfig) -> EvaluationResult:
    """Resume the official Gemini judge and persist each completed grade."""
    examples = load_deepsearchqa(
        config.benchmark.dataset_path,
        verify_checksum=True,
    )
    records = _load_records(config.benchmark.responses_path)
    expected_signature = _run_signature(config)
    signatures = {record.get("run_signature") for record in records}
    if signatures and signatures != {expected_signature}:
        raise ValueError(
            "Response file does not match the current model, prompt, search, "
            "or runtime configuration."
        )
    predictions = load_predictions(config.benchmark.responses_path)
    prediction_index = {
        prediction.example_id: prediction.prediction for prediction in predictions
    }
    cache_keys = {
        example.example_id: _grade_cache_key(
            example,
            prediction_index.get(example.example_id, ""),
            config.grader,
        )
        for example in examples
    }
    reusable_statuses = {"valid", "empty_model_response"}
    cached_grades: dict[str, ExampleGrade] = {}
    retained_grade_records: list[dict[str, Any]] = []
    for record in _load_records(config.benchmark.grades_path):
        example_id = record["example_id"]
        if (
            example_id in cache_keys
            and record.get("cache_key") == cache_keys[example_id]
            and record.get("status") in reusable_statuses
        ):
            cached_grades[example_id] = _grade_from_record(record)
            retained_grade_records.append(record)

    pending = [
        example for example in examples if example.example_id not in cached_grades
    ]
    if pending:
        judge = _create_judge(config)
    config.benchmark.grades_path.parent.mkdir(parents=True, exist_ok=True)
    _write_records(config.benchmark.grades_path, retained_grade_records)
    new_grades: dict[str, ExampleGrade] = {}
    if pending:
        pending_predictions = [
            Prediction(
                example_id=example.example_id,
                prediction=prediction_index.get(example.example_id, ""),
            )
            for example in pending
        ]
        with config.benchmark.grades_path.open(
            "a", encoding="utf-8", newline="\n"
        ) as grade_file:

            def persist_grade(grade: ExampleGrade) -> None:
                record = {
                    "cache_key": cache_keys[grade.example_id],
                    "grader_model": config.grader.model,
                    **grade.to_dict(),
                }
                grade_file.write(json.dumps(record, ensure_ascii=False) + "\n")
                grade_file.flush()
                new_grades[grade.example_id] = grade

            evaluate_predictions(
                pending,
                pending_predictions,
                judge,
                max_workers=config.grader.workers,
                on_grade=persist_grade,
            )

    grade_index = {**cached_grades, **new_grades}
    ordered_grades = tuple(grade_index[example.example_id] for example in examples)
    summary = summarize_grades(ordered_grades)
    result = EvaluationResult(grades=ordered_grades, summary=summary)
    ordered_records = [
        {
            "cache_key": cache_keys[grade.example_id],
            "grader_model": config.grader.model,
            **grade.to_dict(),
        }
        for grade in ordered_grades
    ]
    _write_records(config.benchmark.grades_path, ordered_records)

    report = {
        "schema_version": 1,
        "dataset_sha256": DEEPSEARCHQA_DATASET_SHA256,
        "run_signature": expected_signature,
        "run_configuration": _run_signature_payload(config),
        "model": config.model.model,
        "grader_provider": config.grader.provider,
        "grader_model": config.grader.model,
        "created_at": datetime.now(UTC).isoformat(),
        "generation": _summarize_generation(records),
        "summary": result.summary.to_dict(),
    }
    config.benchmark.report_path.parent.mkdir(parents=True, exist_ok=True)
    config.benchmark.report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return result


def _select_examples(
    examples: list[DeepSearchQAExample],
    *,
    limit: int | None,
    example_ids: Iterable[str] | None,
) -> list[DeepSearchQAExample]:
    requested_ids = list(dict.fromkeys(example_ids or ()))
    if requested_ids:
        index = {example.example_id: example for example in examples}
        unknown = [example_id for example_id in requested_ids if example_id not in index]
        if unknown:
            raise ValueError(f"Unknown DeepSearchQA example IDs: {unknown[:5]}")
        selected = [index[example_id] for example_id in requested_ids]
    else:
        selected = examples
    return selected[:limit] if limit is not None else selected


def _load_records(
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


def _record_succeeded(record: dict[str, Any]) -> bool:
    status = record.get("status")
    if status is not None:
        return status == "success"
    return bool(record.get("prediction"))


def _grade_cache_key(
    example: DeepSearchQAExample,
    prediction: str,
    grader: GraderConfig,
) -> str:
    payload = {
        "grader": asdict(grader),
        "judge_prompt": build_judge_prompt(example, prediction),
        "packages": {
            "openai": _package_version("openai"),
            "google-genai": _package_version("google-genai"),
        },
    }
    serialized = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(serialized).hexdigest()


def _create_judge(config: ExperimentConfig) -> Any:
    api_key = config.require_api_key(config.grader.api_key_env)
    if config.grader.provider == "openai_compatible":
        return OpenAICompatibleJudge(
            api_key=api_key,
            model=config.grader.model,
            base_url=config.grader.base_url,
            max_retries=config.grader.max_retries,
            max_completion_tokens=config.grader.max_completion_tokens,
            thinking=config.grader.thinking,
            timeout_seconds=config.grader.timeout_seconds,
        )
    if config.grader.provider == "gemini":
        return GeminiJudge(
            api_key=api_key,
            model=config.grader.model,
            max_retries=config.grader.max_retries,
        )
    raise ValueError(f"Unsupported grader provider: {config.grader.provider}")


def _summarize_generation(records: list[dict[str, Any]]) -> dict[str, Any]:
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


def _grade_from_record(record: dict[str, Any]) -> ExampleGrade:
    try:
        return ExampleGrade(
            example_id=record["example_id"],
            status=record["status"],
            precision=record.get("precision"),
            recall=record.get("recall"),
            f1_score=record.get("f1_score"),
            explanation=record.get("explanation"),
            correctness_details=record.get("correctness_details"),
            excessive_answers=record.get("excessive_answers"),
            raw_judge_response=record.get("raw_judge_response", ""),
            error=record.get("error"),
        )
    except (KeyError, TypeError) as exc:
        raise ValueError(f"Malformed cached grade record: {exc}") from exc


def _write_records(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    content = "".join(
        json.dumps(record, ensure_ascii=False) + "\n" for record in records
    )
    temporary_path.write_text(content, encoding="utf-8")
    temporary_path.replace(path)


def _checkpoint_path(responses_path: Path, example_id: str) -> Path:
    name = hashlib.sha256(example_id.encode()).hexdigest()[:20]
    return responses_path.parent / "checkpoints" / f"{name}.json"


def _write_checkpoint(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _run_signature(config: ExperimentConfig) -> str:
    payload = _run_signature_payload(config)
    serialized = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(serialized).hexdigest()


def _run_signature_payload(config: ExperimentConfig) -> dict[str, Any]:
    return {
        "dataset_sha256": DEEPSEARCHQA_DATASET_SHA256,
        "model": asdict(config.model),
        "search": asdict(config.search),
        "runtime": asdict(config.runtime),
        "system_prompt": SYSTEM_PROMPT,
        "user_prompt_template": USER_PROMPT_TEMPLATE,
        "ptc_tool_spec": PTC_TOOL_SPEC,
        "implementation_sha256": _implementation_sha256(),
        "python": platform.python_version(),
        "packages": {
            name: _package_version(name)
            for name in ("graphptc", "openai", "tavily-python", "toolregistry", "codecell")
        },
    }


def _implementation_sha256() -> str:
    digest = hashlib.sha256()
    package_dir = Path(__file__).resolve().parent
    for name in ("config.py", "model.py", "ptc.py", "search.py"):
        digest.update(name.encode("utf-8"))
        digest.update((package_dir / name).read_bytes())
    return digest.hexdigest()


def _package_version(name: str) -> str:
    try:
        return version(name)
    except PackageNotFoundError:
        return "unknown"
