from __future__ import annotations

import hashlib
import json
import platform
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict
from datetime import UTC, datetime
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any, Iterable

from .benchmark import (
    BenchmarkRunSummary,
    ProgressCallback,
    _load_records,
    _record_succeeded,
    _summarize_generation,
    _write_records,
)
from .browsecomp import (
    BROWSECOMP_DATASET_SHA256,
    BrowseCompEvaluationResult,
    BrowseCompExample,
    BrowseCompGrade,
    BrowseCompPrediction,
    OpenAICompatibleBrowseCompJudge,
    build_browsecomp_grader_prompt,
    evaluate_browsecomp_predictions,
    load_browsecomp,
    summarize_browsecomp_grades,
)
from .config import ExperimentConfig, GraderConfig
from .model import OpenAIChatModel
from .ptc import (
    PTC_TOOL_SPEC,
    SYSTEM_PROMPT,
    USER_PROMPT_TEMPLATE,
    OriginalPTCAgent,
    extract_result_tag,
)
from .search import TavilySearchTools


def run_browsecomp_benchmark(
    config: ExperimentConfig,
    *,
    limit: int | None = None,
    example_ids: Iterable[str] | None = None,
    resume: bool = True,
    progress: ProgressCallback | None = None,
) -> BenchmarkRunSummary:
    if limit is not None and limit < 1:
        raise ValueError("limit must be at least 1")
    if config.benchmark.workers < 1:
        raise ValueError("benchmark workers must be at least 1")

    examples = load_browsecomp(
        config.benchmark.dataset_path,
        download_if_missing=True,
        verify_checksum=True,
    )
    selected = _select_examples(examples, limit=limit, example_ids=example_ids)
    output_path = config.benchmark.responses_path
    output_path.parent.mkdir(parents=True, exist_ok=True)
    run_signature = _browsecomp_run_signature(config)

    selected_ids = {example.example_id for example in selected}
    existing_records = _load_records(output_path) if resume else []
    incompatible_ids = [
        record["example_id"]
        for record in existing_records
        if record.get("run_signature") != run_signature
    ]
    if incompatible_ids:
        raise ValueError(
            "BrowseComp response file contains records from another run "
            f"configuration (examples: {incompatible_ids[:5]})."
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

    def run_one(example: BrowseCompExample) -> dict[str, Any]:
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
            ).run(example.problem)
            prediction = (
                extract_result_tag(result.answer) if result.status == "success" else None
            )
            status = "success" if prediction is not None else "failed"
            error = result.error
            if result.status == "success" and prediction is None:
                error = "Final answer did not contain a non-empty <result> tag"
            return {
                "schema_version": 1,
                "benchmark": "browsecomp",
                "run_signature": run_signature,
                "example_id": example.example_id,
                "prediction": prediction or "",
                "status": status,
                "error": error,
                "problem_topic": example.problem_topic,
                "model": config.model.model,
                "created_at": datetime.now(UTC).isoformat(),
                "agent": result.to_dict(),
            }
        except Exception as exc:
            return {
                "schema_version": 1,
                "benchmark": "browsecomp",
                "run_signature": run_signature,
                "example_id": example.example_id,
                "prediction": "",
                "status": "failed",
                "error": f"{type(exc).__name__}: {exc}",
                "problem_topic": example.problem_topic,
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


def evaluate_browsecomp_benchmark(
    config: ExperimentConfig,
) -> BrowseCompEvaluationResult:
    examples = load_browsecomp(config.benchmark.dataset_path, verify_checksum=True)
    records = _load_records(config.benchmark.responses_path)
    expected_signature = _browsecomp_run_signature(config)
    signatures = {record.get("run_signature") for record in records}
    if signatures and signatures != {expected_signature}:
        raise ValueError(
            "BrowseComp response file does not match the current run configuration."
        )
    prediction_index = {
        record["example_id"]: str(record.get("prediction", "")) for record in records
    }
    cache_keys = {
        example.example_id: _browsecomp_grade_cache_key(
            example,
            prediction_index.get(example.example_id, ""),
            config.grader,
        )
        for example in examples
    }

    reusable_statuses = {"valid", "empty_model_response"}
    cached_grades: dict[str, BrowseCompGrade] = {}
    retained_grade_records: list[dict[str, Any]] = []
    for record in _load_records(config.benchmark.grades_path):
        example_id = record["example_id"]
        if (
            example_id in cache_keys
            and record.get("cache_key") == cache_keys[example_id]
            and record.get("status") in reusable_statuses
        ):
            cached_grades[example_id] = _browsecomp_grade_from_record(record)
            retained_grade_records.append(record)

    pending = [
        example for example in examples if example.example_id not in cached_grades
    ]
    config.benchmark.grades_path.parent.mkdir(parents=True, exist_ok=True)
    _write_records(config.benchmark.grades_path, retained_grade_records)
    new_grades: dict[str, BrowseCompGrade] = {}
    if pending:
        judge = _create_browsecomp_judge(config)
        predictions = [
            BrowseCompPrediction(
                example_id=example.example_id,
                prediction=prediction_index.get(example.example_id, ""),
            )
            for example in pending
        ]
        with config.benchmark.grades_path.open(
            "a", encoding="utf-8", newline="\n"
        ) as grade_file:

            def persist_grade(grade: BrowseCompGrade) -> None:
                record = {
                    "cache_key": cache_keys[grade.example_id],
                    "grader_model": config.grader.model,
                    **grade.to_dict(),
                }
                grade_file.write(json.dumps(record, ensure_ascii=False) + "\n")
                grade_file.flush()
                new_grades[grade.example_id] = grade

            evaluate_browsecomp_predictions(
                pending,
                predictions,
                judge,
                max_workers=config.grader.workers,
                on_grade=persist_grade,
            )

    grade_index = {**cached_grades, **new_grades}
    grades = tuple(grade_index[example.example_id] for example in examples)
    result = BrowseCompEvaluationResult(
        grades=grades,
        summary=summarize_browsecomp_grades(grades),
    )
    _write_records(
        config.benchmark.grades_path,
        [
            {
                "cache_key": cache_keys[grade.example_id],
                "grader_model": config.grader.model,
                **grade.to_dict(),
            }
            for grade in grades
        ],
    )

    report = {
        "schema_version": 1,
        "benchmark": "browsecomp",
        "dataset_sha256": BROWSECOMP_DATASET_SHA256,
        "run_signature": expected_signature,
        "run_configuration": _browsecomp_run_signature_payload(config),
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
    examples: list[BrowseCompExample],
    *,
    limit: int | None,
    example_ids: Iterable[str] | None,
) -> list[BrowseCompExample]:
    requested_ids = list(dict.fromkeys(example_ids or ()))
    if requested_ids:
        index = {example.example_id: example for example in examples}
        unknown = [example_id for example_id in requested_ids if example_id not in index]
        if unknown:
            raise ValueError(f"Unknown BrowseComp example IDs: {unknown[:5]}")
        selected = [index[example_id] for example_id in requested_ids]
    else:
        selected = examples
    return selected[:limit] if limit is not None else selected


def _browsecomp_grade_cache_key(
    example: BrowseCompExample,
    prediction: str,
    grader: GraderConfig,
) -> str:
    payload = {
        "grader": asdict(grader),
        "judge_prompt": build_browsecomp_grader_prompt(example, prediction),
        "openai": _package_version("openai"),
    }
    serialized = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(serialized).hexdigest()


def _create_browsecomp_judge(
    config: ExperimentConfig,
) -> OpenAICompatibleBrowseCompJudge:
    if config.grader.provider != "openai_compatible":
        raise ValueError("BrowseComp currently requires an OpenAI-compatible grader")
    return OpenAICompatibleBrowseCompJudge(
        api_key=config.require_api_key(config.grader.api_key_env),
        model=config.grader.model,
        base_url=config.grader.base_url,
        max_retries=config.grader.max_retries,
        max_completion_tokens=config.grader.max_completion_tokens,
        thinking=config.grader.thinking,
        timeout_seconds=config.grader.timeout_seconds,
    )


def _browsecomp_grade_from_record(record: dict[str, Any]) -> BrowseCompGrade:
    try:
        return BrowseCompGrade(
            example_id=record["example_id"],
            status=record["status"],
            grader_letter=record.get("grader_letter"),
            accuracy=float(record.get("accuracy", 0.0)),
            raw_judge_response=record.get("raw_judge_response", ""),
            error=record.get("error"),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"Malformed cached BrowseComp grade: {exc}") from exc


def _browsecomp_run_signature(config: ExperimentConfig) -> str:
    serialized = json.dumps(
        _browsecomp_run_signature_payload(config),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(serialized).hexdigest()


def _browsecomp_run_signature_payload(config: ExperimentConfig) -> dict[str, Any]:
    return {
        "benchmark": "browsecomp",
        "dataset_sha256": BROWSECOMP_DATASET_SHA256,
        "model": asdict(config.model),
        "search": asdict(config.search),
        "runtime": asdict(config.runtime),
        "system_prompt": SYSTEM_PROMPT,
        "user_prompt_template": USER_PROMPT_TEMPLATE,
        "ptc_tool_spec": PTC_TOOL_SPEC,
        "implementation_sha256": _browsecomp_implementation_sha256(),
        "python": platform.python_version(),
        "packages": {
            name: _package_version(name)
            for name in ("graphptc", "openai", "tavily-python", "toolregistry", "codecell")
        },
    }


def _browsecomp_implementation_sha256() -> str:
    digest = hashlib.sha256()
    package_dir = Path(__file__).resolve().parent
    for name in (
        "browsecomp.py",
        "browsecomp_benchmark.py",
        "config.py",
        "model.py",
        "ptc.py",
        "search.py",
    ):
        digest.update(name.encode("utf-8"))
        digest.update((package_dir / name).read_bytes())
    return digest.hexdigest()


def _package_version(name: str) -> str:
    try:
        return version(name)
    except PackageNotFoundError:
        return "unknown"
