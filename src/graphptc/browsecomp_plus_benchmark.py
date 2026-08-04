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
    _summarize_generation,
    _write_records,
)
from .browsecomp_plus import (
    BROWSECOMP_PLUS_CORPUS_REVISION,
    BrowseCompPlusEvaluationResult,
    BrowseCompPlusExample,
    BrowseCompPlusGrade,
    OpenAICompatibleBrowseCompPlusJudge,
    build_browsecomp_plus_grader_prompt,
    evaluate_browsecomp_plus_predictions,
    load_browsecomp_plus,
    load_qrels,
    summarize_browsecomp_plus_grades,
)
from .config import ExperimentConfig, GraderConfig
from .codeact_agent import CodeActPTCAgent
from .direct_tool_agent import DirectToolAgent
from .local_search import OfficialCorpusSearchTools
from .model import OpenAIChatModel
from .ptc import extract_result_tag
from .experiments.phase_planning import PHASE_PLANNING_SUFFIX
from .experiments.ptc_fewshot import PTC_FEW_SHOT_MESSAGES


BROWSECOMP_PLUS_RUNTIME_TOOL_MANIFEST: tuple[dict[str, Any], ...] = (
    {
        "name": "search",
        "description": (
            "Search the frozen local BrowseComp-Plus corpus with BM25. Returns a JSON-compatible "
            "list of zero to five best-first objects with exactly docid (string), score (number), "
            "and snippet (string); there is no title or url field. Scores are comparable only "
            "within one search call. Each snippet is the first 512 tokenizer tokens of the "
            "document, not a query-centered passage, so a missing term does not prove the full "
            "document is irrelevant. An empty query raises ValueError."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "minLength": 1,
                    "description": "The BM25 query string.",
                }
            },
            "required": ["query"],
            "additionalProperties": False,
        },
        "allowed_callers": ["programmatic_tool_call"],
    },
    {
        "name": "fetch",
        "description": (
            "Fetch one complete document from the frozen local BrowseComp-Plus corpus. Accepts a "
            "docid previously returned by search and returns a JSON-compatible object with exactly "
            "docid (string) and content (string). An empty docid raises ValueError and an unknown "
            "docid raises KeyError."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "docid": {
                    "type": "string",
                    "minLength": 1,
                    "description": "A document identifier returned by search.",
                }
            },
            "required": ["docid"],
            "additionalProperties": False,
        },
        "allowed_callers": ["programmatic_tool_call"],
    },
)

BROWSECOMP_PLUS_RUNTIME_TOOL_MANIFEST_JSON = json.dumps(
    BROWSECOMP_PLUS_RUNTIME_TOOL_MANIFEST,
    ensure_ascii=False,
    indent=2,
)


_ORIGINAL_PTC_SEMANTIC_GUIDANCE = (
    "Use a PTC block as a Python research program and use Python as the intermediate data-processing "
    "layer, rather than merely wrapping a tool call. It may call the runtime tools multiple times "
    "when subsequent operations follow mechanically from results already available to the program. "
    "Before executing a research phase, include its foreseeable runtime calls and mechanical "
    "downstream processing in the same program. For comparisons, filtering, set operations, "
    "counting, arithmetic, or joining records, keep the results in Python, perform the operation "
    "there, and print compact derived evidence instead of dumping raw results and using another "
    "model turn or PTC block for that processing. Only printed stdout is returned to the "
    "conversation. Return to the model when the evidence requires a new semantic retrieval direction "
    "or is sufficient to answer. There is no required number of calls and no fixed program template. "
    "Call search and fetch with keyword arguments. Do not access files, the shell, "
    "environment variables, or the network. Programs are time-limited and stdout may be "
    "truncated. When the evidence is sufficient, answer directly inside <result> and </result> "
    "tags; separate multiple answers with commas."
)


BROWSECOMP_PLUS_ORIGINAL_PTC_SYSTEM_PROMPT = (
    "You are a research agent. Your only directly callable tool is "
    "programmatic_tool_call, which executes Python in a task-scoped session. Variables and "
    "imports persist between PTC blocks for this task and are reset before the next task. The "
    "following runtime functions are available only inside Python:\n\n"
    "<runtime_tool_definitions>\n"
    + BROWSECOMP_PLUS_RUNTIME_TOOL_MANIFEST_JSON
    + "\n</runtime_tool_definitions>\n\n"
    + _ORIGINAL_PTC_SEMANTIC_GUIDANCE
)

BROWSECOMP_PLUS_PHASE_PLANNING_SYSTEM_PROMPT = (
    BROWSECOMP_PLUS_ORIGINAL_PTC_SYSTEM_PROMPT + PHASE_PLANNING_SUFFIX
)


BROWSECOMP_PLUS_ORIGINAL_PTC_USER_PROMPT_TEMPLATE = """Answer the following question using the
research environment when evidence is needed.

<question>{question}</question>

Return the final concise answer inside <result> and </result> tags; separate multiple answers with
commas."""


BROWSECOMP_PLUS_DIRECT_SYSTEM_PROMPT = """You are a research agent with direct access to search
and fetch. Search the frozen corpus for relevant candidates and fetch a document when its full text
is needed. Use the returned evidence to refine the research direction and avoid repeating searches
that have already been exhausted. When the evidence is sufficient, answer directly and concisely
inside <result> and </result> tags; separate multiple answers with commas."""


BROWSECOMP_PLUS_DIRECT_USER_PROMPT_TEMPLATE = """Answer the following question using search and
fetch when evidence is needed.

<question>{question}</question>

Put the final concise answer inside <result> and </result> tags; separate multiple answers with
commas."""


_PROMPT_VARIANTS = {
    "original-ptc-v1": (
        BROWSECOMP_PLUS_ORIGINAL_PTC_SYSTEM_PROMPT,
        BROWSECOMP_PLUS_ORIGINAL_PTC_USER_PROMPT_TEMPLATE,
    ),
    "phase-planning-v1": (
        BROWSECOMP_PLUS_PHASE_PLANNING_SYSTEM_PROMPT,
        BROWSECOMP_PLUS_ORIGINAL_PTC_USER_PROMPT_TEMPLATE,
    ),
    "fewshot-ptc-v1": (
        BROWSECOMP_PLUS_ORIGINAL_PTC_SYSTEM_PROMPT,
        BROWSECOMP_PLUS_ORIGINAL_PTC_USER_PROMPT_TEMPLATE,
    ),
    "direct-tools-v1": (
        BROWSECOMP_PLUS_DIRECT_SYSTEM_PROMPT,
        BROWSECOMP_PLUS_DIRECT_USER_PROMPT_TEMPLATE,
    ),
}


BROWSECOMP_PLUS_ORIGINAL_PTC_TOOL_SPEC: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "programmatic_tool_call",
        "description": (
            "Execute one Python research program in the persistent task session. The program can "
            "call the nested search and fetch runtime functions multiple times, use Python control "
            "flow and data structures, and return only printed stdout as its observation."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "code": {
                    "type": "string",
                    "minLength": 1,
                    "description": (
                        "A complete Python program for one coherent research phase. Globals: "
                        "search(*, query: str) returns zero to five best-first objects containing "
                        "exactly docid, score, and a first-512-token document-prefix snippet; "
                        "fetch(*, docid: str) accepts a docid returned by search and returns exactly "
                        "docid and full content. Empty queries raise ValueError; unknown docids "
                        "raise KeyError; runtime failures return PTC_ERROR. Variables and imports "
                        "persist between blocks in the same task and reset before the next task. "
                        "Only stdout is returned and it may be truncated. Use keyword arguments. "
                        "If the next operation can be chosen before seeing the result, include it "
                        "in this program rather than returning raw results for another block. "
                        "Files, shell, environment variables, and direct network access are "
                        "unavailable. Programs are time-limited."
                    ),
                }
            },
            "required": ["code"],
            "additionalProperties": False,
        },
        "strict": True,
    },
}


BROWSECOMP_PLUS_DIRECT_TOOL_SPECS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": tool["name"],
            "description": tool["description"],
            "parameters": tool["input_schema"],
            "strict": True,
        },
    }
    for tool in BROWSECOMP_PLUS_RUNTIME_TOOL_MANIFEST
]


def run_browsecomp_plus_benchmark(
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
    system_prompt, user_prompt_template = _prompt_pair(config)
    retriever_metadata = _retriever_metadata(config)

    examples = load_browsecomp_plus(
        config.benchmark.dataset_path,
        expected_examples=config.browsecomp_plus.expected_examples,
    )
    selected = _select_examples(examples, limit=limit, example_ids=example_ids)
    output_path = config.benchmark.responses_path
    output_path.parent.mkdir(parents=True, exist_ok=True)
    run_signature = _run_signature(config, retriever_metadata)

    existing_records = (
        _load_records(output_path, recover_truncated_tail=True) if resume else []
    )
    incompatible = [
        record["example_id"]
        for record in existing_records
        if record.get("run_signature") != run_signature
    ]
    if incompatible:
        raise ValueError(
            "BrowseComp-Plus responses use another run configuration "
            f"(examples: {incompatible[:5]})."
        )
    completed_ids = {record["example_id"] for record in existing_records}
    pending = [example for example in selected if example.example_id not in completed_ids]
    model_api_key = (
        config.require_api_key(config.model.api_key_env) if pending else ""
    )

    if not resume:
        output_path.write_text("", encoding="utf-8")

    def run_one(example: BrowseCompPlusExample) -> dict[str, Any]:
        checkpoint_path = _checkpoint_path(output_path, example.example_id)
        try:
            model = OpenAIChatModel(config.model, model_api_key)
            tools = OfficialCorpusSearchTools(
                config.browsecomp_plus.retriever_url,
                max_tool_calls=config.browsecomp_plus.max_tool_calls,
                timeout_seconds=config.browsecomp_plus.retriever_timeout_seconds,
            )
            agent_kwargs = {
                "model": model,
                "search_tools": tools,
                "runtime": config.runtime,
                "system_prompt": system_prompt,
                "user_prompt_template": user_prompt_template,
                "runtime_functions": (tools.search, tools.fetch),
                "checkpoint_callback": lambda snapshot: _write_checkpoint(
                    checkpoint_path,
                    {
                        "run_signature": run_signature,
                        "example_id": example.example_id,
                        "updated_at": datetime.now(UTC).isoformat(),
                        **snapshot,
                    },
                ),
            }
            if config.browsecomp_plus.prompt_variant == "direct-tools-v1":
                agent = DirectToolAgent(
                    model=model,
                    search_tools=tools,
                    runtime=config.runtime,
                    system_prompt=system_prompt,
                    user_prompt_template=user_prompt_template,
                    functions={"search": tools.search, "fetch": tools.fetch},
                    tool_specs=BROWSECOMP_PLUS_DIRECT_TOOL_SPECS,
                )
            else:
                agent = CodeActPTCAgent(
                    **agent_kwargs,
                    ptc_tool_spec=BROWSECOMP_PLUS_ORIGINAL_PTC_TOOL_SPEC,
                    persistent=True,
                    structured_observation=False,
                    demonstration_messages=_demonstration_messages(config),
                )
            result = agent.run(example.question)
            prediction = (
                extract_result_tag(result.answer) if result.status == "success" else None
            )
            status = "success" if prediction is not None else "failed"
            error = result.error
            if result.status == "success" and prediction is None:
                error = "Final answer did not contain a non-empty <result> tag"
            candidate_docids = sorted(
                {
                    str(docid)
                    for call in result.search_calls
                    if call.get("operation") == "search"
                    for docid in call.get("docids", ())
                }
            )
            fetched_docids = sorted(
                {
                    str(call.get("docid"))
                    for call in result.search_calls
                    if call.get("operation") == "fetch" and call.get("docid") is not None
                }
            )
            record = {
                "schema_version": 1,
                "benchmark": "browsecomp_plus",
                "run_signature": run_signature,
                "example_id": example.example_id,
                "prediction": prediction or "",
                "status": status,
                "error": error,
                "candidate_docids": candidate_docids,
                "fetched_docids": fetched_docids,
                "retriever": retriever_metadata,
                "model": config.model.model,
                "created_at": datetime.now(UTC).isoformat(),
                "agent": result.to_dict(),
            }
            checkpoint_path.unlink(missing_ok=True)
            return record
        except Exception as exc:
            return {
                "schema_version": 1,
                "benchmark": "browsecomp_plus",
                "run_signature": run_signature,
                "example_id": example.example_id,
                "prediction": "",
                "status": "failed",
                "error": f"{type(exc).__name__}: {exc}",
                "candidate_docids": [],
                "fetched_docids": [],
                "retriever": retriever_metadata,
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
                for index, future in enumerate(as_completed(futures), 1):
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


def evaluate_browsecomp_plus_benchmark(
    config: ExperimentConfig,
) -> BrowseCompPlusEvaluationResult:
    examples = load_browsecomp_plus(
        config.benchmark.dataset_path,
        expected_examples=config.browsecomp_plus.expected_examples,
    )
    records = _load_records(config.benchmark.responses_path)
    retriever_metadata = _response_retriever_metadata(records)
    expected_signature = _run_signature(config, retriever_metadata)
    signatures = {record.get("run_signature") for record in records}
    if signatures and signatures != {expected_signature}:
        raise ValueError(
            "BrowseComp-Plus responses do not match the current configuration."
        )
    _validate_complete_responses(examples, records)
    prediction_index = {
        record["example_id"]: str(record.get("prediction", "")) for record in records
    }
    cache_keys = {
        example.example_id: _grade_cache_key(
            example, prediction_index.get(example.example_id, ""), config.grader
        )
        for example in examples
    }
    reusable = {"valid", "empty_model_response"}
    cached: dict[str, BrowseCompPlusGrade] = {}
    retained: list[dict[str, Any]] = []
    for record in _load_records(config.benchmark.grades_path):
        example_id = record["example_id"]
        if (
            example_id in cache_keys
            and record.get("cache_key") == cache_keys[example_id]
            and record.get("status") in reusable
        ):
            cached[example_id] = _grade_from_record(record)
            retained.append(record)

    pending = [example for example in examples if example.example_id not in cached]
    config.benchmark.grades_path.parent.mkdir(parents=True, exist_ok=True)
    _write_records(config.benchmark.grades_path, retained)
    new: dict[str, BrowseCompPlusGrade] = {}
    if pending:
        judge = _create_judge(config)

        def persist(grade: BrowseCompPlusGrade) -> None:
            record = {
                "cache_key": cache_keys[grade.example_id],
                "grader_model": config.grader.model,
                **grade.to_dict(),
            }
            with config.benchmark.grades_path.open(
                "a", encoding="utf-8", newline="\n"
            ) as handle:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            new[grade.example_id] = grade

        evaluate_browsecomp_plus_predictions(
            pending,
            prediction_index,
            judge,
            max_workers=config.grader.workers,
            on_grade=persist,
        )

    grade_index = {**cached, **new}
    grades = tuple(grade_index[example.example_id] for example in examples)
    candidate_retrieval_recall = _evidence_recall(
        examples,
        records,
        config.browsecomp_plus.qrels_evidence_path,
        record_field="candidate_docids",
    )
    fetched_evidence_recall = _evidence_recall(
        examples,
        records,
        config.browsecomp_plus.qrels_evidence_path,
        record_field="fetched_docids",
    )
    summary = summarize_browsecomp_plus_grades(
        grades,
        candidate_retrieval_recall=candidate_retrieval_recall,
        fetched_evidence_recall=fetched_evidence_recall,
    )
    result = BrowseCompPlusEvaluationResult(grades=grades, summary=summary)
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
        "benchmark": "browsecomp_plus",
        "run_signature": expected_signature,
        "run_configuration": _run_signature_payload(config, retriever_metadata),
        "model": config.model.model,
        "grader_provider": config.grader.provider,
        "grader_model": config.grader.model,
        "created_at": datetime.now(UTC).isoformat(),
        "generation": _summarize_generation(records),
        "summary": summary.to_dict(),
    }
    config.benchmark.report_path.parent.mkdir(parents=True, exist_ok=True)
    config.benchmark.report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return result


def _select_examples(
    examples: list[BrowseCompPlusExample],
    *,
    limit: int | None,
    example_ids: Iterable[str] | None,
) -> list[BrowseCompPlusExample]:
    requested = list(dict.fromkeys(example_ids or ()))
    if requested:
        index = {example.example_id: example for example in examples}
        unknown = [example_id for example_id in requested if example_id not in index]
        if unknown:
            raise ValueError(f"Unknown BrowseComp-Plus IDs: {unknown[:5]}")
        selected = [index[example_id] for example_id in requested]
    else:
        selected = examples
    return selected[:limit] if limit is not None else selected


def _validate_complete_responses(
    examples: list[BrowseCompPlusExample], records: list[dict[str, Any]]
) -> None:
    expected_ids = {example.example_id for example in examples}
    response_ids = {record["example_id"] for record in records}
    missing = sorted(expected_ids - response_ids)
    unknown = sorted(response_ids - expected_ids)
    invalid_status = sorted(
        record["example_id"]
        for record in records
        if record.get("status") not in {"success", "failed"}
    )
    if missing or unknown or invalid_status:
        raise ValueError(
            "BrowseComp-Plus evaluation requires one terminal response for every "
            f"example (missing={missing[:5]}, unknown={unknown[:5]}, "
            f"invalid_status={invalid_status[:5]})."
        )


def _evidence_recall(
    examples: list[BrowseCompPlusExample],
    records: list[dict[str, Any]],
    qrels_path: Path,
    *,
    record_field: str,
) -> float:
    qrels = load_qrels(qrels_path)
    retrieved = {
        record["example_id"]: {str(value) for value in record.get(record_field, [])}
        for record in records
    }
    recalls = []
    for example in examples:
        relevant = qrels.get(example.example_id, set())
        if relevant:
            recalls.append(
                len(retrieved.get(example.example_id, set()) & relevant) / len(relevant)
            )
    return sum(recalls) / len(recalls) if recalls else 0.0


def _grade_cache_key(
    example: BrowseCompPlusExample, prediction: str, grader: GraderConfig
) -> str:
    payload = {
        "grader": asdict(grader),
        "judge_prompt": build_browsecomp_plus_grader_prompt(example, prediction),
        "openai": _package_version("openai"),
    }
    serialized = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode()
    return hashlib.sha256(serialized).hexdigest()


def _create_judge(config: ExperimentConfig) -> OpenAICompatibleBrowseCompPlusJudge:
    if config.grader.provider != "openai_compatible":
        raise ValueError(
            "BrowseComp-Plus development grading requires an OpenAI-compatible judge"
        )
    return OpenAICompatibleBrowseCompPlusJudge(
        api_key=config.require_api_key(config.grader.api_key_env),
        model=config.grader.model,
        base_url=config.grader.base_url,
        max_retries=config.grader.max_retries,
        max_completion_tokens=config.grader.max_completion_tokens,
        thinking=config.grader.thinking,
        timeout_seconds=config.grader.timeout_seconds,
    )


def _grade_from_record(record: dict[str, Any]) -> BrowseCompPlusGrade:
    try:
        return BrowseCompPlusGrade(
            example_id=record["example_id"],
            status=record["status"],
            correct=record.get("correct"),
            confidence=record.get("confidence"),
            accuracy=float(record.get("accuracy", 0.0)),
            raw_judge_response=record.get("raw_judge_response", ""),
            error=record.get("error"),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"Malformed BrowseComp-Plus grade: {exc}") from exc


def _run_signature(
    config: ExperimentConfig, retriever_metadata: dict[str, Any]
) -> str:
    serialized = json.dumps(
        _run_signature_payload(config, retriever_metadata),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return hashlib.sha256(serialized).hexdigest()


def _run_signature_payload(
    config: ExperimentConfig, retriever_metadata: dict[str, Any]
) -> dict[str, Any]:
    local = config.browsecomp_plus
    system_prompt, user_prompt_template = _prompt_pair(config)
    return {
        "benchmark": "browsecomp_plus",
        "corpus_revision": BROWSECOMP_PLUS_CORPUS_REVISION,
        "dataset_sha256": _file_sha256(config.benchmark.dataset_path),
        "qrels_gold_sha256": _file_sha256(local.qrels_gold_path),
        "qrels_evidence_sha256": _file_sha256(local.qrels_evidence_path),
        "generation_workers": config.benchmark.workers,
        "retriever": {
            **retriever_metadata,
            "url": local.retriever_url,
            "timeout_seconds": local.retriever_timeout_seconds,
            "max_tool_calls": local.max_tool_calls,
        },
        "model": asdict(config.model),
        "runtime": asdict(config.runtime),
        "prompt_variant": local.prompt_variant,
        "action_transport": "function_call",
        "system_prompt": system_prompt,
        "user_prompt_template": user_prompt_template,
        "demonstration_messages": _demonstration_messages(config),
        "runtime_tool_manifest": _runtime_tool_manifest(config),
        "ptc_tool_spec": _ptc_tool_spec(config),
        "direct_tool_specs": (
            BROWSECOMP_PLUS_DIRECT_TOOL_SPECS
            if local.prompt_variant == "direct-tools-v1"
            else []
        ),
        "implementation_sha256": _implementation_sha256(),
        "python": platform.python_version(),
        "packages": {
            name: _package_version(name)
            for name in ("graphptc", "openai", "toolregistry", "codecell")
        },
    }


def _prompt_pair(config: ExperimentConfig) -> tuple[str, str]:
    variant = config.browsecomp_plus.prompt_variant
    try:
        return _PROMPT_VARIANTS[variant]
    except KeyError as exc:
        supported = ", ".join(sorted(_PROMPT_VARIANTS))
        raise ValueError(
            f"Unknown BrowseComp-Plus prompt variant {variant!r}; supported: {supported}"
        ) from exc


def _ptc_tool_spec(config: ExperimentConfig) -> dict[str, Any] | None:
    if config.browsecomp_plus.prompt_variant == "direct-tools-v1":
        return None
    return BROWSECOMP_PLUS_ORIGINAL_PTC_TOOL_SPEC


def _runtime_tool_manifest(
    config: ExperimentConfig,
) -> tuple[dict[str, Any], ...]:
    if config.browsecomp_plus.prompt_variant == "direct-tools-v1":
        return ()
    return BROWSECOMP_PLUS_RUNTIME_TOOL_MANIFEST


def _demonstration_messages(
    config: ExperimentConfig,
) -> tuple[dict[str, Any], ...]:
    variant = config.browsecomp_plus.prompt_variant
    if variant == "fewshot-ptc-v1":
        return PTC_FEW_SHOT_MESSAGES
    return ()


def _retriever_metadata(config: ExperimentConfig) -> dict[str, Any]:
    local = config.browsecomp_plus
    metadata = OfficialCorpusSearchTools(
        local.retriever_url,
        max_tool_calls=local.max_tool_calls,
        timeout_seconds=local.retriever_timeout_seconds,
    ).metadata()
    expected = {
        "backend": "browsecomp_plus_official_bm25",
        "top_k": local.top_k,
        "snippet_max_tokens": local.snippet_max_tokens,
    }
    mismatches = {
        key: (metadata.get(key), value)
        for key, value in expected.items()
        if metadata.get(key) != value
    }
    if mismatches:
        raise ValueError(f"BrowseComp-Plus retriever metadata mismatch: {mismatches}")
    return metadata


def _response_retriever_metadata(records: list[dict[str, Any]]) -> dict[str, Any]:
    values = [record.get("retriever") for record in records]
    if not values or not all(isinstance(value, dict) for value in values):
        raise ValueError("BrowseComp-Plus responses have no retriever metadata.")
    serialized = {
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        for value in values
    }
    if len(serialized) != 1:
        raise ValueError("BrowseComp-Plus responses contain mixed retriever metadata.")
    return dict(values[0])


def _implementation_sha256() -> str:
    digest = hashlib.sha256()
    package_dir = Path(__file__).resolve().parent
    for name in (
        "browsecomp_plus.py",
        "browsecomp_plus_benchmark.py",
        "config.py",
        "local_search.py",
        "model.py",
        "ptc.py",
        "codeact_agent.py",
        "persistent_runtime.py",
        "persistent_worker.py",
        "direct_tool_agent.py",
    ):
        digest.update(name.encode())
        digest.update((package_dir / name).read_bytes())
    return digest.hexdigest()


def _package_version(name: str) -> str:
    try:
        return version(name)
    except PackageNotFoundError:
        return "unknown"


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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
