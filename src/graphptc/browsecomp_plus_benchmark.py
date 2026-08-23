from __future__ import annotations

import copy
import gzip
import hashlib
import json
import platform
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict
from datetime import UTC, datetime
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any, Callable, Iterable

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
from .graph_agent import (
    GraphAgentHooks,
    extend_ptc_spec_with_graph_control,
)
from .goal_adaptation import GoalGraphAdaptation
from .tool_effects import ToolEffectContract
from .model import OpenAIChatModel
from .observability import ExecutionObserver
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
    observer_factory: Callable[[str, str], ExecutionObserver] | None = None,
    post_episode_callback: Callable[[str, str, dict[str, Any]], None] | None = None,
    checkpoint_archive_dir: Path | None = None,
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
            base_tools = OfficialCorpusSearchTools(
                config.browsecomp_plus.retriever_url,
                max_tool_calls=config.browsecomp_plus.max_tool_calls,
                timeout_seconds=config.browsecomp_plus.retriever_timeout_seconds,
            )
            tools = base_tools
            adaptation_mode = config.runtime.graph_adaptation_mode
            _validate_control_modes(config)
            if adaptation_mode == "generic":
                graph_adaptation = GoalGraphAdaptation(
                    {"search": tools.search, "fetch": tools.fetch},
                    {
                        "search": ToolEffectContract(
                            name="search",
                            effect="read",
                            deterministic=True,
                            cacheable=True,
                            artifact_kind="search_result",
                        ),
                        "fetch": ToolEffectContract(
                            name="fetch",
                            effect="read",
                            deterministic=True,
                            cacheable=True,
                            artifact_kind="fetched_resource",
                        ),
                    },
                    task=example.question,
                    expose_graph_api=False,
                )
            else:
                graph_adaptation = None
            graph_hooks = (
                GraphAgentHooks.from_controller(graph_adaptation)
                if graph_adaptation is not None
                else None
            )
            agent_kwargs = {
                "model": model,
                "search_tools": tools,
                "runtime": config.runtime,
                "system_prompt": system_prompt,
                "user_prompt_template": user_prompt_template,
                "runtime_functions": (
                    graph_hooks.runtime_functions
                    if graph_hooks is not None
                    else (tools.search, tools.fetch)
                ),
                "post_block_message_factory": None,
                "post_block_message_on_error": False,
                "block_observation_factory": (
                    None if graph_hooks is None else graph_hooks.block_observation_factory
                ),
                "ptc_call_metadata_callback": (
                    None
                    if graph_hooks is None
                    else graph_hooks.ptc_call_metadata_callback
                ),
                "adaptation_initial_observation": (
                    None
                    if graph_hooks is None
                    else graph_hooks.adaptation_initial_observation
                ),
                "checkpoint_callback": lambda snapshot: _write_checkpoint_bundle(
                    checkpoint_path,
                    {
                        "run_signature": run_signature,
                        "example_id": example.example_id,
                        "updated_at": datetime.now(UTC).isoformat(),
                        **snapshot,
                    },
                    archive_dir=checkpoint_archive_dir,
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
                    ptc_tool_spec=_ptc_tool_spec(config),
                    persistent=True,
                    demonstration_messages=_demonstration_messages(config),
                    observer=(
                        observer_factory(example.example_id, run_signature)
                        if observer_factory is not None
                        else None
                    ),
                )
            result = agent.run(example.question)
            prediction = (
                extract_result_tag(result.answer) if result.status == "success" else None
            )
            if graph_adaptation is not None:
                graph_adaptation.finish(answered=prediction is not None)
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
                "graph_adaptation": (
                    None if graph_adaptation is None else graph_adaptation.telemetry()
                ),
            }
            checkpoint_path.unlink(missing_ok=True)
        except Exception as exc:
            record = {
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
        if post_episode_callback is not None:
            try:
                post_episode_callback(
                    example.example_id,
                    run_signature,
                    copy.deepcopy(record),
                )
            except Exception:
                pass
        return record

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
        "graph_adaptation": _summarize_graph_adaptation(records),
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
        system_prompt, user_prompt = _PROMPT_VARIANTS[variant]
    except KeyError as exc:
        supported = ", ".join(sorted(_PROMPT_VARIANTS))
        raise ValueError(
            f"Unknown BrowseComp-Plus prompt variant {variant!r}; supported: {supported}"
        ) from exc
    if config.runtime.graph_adaptation_mode != "generic":
        return system_prompt, user_prompt
    graph_guidance = (
        "GRAPH_ASSESSMENT and GRAPH_DELTA expose a compact, domain-neutral effect frontier. "
        "The runtime records actions, artifacts, state dependencies, failures, and whether recent "
        "actions produced new or equivalent results; it never chooses a tool or its arguments. "
        "Describe the action actually taken and its expected observable change. If REPLAN is offered, "
        "preserve productive paths, avoid exhausted ones, and change the dependency path. If PATCH is "
        "offered, correct and re-execute the failed operation. Answer directly when the available "
        "results satisfy the task."
    )
    return system_prompt + "\n\n" + graph_guidance, user_prompt


def _ptc_tool_spec(config: ExperimentConfig) -> dict[str, Any] | None:
    _validate_control_modes(config)
    if config.browsecomp_plus.prompt_variant == "direct-tools-v1":
        return None
    if config.runtime.graph_adaptation_mode == "generic":
        spec = copy.deepcopy(BROWSECOMP_PLUS_ORIGINAL_PTC_TOOL_SPEC)
        spec["function"]["parameters"]["properties"]["code"]["description"] += (
            " Graph dependency tracking and effect recording are automatic."
        )
        return extend_ptc_spec_with_graph_control(
            spec,
            include_target=False,
            include_input_artifacts=False,
            action_description="The graph-control intent implemented by this PTC block.",
            expected_change_description="The new artifact, state effect, or goal change expected from this block.",
        )
    return BROWSECOMP_PLUS_ORIGINAL_PTC_TOOL_SPEC


def _runtime_tool_manifest(
    config: ExperimentConfig,
) -> tuple[dict[str, Any], ...]:
    _validate_control_modes(config)
    if config.browsecomp_plus.prompt_variant == "direct-tools-v1":
        return ()
    return BROWSECOMP_PLUS_RUNTIME_TOOL_MANIFEST


def _validate_control_modes(config: ExperimentConfig) -> None:
    adaptation_mode = config.runtime.graph_adaptation_mode
    if adaptation_mode not in {"off", "generic"}:
        raise ValueError("runtime.graph_adaptation_mode must be one of off, generic")
    if (
        adaptation_mode == "generic"
        and config.browsecomp_plus.prompt_variant == "direct-tools-v1"
    ):
        raise ValueError("graph adaptation currently requires a PTC prompt variant")


def _demonstration_messages(
    config: ExperimentConfig,
) -> tuple[dict[str, Any], ...]:
    variant = config.browsecomp_plus.prompt_variant
    if variant == "fewshot-ptc-v1":
        if config.runtime.graph_adaptation_mode == "generic":
            messages = copy.deepcopy(PTC_FEW_SHOT_MESSAGES)
            for message in messages:
                for call in message.get("tool_calls", ()):
                    function = call.get("function", {})
                    if function.get("name") != "programmatic_tool_call":
                        continue
                    arguments = json.loads(function["arguments"])
                    arguments.update(
                        {
                            "action": "CONTINUE",
                            "expected_change": "produce new task-relevant artifacts",
                        }
                    )
                    function["arguments"] = json.dumps(arguments)
            return tuple(messages)
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
        "episode_graph.py",
        "execution_projection.py",
        "graph_agent.py",
        "goal_adaptation.py",
        "tool_effects.py",
    ):
        digest.update(name.encode())
        digest.update((package_dir / name).read_bytes())
    return digest.hexdigest()


def _summarize_graph_adaptation(records: list[dict[str, Any]]) -> dict[str, Any] | None:
    values = [
        record["graph_adaptation"]
        for record in records
        if isinstance(record.get("graph_adaptation"), dict)
    ]
    if not values:
        return None
    actions: Counter[str] = Counter()
    interfaces: Counter[str] = Counter()
    requirement_states: Counter[str] = Counter()
    for value in values:
        actions.update(value.get("action_distribution") or {})
        graph = value.get("research_graph") or {}
        interfaces.update(graph.get("interface_calls") or {})
        requirement_states.update(graph.get("requirement_states") or {})
    return {
        "episodes": len(values),
        "observation_calls": sum(int(value.get("observation_calls", 0)) for value in values),
        "action_distribution": dict(actions),
        "task_graph_initialized_episodes": sum(
            bool((value.get("research_graph") or {}).get("task_graph_initialized"))
            for value in values
        ),
        "realized_graph_deltas": sum(
            int(value.get("realized_graph_deltas", 0)) for value in values
        ),
        "missed_graph_deltas": sum(
            int(value.get("missed_graph_deltas", 0)) for value in values
        ),
        "requirement_states": dict(requirement_states),
        "invalid_action_targets": sum(
            int(value.get("invalid_action_targets", 0)) for value in values
        ),
        "aligned_actions": sum(
            int(value.get("aligned_actions", 0)) for value in values
        ),
        "misaligned_actions": sum(
            int(value.get("misaligned_actions", 0)) for value in values
        ),
        "node_count": sum(
            int((value.get("research_graph") or {}).get("node_count", 0))
            for value in values
        ),
        "edge_count": sum(
            int((value.get("research_graph") or {}).get("edge_count", 0))
            for value in values
        ),
        "artifact_count": sum(
            int((value.get("research_graph") or {}).get("artifact_count", 0))
            for value in values
        ),
        "artifact_reuse_hits": sum(
            int((value.get("research_graph") or {}).get("artifact_reuse_hits", 0))
            for value in values
        ),
        "interface_calls": dict(interfaces),
        "tool_reuse_hits": sum(int(value.get("tool_reuse_hits", 0)) for value in values),
        "artifact_loads": sum(int(value.get("artifact_loads", 0)) for value in values),
    }


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


def _write_checkpoint_bundle(
    path: Path,
    payload: dict[str, Any],
    *,
    archive_dir: Path | None,
) -> None:
    _write_checkpoint(path, payload)
    if archive_dir is None:
        return
    example_id = str(payload["example_id"])
    next_turn = int(payload["next_turn"])
    episode_dir = archive_dir / hashlib.sha256(example_id.encode()).hexdigest()[:20]
    episode_dir.mkdir(parents=True, exist_ok=True)
    destination = episode_dir / f"turn-{next_turn:03d}.json.gz"
    temporary = destination.with_suffix(".tmp")
    with gzip.open(temporary, "wt", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, separators=(",", ":"))
        handle.write("\n")
    temporary.replace(destination)
