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
    _write_records,
)
from .browsecomp_plus import BROWSECOMP_PLUS_CORPUS_REVISION, BrowseCompPlusExample
from .browsecomp_plus import load_browsecomp_plus
from .config import ExperimentConfig
from .graph_agent import GraphPTCAgent
from .local_search import SQLiteCorpusSearchTools, index_document_count
from .model import OpenAIChatModel
from .observability import JsonlEventSink
from .ptc import PTC_TOOL_SPEC, extract_result_tag


GRAPHPTC_BROWSECOMP_PLUS_SYSTEM_PROMPT = """You are a research agent with access to programmatic_tool_call. Its Python
environment provides these global functions:

- search(*, query: str) -> list[{"docid": str, "score": float, "snippet": str}]
- fetch(*, docid: str) -> {"docid": str, "content": str}

Search returns structured candidates and Fetch returns the complete indexed document. A program can
call these functions repeatedly or conditionally and process their results before printing; only
stdout is returned to the conversation. Treat a block as one coherent research step: when subsequent
operations can be chosen mechanically from current results, use Python to loop over task-specific
queries, deduplicate candidates by docid, filter or rank snippets, conditionally Fetch useful documents,
and aggregate compact evidence with source docids. Return to the model when the evidence requires a new
semantic judgment or search direction. This is guidance, not a required number of calls or syntax forms.

When this pattern fits the task, adapt a helper of this shape rather than forwarding raw results:

def collect_evidence(queries, keywords, candidate_limit=10):
    import json
    seen = {}
    for query in queries:
        for hit in search(query=query):
            seen.setdefault(hit["docid"], hit)
    evidence = []
    ranked = sorted(seen.values(), key=lambda item: item["score"], reverse=True)
    for hit in ranked[:candidate_limit]:
        if any(word in hit["snippet"].lower() for word in keywords):
            document = fetch(docid=hit["docid"])
            lines = [line for line in document["content"].splitlines()
                     if any(word in line.lower() for word in keywords)]
            evidence.append({"docid": hit["docid"], "lines": lines[:5]})
    print(json.dumps(evidence, ensure_ascii=False))

Choose queries, keywords, selection logic, and extraction logic from the task. The helper is only an
illustration; use simpler or different code when appropriate and never print complete result sets.

The program may import safe computation modules such as json, re, collections, itertools, statistics, datetime, and csv.
Always call runtime tools with keyword arguments. You must not access files, the shell, environment
variables, or the network. Programs are time-limited and stdout may be truncated. You decide whether
tools are needed and how many PTC blocks to generate. Each block must define all variables and imports
it uses because local Python state does not persist between blocks."""

GRAPHPTC_BROWSECOMP_PLUS_USER_PROMPT_TEMPLATE = """I want you to answer the following question.

<question>{question}</question>

First plan out your response. This part can be as long as needed. You may need to run many searches,
this is totally fine.
Then provide a short and concise answer in <result> tags. For questions expecting multiple answers,
separate them with commas."""


def run_graphptc_browsecomp_plus_benchmark(
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
    index_document_count(config.browsecomp_plus.index_path)

    examples = load_browsecomp_plus(config.benchmark.dataset_path)
    selected = _select_examples(examples, limit=limit, example_ids=example_ids)
    output_path = config.benchmark.responses_path
    events_path = output_path.with_name("events.jsonl")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    run_signature = _run_signature(config)

    selected_ids = {example.example_id for example in selected}
    existing_records = _load_records(output_path) if resume else []
    incompatible = [
        record["example_id"]
        for record in existing_records
        if record.get("run_signature") != run_signature
    ]
    if incompatible:
        raise ValueError(
            "GraphPTC BrowseComp-Plus responses use another run configuration "
            f"(examples: {incompatible[:5]})."
        )
    successful_ids = {
        record["example_id"]
        for record in existing_records
        if _record_succeeded(record)
    }
    pending = [
        example for example in selected if example.example_id not in successful_ids
    ]
    model_api_key = (
        config.require_api_key(config.model.api_key_env) if pending else ""
    )

    if not resume:
        output_path.write_text("", encoding="utf-8")
        events_path.write_text("", encoding="utf-8")
    else:
        retained = [
            record
            for record in existing_records
            if record["example_id"] not in selected_ids or _record_succeeded(record)
        ]
        if len(retained) != len(existing_records):
            _write_records(output_path, retained)

    event_sink = JsonlEventSink(events_path)

    def run_one(example: BrowseCompPlusExample) -> dict[str, Any]:
        try:
            model = OpenAIChatModel(config.model, model_api_key)
            tools = SQLiteCorpusSearchTools(
                config.browsecomp_plus.index_path,
                top_k=config.browsecomp_plus.top_k,
                snippet_max_chars=config.browsecomp_plus.snippet_max_chars,
                max_tool_calls=config.browsecomp_plus.max_tool_calls,
            )
            graph_result = GraphPTCAgent(
                model=model,
                search_tools=tools,  # type: ignore[arg-type]
                runtime=config.runtime,
                system_prompt=GRAPHPTC_BROWSECOMP_PLUS_SYSTEM_PROMPT,
                user_prompt_template=GRAPHPTC_BROWSECOMP_PLUS_USER_PROMPT_TEMPLATE,
                runtime_functions=(tools.search_local, tools.search_local_batch),
                event_sink=event_sink,
            ).run(example.question)
            result = graph_result.agent
            prediction = (
                extract_result_tag(result.answer) if result.status == "success" else None
            )
            status = "success" if prediction is not None else "failed"
            error = result.error
            if result.status == "success" and prediction is None:
                error = "Final answer did not contain a non-empty <result> tag"
            retrieved_docids = sorted(
                {
                    str(docid)
                    for call in result.search_calls
                    for docid in call.get("docids", ())
                }
            )
            return {
                "schema_version": 1,
                "benchmark": "browsecomp_plus_graphptc_stage1",
                "run_signature": run_signature,
                "example_id": example.example_id,
                "prediction": prediction or "",
                "status": status,
                "error": error,
                "retrieved_docids": retrieved_docids,
                "model": config.model.model,
                "created_at": datetime.now(UTC).isoformat(),
                "agent": result.to_dict(),
                "graphptc": {
                    "stage": 1,
                    "episode_id": graph_result.episode_id,
                    "event_count": len(graph_result.events),
                    "events_path": str(events_path),
                },
            }
        except Exception as exc:
            return {
                "schema_version": 1,
                "benchmark": "browsecomp_plus_graphptc_stage1",
                "run_signature": run_signature,
                "example_id": example.example_id,
                "prediction": "",
                "status": "failed",
                "error": f"{type(exc).__name__}: {exc}",
                "retrieved_docids": [],
                "model": config.model.model,
                "created_at": datetime.now(UTC).isoformat(),
                "agent": None,
                "graphptc": None,
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


def _run_signature(config: ExperimentConfig) -> str:
    serialized = json.dumps(
        _run_signature_payload(config),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return hashlib.sha256(serialized).hexdigest()


def _run_signature_payload(config: ExperimentConfig) -> dict[str, Any]:
    local = config.browsecomp_plus
    return {
        "benchmark": "browsecomp_plus_graphptc_stage1",
        "graphptc_stage": 1,
        "corpus_revision": BROWSECOMP_PLUS_CORPUS_REVISION,
        "document_count": index_document_count(local.index_path),
        "local_search": {
            "top_k": local.top_k,
            "snippet_max_chars": local.snippet_max_chars,
            "max_tool_calls": local.max_tool_calls,
        },
        "model": asdict(config.model),
        "runtime": asdict(config.runtime),
        "system_prompt": GRAPHPTC_BROWSECOMP_PLUS_SYSTEM_PROMPT,
        "user_prompt_template": GRAPHPTC_BROWSECOMP_PLUS_USER_PROMPT_TEMPLATE,
        "ptc_tool_spec": PTC_TOOL_SPEC,
        "implementation_sha256": _implementation_sha256(),
        "python": platform.python_version(),
        "packages": {
            name: _package_version(name)
            for name in ("graphptc", "openai", "toolregistry", "codecell")
        },
    }


def _implementation_sha256() -> str:
    digest = hashlib.sha256()
    package_dir = Path(__file__).resolve().parent
    for name in (
        "graph_agent.py",
        "graph_browsecomp_plus_benchmark.py",
        "observability.py",
        "config.py",
        "local_search.py",
        "model.py",
        "ptc.py",
    ):
        digest.update(name.encode())
        digest.update((package_dir / name).read_bytes())
    return digest.hexdigest()


def _package_version(name: str) -> str:
    try:
        return version(name)
    except PackageNotFoundError:
        return "unknown"
