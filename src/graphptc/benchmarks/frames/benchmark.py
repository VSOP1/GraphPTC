from __future__ import annotations

import copy
import csv
import dataclasses
import json
import re
import string
import threading
import time
from collections.abc import Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from ...agents.direct_tools import DirectToolAgent
from ...agents.original_ptc import PTC_TOOL_SPEC, OriginalPTCAgent
from ...config import ExperimentConfig, ModelConfig
from ...graph.adaptation import GoalGraphAdaptation
from ...graph.hooks import GraphAgentHooks, extend_ptc_spec_with_graph_control
from ...graph.tool_effects import ToolEffectContract
from ...model import OpenAIChatModel

OFFICIAL_USER_PROMPT = """Answer the following factoid question using the Wikipedia research
environment. Think step by step, issue distinct search queries, and do not repeat a query.

[Question]: {question}

Return only the concise final answer, without an explanation."""

SYSTEM_PROMPT = """You are solving the official FRAMES multi-hop Wikipedia benchmark through
programmatic tool calling. The official retrieval setting is BM25 over the 2023-06-01 English
Wikipedia snapshot, with up to five planning rounds, five distinct queries per round, and ten
documents per query.

Your only directly callable model tool is programmatic_tool_call. Its Python source runs in one
persistent namespace for this question. Two functions are Python globals:

- wiki_search(query): retrieve the ten best BM25 results as dictionaries with docid, title, score,
  and snippet. At most 25 calls are available for the question.
- wiki_content(doc): fetch the complete article selected from a wiki_search result.

Plan searches step by step. Use distinct queries that move from the entities and constraints in the
question to bridge facts and then to the final fact; do not repeat a query. Use each PTC block as a
coherent research program. Python variables persist across blocks. Use loops, filtering, joins,
arithmetic, date comparison, and aggregation in Python when downstream operations are mechanical.
Only printed stdout is visible on the next turn, so keep full articles in variables and print compact
evidence needed for the next semantic decision. Use only the supplied Wikipedia functions; do not
access files, environment variables, the shell, or another network source."""

FINALIZE_PROMPT = """Wikipedia tools are now unavailable. Return only the concise answer supported
by the retrieved evidence. Do not emit Python, a plan, an explanation, or result tags."""

DIRECT_SYSTEM_PROMPT = """You are solving the official FRAMES multi-hop benchmark with direct
access to the frozen Wikipedia tools. Plan distinct searches that move from the question entities
through bridge facts to the final fact. Use wiki_search for the ten best BM25 results and
wiki_content for selected full articles. Do not repeat queries. Return only the concise final
answer without an explanation."""

DIRECT_USER_PROMPT = """Answer this factoid question using the Wikipedia tools when needed.
[Question]: {task}
Return only the concise final answer."""

DIRECT_TOOL_SPECS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "wiki_search",
            "description": "Return the ten best BM25 results from the frozen Wikipedia snapshot.",
            "parameters": {
                "type": "object",
                "properties": {"query": {"type": "string", "minLength": 1}},
                "required": ["query"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "wiki_content",
            "description": "Fetch one full Wikipedia article selected from wiki_search results.",
            "parameters": {
                "type": "object",
                "properties": {"doc": {"type": "object"}},
                "required": ["doc"],
                "additionalProperties": False,
            },
        },
    },
]

PTC_SPEC = {
    **PTC_TOOL_SPEC,
    "function": {
        **PTC_TOOL_SPEC["function"],
        "description": (
            "Execute one coherent Python research program in the persistent FRAMES namespace. "
            "wiki_search and wiki_content are available as Python globals."
        ),
    },
}

JUDGE_SYSTEM_PROMPT = "You are a helpful assistant."
JUDGE_PROMPT = """===Task===
I need your help in evaluating an answer provided by an LLM against a ground truth
answer. Your task is to determine if the ground truth answer is present in the LLM's response.
Please analyze the provided data and make a decision.
===Instructions===
1. Carefully compare the "Predicted Answer" with the "Ground Truth Answer".
2. Consider the substance of the answers - look for equivalent information or correct answers. Do
not focus on exact wording unless the exact wording is crucial to the meaning.
3. Your final decision should be based on whether the meaning and the vital facts of the "Ground
Truth Answer" are present in the "Predicted Answer:"
===Input Data===
- Question: {question}
- Predicted Answer: {answer}
- Ground Truth Answer: {reference}
===Output Format===
Provide your final evaluation in the following format:
"Explanation:" (How you made the decision?)
"Decision:" ("TRUE" or "FALSE" )
Please proceed with the evaluation."""

RETRIEVER_TIMEOUT_SECONDS = 90.0


class _EmptySearchTools:
    calls: list[dict[str, Any]] = []


@dataclass(frozen=True)
class FramesExample:
    id: str
    question: str
    answer: str
    reasoning_types: tuple[str, ...]


@dataclass(frozen=True)
class FramesRunSummary:
    selected: int
    processed: int
    successful: int
    failed: int
    wiki_search_calls: int
    wiki_content_calls: int
    input_tokens: int
    output_tokens: int

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


class FramesWikiTools:
    def __init__(self, *, base_url: str, max_search_calls: int) -> None:
        self._base_url = base_url.rstrip("/")
        self._max_search_calls = max_search_calls
        self._search_calls = 0
        self._calls: list[dict[str, Any]] = []
        self._lock = threading.Lock()

    @property
    def calls(self) -> list[dict[str, Any]]:
        with self._lock:
            return copy.deepcopy(self._calls)

    def wiki_search(self, query: str) -> list[dict[str, Any]]:
        with self._lock:
            if self._search_calls >= self._max_search_calls:
                raise RuntimeError(
                    f"FRAMES search budget exhausted ({self._max_search_calls} calls)"
                )
            self._search_calls += 1
            sequence = self._search_calls
        started = time.perf_counter()
        results = _retriever_request(self._base_url, "/search", {"query": str(query)})
        if not isinstance(results, list) or not all(isinstance(item, dict) for item in results):
            raise ValueError("FRAMES retriever search response must be a list of objects")
        output = [dict(item) for item in results]
        with self._lock:
            self._calls.append(
                {
                    "operation": "wiki_search",
                    "sequence": sequence,
                    "query": str(query),
                    "documents": [
                        {
                            "docid": str(item.get("docid", "")),
                            "title": str(item.get("title", "")),
                            "score": item.get("score"),
                        }
                        for item in output
                    ],
                    "duration_ms": (time.perf_counter() - started) * 1_000,
                }
            )
        return output

    def wiki_content(self, doc: Mapping[str, Any]) -> str:
        docid = str(doc["docid"])
        started = time.perf_counter()
        result = _retriever_request(self._base_url, "/fetch", {"docid": docid})
        if not isinstance(result, dict) or "content" not in result:
            raise ValueError("FRAMES retriever fetch response must contain content")
        content = str(result["content"])
        with self._lock:
            self._calls.append(
                {
                    "operation": "wiki_content",
                    "docid": docid,
                    "title": str(result.get("title", doc.get("title", ""))),
                    "content_chars": len(content),
                    "duration_ms": (time.perf_counter() - started) * 1_000,
                }
            )
        return content


def inspect_frames(config: ExperimentConfig) -> dict[str, Any]:
    _validate_config(config)
    examples = _load_examples(config.frames.dataset_path)
    if len(examples) != config.frames.expected_tasks:
        raise ValueError(
            f"expected {config.frames.expected_tasks} FRAMES examples, found {len(examples)}"
        )
    ids = [example.id for example in examples]
    if len(ids) != len(set(ids)):
        raise ValueError("FRAMES task IDs are not unique")
    metadata = _retriever_request(config.frames.retriever_base_url, "/metadata", None)
    if not isinstance(metadata, dict):
        raise ValueError("FRAMES retriever metadata must be an object")
    if metadata.get("corpus_snapshot") != config.frames.corpus_snapshot:
        raise ValueError(
            "FRAMES retriever corpus mismatch: "
            f"expected {config.frames.corpus_snapshot}, found {metadata.get('corpus_snapshot')}"
        )
    if int(metadata.get("document_count", 0)) != 6_672_479:
        raise ValueError("FRAMES retriever must contain all 6,672,479 official TFDS articles")
    return {
        "benchmark": "frames",
        "split": "test",
        "examples": len(examples),
        "dataset_revision": config.frames.dataset_revision,
        "corpus_snapshot": config.frames.corpus_snapshot,
        "retrieval": {
            "algorithm": "BM25",
            "planning_rounds": 5,
            "queries_per_round": 5,
            "documents_per_query": config.frames.search_results,
            "max_search_calls": config.frames.max_search_calls,
            "metadata": metadata,
        },
        "prompt_variant": config.frames.prompt_variant,
        "graph_adaptation_mode": config.runtime.graph_adaptation_mode,
        "gold_wikipedia_links_exposed": False,
    }


def probe_frames_wikipedia(config: ExperimentConfig) -> dict[str, Any]:
    inspection = inspect_frames(config)
    tools = FramesWikiTools(
        base_url=config.frames.retriever_base_url,
        max_search_calls=config.frames.max_search_calls,
    )
    hits = tools.wiki_search("James Buchanan Harriet Lane")
    if len(hits) != config.frames.search_results:
        raise ValueError(
            f"expected {config.frames.search_results} BM25 results, found {len(hits)}"
        )
    content = tools.wiki_content(hits[0])
    if not content.strip():
        raise ValueError("FRAMES wiki_content returned an empty article")
    return {
        "corpus_snapshot": inspection["corpus_snapshot"],
        "search_results": len(hits),
        "first_title": hits[0].get("title"),
        "content_chars": len(content),
    }


def run_frames_benchmark(
    config: ExperimentConfig,
    *,
    task_ids: Sequence[str] = (),
    limit: int | None = None,
    restart: bool = False,
) -> FramesRunSummary:
    inspection = inspect_frames(config)
    examples = _load_examples(config.frames.dataset_path)
    by_id = {example.id: example for example in examples}
    if task_ids:
        unknown = sorted(set(task_ids) - by_id.keys())
        if unknown:
            raise ValueError(f"unknown FRAMES task IDs: {unknown[:5]}")
        selected = [by_id[task_id] for task_id in task_ids]
    else:
        selected = examples
    if limit is not None:
        selected = selected[:limit]

    output_path = config.frames.results_path
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if restart:
        output_path.write_text("", encoding="utf-8")
    existing = _load_records(output_path)
    pending = [example for example in selected if example.id not in existing]
    api_key = config.require_api_key(config.model.api_key_env) if pending else ""
    if pending:
        with output_path.open("a", encoding="utf-8", newline="\n") as handle:
            with ThreadPoolExecutor(
                max_workers=min(config.frames.workers, len(pending))
            ) as executor:
                futures = {
                    executor.submit(_run_one, config, example, api_key): example
                    for example in pending
                }
                for future in as_completed(futures):
                    record = future.result()
                    existing[record["id"]] = record
                    handle.write(json.dumps(record, ensure_ascii=False, default=repr) + "\n")
                    handle.flush()

    records = [existing[example.id] for example in selected if example.id in existing]
    summary = _summarize_run(selected, records)
    report = {
        "schema_version": 1,
        "benchmark": "frames",
        "official_alignment": inspection,
        "model": dataclasses.asdict(config.model),
        "runtime": dataclasses.asdict(config.runtime),
        "summary": summary.to_dict(),
        "scoring": "pending",
    }
    config.frames.report_path.parent.mkdir(parents=True, exist_ok=True)
    config.frames.report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, default=repr), encoding="utf-8"
    )
    return summary


def evaluate_frames_benchmark(config: ExperimentConfig) -> dict[str, Any]:
    _validate_config(config)
    examples = _load_examples(config.frames.dataset_path)
    records = _load_records(config.frames.results_path)
    if set(records) != {example.id for example in examples}:
        raise ValueError(
            f"FRAMES results are incomplete: expected {len(examples)}, found {len(records)}"
        )
    ordered_records = [records[example.id] for example in examples]
    grades = _score_with_mimo(config, examples, ordered_records)
    exact = {
        example.id: float(
            _normalize_answer(records[example.id].get("answer", ""))
            == _normalize_answer(example.answer)
        )
        for example in examples
    }
    scoring = {
        "mimo_judge": _summarize_scores(examples, grades),
        "normalized_exact_match": _summarize_numeric_scores(examples, exact),
        "official_judge_prompt": "FRAMES paper Figure 6",
        "judge_model_note": "MiMo is used in place of the paper's Gemini-Pro-1.5-0514 autorater",
    }
    report = json.loads(config.frames.report_path.read_text(encoding="utf-8"))
    report["scoring"] = scoring
    config.frames.report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, default=repr), encoding="utf-8"
    )
    return report


def compare_frames_benchmarks(
    graph_config: ExperimentConfig,
    baseline_config: ExperimentConfig,
    output_path: Path,
) -> dict[str, Any]:
    _validate_matched_configs(graph_config, baseline_config)
    examples = _load_examples(graph_config.frames.dataset_path)
    expected_ids = {example.id for example in examples}
    graph_report = json.loads(graph_config.frames.report_path.read_text(encoding="utf-8"))
    baseline_report = json.loads(
        baseline_config.frames.report_path.read_text(encoding="utf-8")
    )
    graph_grades = _load_records(graph_config.frames.grades_path)
    baseline_grades = _load_records(baseline_config.frames.grades_path)
    if set(graph_grades) != expected_ids or set(baseline_grades) != expected_ids:
        raise ValueError("FRAMES paired comparison requires complete matched MiMo grades")
    graph_scores = {key: float(value.get("score", 0.0)) for key, value in graph_grades.items()}
    baseline_scores = {
        key: float(value.get("score", 0.0)) for key, value in baseline_grades.items()
    }
    wins = sum(graph_scores[key] > baseline_scores[key] for key in expected_ids)
    losses = sum(graph_scores[key] < baseline_scores[key] for key in expected_ids)
    graph_scoring = graph_report["scoring"]
    baseline_scoring = baseline_report["scoring"]
    report = {
        "schema_version": 1,
        "benchmark": "frames",
        "split": "test",
        "tasks": len(expected_ids),
        "graphptc": graph_scoring,
        "fewshot_ptc": baseline_scoring,
        "difference": {
            "mimo_judge_accuracy": (
                graph_scoring["mimo_judge"]["accuracy"]
                - baseline_scoring["mimo_judge"]["accuracy"]
            ),
            "normalized_exact_match": (
                graph_scoring["normalized_exact_match"]["accuracy"]
                - baseline_scoring["normalized_exact_match"]["accuracy"]
            ),
            "paired_judge_wins": wins,
            "paired_judge_losses": losses,
            "paired_judge_ties": len(expected_ids) - wins - losses,
        },
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def _run_one(config: ExperimentConfig, example: FramesExample, api_key: str) -> dict[str, Any]:
    started = time.time()
    tools: FramesWikiTools | None = None
    controller: GoalGraphAdaptation | None = None
    result = None
    answer = ""
    error: str | None = None
    try:
        tools = FramesWikiTools(
            base_url=config.frames.retriever_base_url,
            max_search_calls=config.frames.max_search_calls,
        )
        functions = {"wiki_search": tools.wiki_search, "wiki_content": tools.wiki_content}
        if config.runtime.graph_adaptation_mode == "generic":
            controller = GoalGraphAdaptation(
                functions,
                {
                    "wiki_search": ToolEffectContract(
                        name="wiki_search",
                        effect="read",
                        deterministic=True,
                        cacheable=True,
                        artifact_kind="search_result",
                    ),
                    "wiki_content": ToolEffectContract(
                        name="wiki_content",
                        effect="read",
                        deterministic=True,
                        cacheable=True,
                        artifact_kind="fetched_resource",
                    ),
                },
                task=example.question,
                expose_graph_api=False,
            )
            hooks = GraphAgentHooks.from_controller(controller).agent_kwargs()
        else:
            hooks = {"runtime_functions": tuple(functions.values())}
        model = OpenAIChatModel(config.model, api_key)
        if config.frames.prompt_variant == "frames-direct-tools-v1":
            agent = DirectToolAgent(
                model=model,
                runtime=config.runtime,
                system_prompt=DIRECT_SYSTEM_PROMPT,
                user_prompt_template=DIRECT_USER_PROMPT,
                functions=functions,
                tool_specs=DIRECT_TOOL_SPECS,
                finalize_prompt=FINALIZE_PROMPT,
            )
        else:
            agent = OriginalPTCAgent(
                model=model,
                search_tools=_EmptySearchTools(),  # type: ignore[arg-type]
                runtime=config.runtime,
                system_prompt=SYSTEM_PROMPT,
                user_prompt_template=OFFICIAL_USER_PROMPT,
                finalize_prompt=FINALIZE_PROMPT,
                ptc_tool_spec=_ptc_spec(config),
                demonstration_messages=_demo_messages(
                    graph_enabled=config.runtime.graph_adaptation_mode == "generic"
                ),
                **hooks,
            )
        result = agent.run(example.question)
        answer = result.answer.strip()
        if result.status != "success" or not answer:
            error = result.error or "agent returned no answer"
    except BaseException as exc:
        error = f"{type(exc).__name__}: {exc}"
    finally:
        if controller is not None:
            controller.finish(answered=bool(answer))

    artifact = config.frames.artifact_dir / example.id
    artifact.mkdir(parents=True, exist_ok=True)
    if result is not None:
        (artifact / "execution.json").write_text(
            json.dumps(
                {"agent": result.to_dict(), "wiki_calls": [] if tools is None else tools.calls},
                ensure_ascii=False,
                indent=2,
                default=repr,
            ),
            encoding="utf-8",
        )
    if controller is not None:
        config.frames.graph_dir.mkdir(parents=True, exist_ok=True)
        (config.frames.graph_dir / f"{example.id}.json").write_text(
            json.dumps(
                controller.graph_artifact(), ensure_ascii=False, indent=2, default=repr
            ),
            encoding="utf-8",
        )
    return {
        "schema_version": 1,
        "benchmark": "frames",
        "split": "test",
        "id": example.id,
        "answer": answer,
        "status": "success" if error is None else "failed",
        "error": error,
        "model": config.model.model,
        "agent": None if result is None else result.to_dict(),
        "wiki_calls": [] if tools is None else tools.calls,
        "graph_telemetry": None if controller is None else controller.telemetry(),
        "duration_seconds": time.time() - started,
    }


def _score_with_mimo(
    config: ExperimentConfig,
    examples: Sequence[FramesExample],
    records: Sequence[Mapping[str, Any]],
) -> dict[str, dict[str, Any]]:
    by_id = {example.id: example for example in examples}
    existing = _load_records(config.frames.grades_path)
    pending = [
        record
        for record in records
        if record["id"] not in existing
        or existing[record["id"]].get("answer") != record.get("answer", "")
    ]
    api_key = config.require_api_key(config.grader.api_key_env) if pending else ""
    model_config = ModelConfig(
        model=config.grader.model,
        base_url=config.grader.base_url,
        api_key_env=config.grader.api_key_env,
        max_completion_tokens=config.grader.max_completion_tokens,
        thinking=config.grader.thinking,
        timeout_seconds=config.grader.timeout_seconds,
        max_retries=config.grader.max_retries,
        temperature=0.0,
    )
    config.frames.grades_path.parent.mkdir(parents=True, exist_ok=True)
    if pending:
        with config.frames.grades_path.open("a", encoding="utf-8", newline="\n") as handle:
            with ThreadPoolExecutor(
                max_workers=min(config.grader.workers, len(pending))
            ) as executor:
                futures = {
                    executor.submit(
                        _judge_one,
                        model_config,
                        api_key,
                        by_id[str(record["id"])],
                        str(record.get("answer", "")),
                    ): str(record["id"])
                    for record in pending
                }
                for future in as_completed(futures):
                    grade = future.result()
                    existing[grade["id"]] = grade
                    handle.write(json.dumps(grade, ensure_ascii=False) + "\n")
                    handle.flush()
    return {example.id: existing[example.id] for example in examples}


def _judge_one(
    model_config: ModelConfig,
    api_key: str,
    example: FramesExample,
    answer: str,
) -> dict[str, Any]:
    prompt = JUDGE_PROMPT.format(
        question=example.question,
        answer=answer,
        reference=example.answer,
    )
    turn = OpenAIChatModel(model_config, api_key).create_turn(
        system=JUDGE_SYSTEM_PROMPT,
        messages=[{"role": "user", "content": prompt}],
        tools=[],
    )
    decision_text = turn.text.rsplit("Decision", 1)[-1]
    matches = re.findall(r"\b(TRUE|FALSE)\b", decision_text, re.IGNORECASE)
    decision = matches[-1].upper() if matches else ""
    return {
        "id": example.id,
        "answer": answer,
        "model": model_config.model,
        "decision": decision,
        "score": 1.0 if decision == "TRUE" else 0.0,
        "raw_response": turn.text,
        "usage": dataclasses.asdict(turn.usage),
    }


def _summarize_scores(
    examples: Sequence[FramesExample], grades: Mapping[str, Mapping[str, Any]]
) -> dict[str, Any]:
    scores = {key: float(value.get("score", 0.0)) for key, value in grades.items()}
    summary = _summarize_numeric_scores(examples, scores)
    summary.update(
        {
            "model": next(iter(grades.values()), {}).get("model"),
            "invalid_decisions": sum(
                grade.get("decision") not in {"TRUE", "FALSE"} for grade in grades.values()
            ),
        }
    )
    return summary


def _summarize_numeric_scores(
    examples: Sequence[FramesExample], scores: Mapping[str, float]
) -> dict[str, Any]:
    by_type: dict[str, dict[str, float | int]] = {}
    labels = sorted({label for example in examples for label in example.reasoning_types})
    for label in labels:
        selected = [example for example in examples if label in example.reasoning_types]
        by_type[label] = {
            "tasks": len(selected),
            "accuracy": sum(scores[example.id] for example in selected) / len(selected),
        }
    return {
        "accuracy": sum(scores[example.id] for example in examples) / len(examples),
        "scored": len(examples),
        "by_reasoning_type": by_type,
    }


def _demo_messages(*, graph_enabled: bool) -> tuple[dict[str, Any], ...]:
    arguments: dict[str, Any] = {
        "code": (
            "queries = ['inventor of Synthetic Meridian instrument',\n"
            "           'Synthetic Meridian inventor birthplace',\n"
            "           'Synthetic Meridian birthplace founding date']\n"
            "hits = [wiki_search(query=q) for q in queries]\n"
            "pages = [wiki_content(doc=group[0]) for group in hits]\n"
            "print([(group[0]['title'], page[:240]) for group, page in zip(hits, pages)])"
        )
    }
    if graph_enabled:
        arguments.update(
            {
                "action": "CONTINUE",
                "target": "task",
                "expected_change": "retrieve the synthetic bridge facts with distinct queries",
            }
        )
    return (
        {
            "role": "user",
            "content": (
                "Planning example with synthetic Wikipedia pages: find the founding date of the "
                "birthplace of the inventor of a hypothetical instrument."
            ),
        },
        {
            "role": "assistant",
            "content": "I will move from the instrument to its inventor, then birthplace and date.",
            "tool_calls": [
                {
                    "id": "frames_demo_1",
                    "type": "function",
                    "function": {
                        "name": "programmatic_tool_call",
                        "arguments": json.dumps(arguments),
                    },
                }
            ],
        },
        {
            "role": "tool",
            "tool_call_id": "frames_demo_1",
            "content": (
                "[('Synthetic Meridian', 'invented by Ada North'), "
                "('Ada North', 'born in Exampleford'), "
                "('Exampleford', 'founded in 1842')]"
            ),
        },
        {"role": "assistant", "content": "1842"},
    )


def _ptc_spec(config: ExperimentConfig) -> dict[str, Any]:
    if config.runtime.graph_adaptation_mode == "off":
        return copy.deepcopy(PTC_SPEC)
    return extend_ptc_spec_with_graph_control(
        PTC_SPEC,
        include_input_artifacts=False,
        target_description="Use task for this FRAMES question.",
    )


def _load_examples(path: Path) -> list[FramesExample]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    return [
        FramesExample(
            id=str(row[""]),
            question=row["Prompt"].strip(),
            answer=row["Answer"].strip(),
            reasoning_types=tuple(
                label.strip() for label in row["reasoning_types"].split("|") if label.strip()
            ),
        )
        for row in rows
    ]


def _load_records(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    records: dict[str, dict[str, Any]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            record = json.loads(line)
            records[str(record["id"])] = record
    return records


def _summarize_run(
    examples: Sequence[FramesExample], records: Sequence[Mapping[str, Any]]
) -> FramesRunSummary:
    calls = [call for record in records for call in record.get("wiki_calls", [])]
    agents = [record.get("agent") or {} for record in records]
    return FramesRunSummary(
        selected=len(examples),
        processed=len(records),
        successful=sum(record.get("status") == "success" for record in records),
        failed=sum(record.get("status") != "success" for record in records),
        wiki_search_calls=sum(call.get("operation") == "wiki_search" for call in calls),
        wiki_content_calls=sum(call.get("operation") == "wiki_content" for call in calls),
        input_tokens=sum(
            int((agent.get("usage") or {}).get("input_tokens", 0)) for agent in agents
        ),
        output_tokens=sum(
            int((agent.get("usage") or {}).get("output_tokens", 0)) for agent in agents
        ),
    )


def _normalize_answer(value: str) -> str:
    lowered = str(value).casefold()
    without_punctuation = "".join(
        " " if character in string.punctuation else character for character in lowered
    )
    without_articles = re.sub(r"\b(a|an|the)\b", " ", without_punctuation)
    return " ".join(without_articles.split())


def _retriever_request(base_url: str, path: str, payload: dict[str, Any] | None) -> Any:
    data = None if payload is None else json.dumps(payload).encode()
    request = Request(
        f"{base_url.rstrip('/')}{path}",
        data=data,
        headers={"Content-Type": "application/json"},
        method="GET" if payload is None else "POST",
    )
    try:
        with urlopen(request, timeout=RETRIEVER_TIMEOUT_SECONDS) as response:
            return json.loads(response.read().decode())
    except HTTPError as exc:
        body = json.loads(exc.read().decode())
        raise RuntimeError(f"FRAMES retriever HTTP {exc.code}: {body.get('error')}") from exc


def _validate_config(config: ExperimentConfig) -> None:
    if config.frames.prompt_variant not in {
        "frames-ptc-official-planning-fewshot",
        "frames-direct-tools-v1",
    }:
        raise ValueError("unsupported FRAMES prompt variant")
    if config.frames.dataset_revision != "58d9fb6330f3ab1316d1eca12e5e8ef23dcc22ef":
        raise ValueError("FRAMES requires the pinned official dataset revision")
    if config.frames.corpus_snapshot != "wikipedia/20230601.en":
        raise ValueError("FRAMES requires the official 2023-06-01 English Wikipedia snapshot")
    if config.frames.search_results != 10 or config.frames.max_search_calls != 25:
        raise ValueError("FRAMES official planning configuration requires top-10 and 25 searches")
    if config.runtime.graph_adaptation_mode not in {"off", "generic"}:
        raise ValueError("FRAMES graph adaptation must be off or generic")
    if (
        config.frames.prompt_variant == "frames-direct-tools-v1"
        and config.runtime.graph_adaptation_mode != "off"
    ):
        raise ValueError("FRAMES direct tools requires graph adaptation off")


def _validate_matched_configs(
    graph_config: ExperimentConfig, baseline_config: ExperimentConfig
) -> None:
    graph = dataclasses.asdict(graph_config)
    baseline = dataclasses.asdict(baseline_config)
    graph["runtime"].pop("graph_adaptation_mode")
    baseline["runtime"].pop("graph_adaptation_mode")
    for key in ("results_path", "grades_path", "report_path", "artifact_dir", "graph_dir"):
        graph["frames"].pop(key)
        baseline["frames"].pop(key)
    if graph != baseline:
        raise ValueError("FRAMES GraphPTC and baseline configs are not matched")
    if graph_config.runtime.graph_adaptation_mode != "generic":
        raise ValueError("FRAMES GraphPTC config must use generic graph adaptation")
    if baseline_config.runtime.graph_adaptation_mode != "off":
        raise ValueError("FRAMES baseline config must disable graph adaptation")
