from __future__ import annotations

import copy
import dataclasses
import importlib
import importlib.metadata
import json
import os
import threading
import time
import urllib.parse
from collections.abc import Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from xml.etree import ElementTree

import httpx

from ...agents.direct_tools import DirectToolAgent
from ...agents.original_ptc import PTC_TOOL_SPEC, OriginalPTCAgent
from ...config import ExperimentConfig, ModelConfig
from ...graph.adaptation import GoalGraphAdaptation
from ...graph.hooks import GraphAgentHooks, extend_ptc_spec_with_graph_control
from ...graph.tool_effects import ToolEffectContract
from ...model import OpenAIChatModel

OFFICIAL_OPENBOOK_PROMPT = """Answer the following question, and output only a function call or your answer. If the
answer is a list, output one on each line. Current date: 11-20-2023.
[Question]: {question}"""

SYSTEM_PROMPT = """You are solving the official FanOutQA open-book task through programmatic tool calling.

Your only directly callable model tool is programmatic_tool_call. Its Python source runs in one
persistent namespace for this question. The following official Wikipedia functions are Python globals:

- wiki_search(query, results=10): return best-first page dictionaries containing pageid, revid,
  title, and url.
- wiki_content(doc): return the selected page as Markdown, including tables and infoboxes. Pass one
  dictionary returned by wiki_search.

Use each PTC block as a coherent research program. Python variables persist across blocks. Use loops,
filtering, joins, arithmetic, and aggregation in Python when downstream operations follow mechanically
from retrieved values. Only printed stdout is visible on the next turn, so keep full pages in variables
and print compact evidence needed for the next semantic decision. Use only the supplied Wikipedia
functions and call them with keyword arguments; do not access files, environment variables, the shell,
or any other network source."""

FINALIZE_PROMPT = """Wikipedia tools are now unavailable. Return only the answer supported by the retrieved
evidence. If the answer is a list, output one item per line. Do not emit Python, a plan, or result tags."""

DIRECT_SYSTEM_PROMPT = """You are solving the official FanOutQA open-book task with direct access
to the frozen Wikipedia tools. Use wiki_search to identify relevant pages and wiki_content to read
selected pages. Fan out across all entities required by the question, reconcile the evidence, and
avoid repeated searches. Current date: 11-20-2023. When sufficient evidence is available, return
only the answer; if it is a list, output one item per line."""

DIRECT_USER_PROMPT = """Answer the following question using the Wikipedia tools when needed.
[Question]: {task}"""

DIRECT_TOOL_SPECS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "wiki_search",
            "description": "Search the frozen Wikipedia snapshot and return best-first page records.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "minLength": 1},
                    "results": {"type": "integer", "minimum": 1, "maximum": 50},
                },
                "required": ["query"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "wiki_content",
            "description": "Fetch one Wikipedia page returned by wiki_search as Markdown.",
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
            "Execute one coherent Python research program in the persistent FanOutQA namespace. "
            "wiki_search and wiki_content are available as Python globals."
        ),
    },
}

JUDGE_SYSTEM_PROMPT = (
    "You are comparing a submitted answer to an expert answer on a given question."
)
JUDGE_PROMPT = """[BEGIN DATA]
************
[Question]: {question}
************
[Expert]: {reference}
************
[Submission]: {answer}
************
[END DATA]

Compare the factual content of the submitted answer with the expert answer. Ignore any differences in
style, grammar, or punctuation. The submitted answer may either be a subset or superset of the expert
answer, or it may conflict with it. Determine which case applies. First, write out in a step by step
manner your reasoning about the factual content to be sure that your conclusion is correct. Avoid simply
stating the correct answers at the outset. Then print only the single character "A", "B", "C", "D",
"E", or "F" (without quotes or punctuation) on its own line corresponding to the correct answer. At the
end, repeat just the letter again by itself on a new line.
(A) The submitted answer is a subset of the expert answer and is fully consistent with it.
(B) The submitted answer is a superset of the expert answer and is fully consistent with it.
(C) The submitted answer contains all the same details as the expert answer.
(D) There is a disagreement between the submitted answer and the expert answer.
(E) The answers differ, but these differences don't matter from the perspective of factuality.
(F) The submitted answer does not answer the question or is otherwise invalid."""

KIWIX_HTTP_TIMEOUT_SECONDS = 90.0

_CONTENT_LOCKS: dict[str, threading.Lock] = {}
_CONTENT_LOCKS_GUARD = threading.Lock()


class _EmptySearchTools:
    calls: list[dict[str, Any]] = []


class FanOutWikiTools:
    def __init__(self, *, default_results: int, cache_dir: Path) -> None:
        self.default_results = default_results
        self.calls: list[dict[str, Any]] = []
        self._lock = threading.Lock()
        self._wiki = importlib.import_module("fanoutqa.wiki")
        self._models = importlib.import_module("fanoutqa.models")
        cache_dir.mkdir(parents=True, exist_ok=True)
        self._wiki.KIWIX_CACHE_DIR = cache_dir

    def wiki_search(self, query: str, results: int | None = None) -> list[dict[str, Any]]:
        selected_results = self.default_results if results is None else int(results)
        params = urllib.parse.urlencode(
            {
                "pattern": str(query),
                "start": 0,
                "pageLength": selected_results,
                "books.name": self._wiki.FANOUTQA_KIWIX_ZIMNAME,
                "format": "xml",
            }
        )
        response = httpx.get(
            f"{self._wiki.FANOUTQA_KIWIX_BASE}/search?{params}",
            timeout=KIWIX_HTTP_TIMEOUT_SECONDS,
        )
        response.raise_for_status()
        root = ElementTree.fromstring(response.text)
        docs = [
            self._models.Evidence(
                pageid=0,
                revid=0,
                title=item.findtext("title", default=""),
                url=item.findtext("link", default=""),
            )
            for item in root.findall("./channel/item")
        ]
        output = [dataclasses.asdict(doc) for doc in docs]
        with self._lock:
            self.calls.append(
                {
                    "operation": "wiki_search",
                    "query": str(query),
                    "results": selected_results,
                    "documents": [
                        {"pageid": doc["pageid"], "revid": doc["revid"], "title": doc["title"]}
                        for doc in output
                    ],
                }
            )
        return output

    def wiki_content(self, doc: Mapping[str, Any]) -> str:
        evidence = self._models.Evidence.from_dict(dict(doc))
        with _content_lock(str(evidence.url)):
            cache_name = evidence.url.lstrip("/").replace("/", "-")
            cache_filename = self._wiki.KIWIX_CACHE_DIR / f"{cache_name}.md"
            if cache_filename.exists():
                content = cache_filename.read_text(encoding="utf-8")
            else:
                response = httpx.get(
                    f"{self._wiki.FANOUTQA_KIWIX_BASE}{evidence.url}",
                    timeout=KIWIX_HTTP_TIMEOUT_SECONDS,
                )
                if response.status_code == 404:
                    content = "This page does not exist."
                else:
                    response.raise_for_status()
                    content = self._wiki.markdownify(response.text)
                    cache_filename.write_text(content, encoding="utf-8")
        with self._lock:
            self.calls.append(
                {
                    "operation": "wiki_content",
                    "pageid": evidence.pageid,
                    "revid": evidence.revid,
                    "title": evidence.title,
                    "content_chars": len(content),
                }
            )
        return content


@dataclass(frozen=True)
class FanOutQARunSummary:
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


def inspect_fanoutqa(config: ExperimentConfig) -> dict[str, Any]:
    _validate_config(config)
    _configure_wikipedia(config)
    questions = _load_questions(config.fanoutqa.split)
    if len(questions) != config.fanoutqa.expected_tasks:
        raise ValueError(
            f"expected {config.fanoutqa.expected_tasks} FanOutQA questions, found {len(questions)}"
        )
    ids = [str(question.id) for question in questions]
    if len(ids) != len(set(ids)):
        raise ValueError("FanOutQA question IDs are not unique")
    return {
        "benchmark": "fanoutqa",
        "version": importlib.metadata.version("fanoutqa"),
        "split": config.fanoutqa.split,
        "setting": config.fanoutqa.setting,
        "questions": len(questions),
        "wikipedia_type": config.fanoutqa.wikipedia_type,
        "kiwix_base": config.fanoutqa.kiwix_base,
        "kiwix_zimname": config.fanoutqa.kiwix_zimname,
        "kiwix_http_timeout_seconds": KIWIX_HTTP_TIMEOUT_SECONDS,
        "prompt_variant": config.fanoutqa.prompt_variant,
        "graph_adaptation_mode": config.runtime.graph_adaptation_mode,
    }


def probe_fanoutqa_wikipedia(config: ExperimentConfig) -> dict[str, Any]:
    _validate_config(config)
    _configure_wikipedia(config)
    tools = FanOutWikiTools(
        default_results=config.fanoutqa.search_results,
        cache_dir=config.fanoutqa.wiki_cache_dir,
    )
    hits = tools.wiki_search("Wikipedia", results=1)
    if not hits:
        raise ValueError("official wiki_search returned no pages")
    content = tools.wiki_content(hits[0])
    if not content.strip():
        raise ValueError("official wiki_content returned an empty page")
    return {
        "search_results": len(hits),
        "title": hits[0]["title"],
        "content_chars": len(content),
    }


def run_fanoutqa_benchmark(
    config: ExperimentConfig,
    *,
    task_ids: Sequence[str] = (),
    limit: int | None = None,
    restart: bool = False,
) -> FanOutQARunSummary:
    inspection = inspect_fanoutqa(config)
    questions = _load_questions(config.fanoutqa.split)
    by_id = {str(question.id): question for question in questions}
    if task_ids:
        unknown = sorted(set(task_ids) - by_id.keys())
        if unknown:
            raise ValueError(f"unknown FanOutQA task IDs: {unknown[:5]}")
        selected = [by_id[task_id] for task_id in task_ids]
    else:
        selected = questions
    if limit is not None:
        selected = selected[:limit]

    output_path = config.fanoutqa.results_path
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if restart:
        output_path.write_text("", encoding="utf-8")
    existing = _load_records(output_path)
    pending = [question for question in selected if str(question.id) not in existing]
    api_key = config.require_api_key(config.model.api_key_env) if pending else ""

    if pending:
        with output_path.open("a", encoding="utf-8", newline="\n") as handle:
            with ThreadPoolExecutor(
                max_workers=min(config.fanoutqa.workers, len(pending))
            ) as executor:
                futures = {
                    executor.submit(_run_one, config, question, api_key): question
                    for question in pending
                }
                for future in as_completed(futures):
                    record = future.result()
                    existing[record["id"]] = record
                    handle.write(json.dumps(record, ensure_ascii=False, default=repr) + "\n")
                    handle.flush()

    records = [existing[str(question.id)] for question in selected if str(question.id) in existing]
    _write_submission(config.fanoutqa.submission_path, records)
    summary = _summarize_run(selected, records)
    report = {
        "schema_version": 1,
        "benchmark": "fanoutqa",
        "official_alignment": inspection,
        "model": dataclasses.asdict(config.model),
        "runtime": dataclasses.asdict(config.runtime),
        "summary": summary.to_dict(),
        "scoring": "pending",
    }
    config.fanoutqa.report_path.parent.mkdir(parents=True, exist_ok=True)
    config.fanoutqa.report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, default=repr), encoding="utf-8"
    )
    return summary


def evaluate_fanoutqa_benchmark(config: ExperimentConfig) -> dict[str, Any]:
    _validate_config(config)
    questions = _load_questions(config.fanoutqa.split)
    records = _load_records(config.fanoutqa.results_path)
    selected = [record for record in records.values()]
    if len(selected) != config.fanoutqa.expected_tasks:
        raise ValueError(
            f"FanOutQA results are incomplete: expected {config.fanoutqa.expected_tasks}, "
            f"found {len(selected)}"
        )
    _write_submission(config.fanoutqa.submission_path, selected)

    report = json.loads(config.fanoutqa.report_path.read_text(encoding="utf-8"))
    if config.fanoutqa.split == "test":
        report["scoring"] = {
            "status": "hidden_official_test",
            "submission_path": str(config.fanoutqa.submission_path),
        }
    else:
        answers = [{"id": record["id"], "answer": record["answer"]} for record in selected]
        official = _official_dev_scores(config, questions, answers)
        mimo = _score_with_mimo(config, questions, answers)
        report["scoring"] = {"official_local": official, "mimo_judge": mimo}
    config.fanoutqa.report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, default=repr), encoding="utf-8"
    )
    return report


def compare_fanoutqa_benchmarks(
    graph_config: ExperimentConfig,
    baseline_config: ExperimentConfig,
    output_path: Path,
) -> dict[str, Any]:
    _validate_matched_configs(graph_config, baseline_config)
    graph_report = json.loads(graph_config.fanoutqa.report_path.read_text(encoding="utf-8"))
    baseline_report = json.loads(
        baseline_config.fanoutqa.report_path.read_text(encoding="utf-8")
    )
    graph_scores = graph_report.get("scoring") or {}
    baseline_scores = baseline_report.get("scoring") or {}
    graph_raw = ((graph_scores.get("official_local") or {}).get("raw_accuracy") or {})
    baseline_raw = ((baseline_scores.get("official_local") or {}).get("raw_accuracy") or {})
    if set(graph_raw) != set(baseline_raw) or len(graph_raw) != graph_config.fanoutqa.expected_tasks:
        raise ValueError("FanOutQA paired reports do not contain the same complete dev IDs")
    wins = sum(graph_raw[key] > baseline_raw[key] for key in graph_raw)
    losses = sum(graph_raw[key] < baseline_raw[key] for key in graph_raw)
    report = {
        "schema_version": 1,
        "benchmark": "fanoutqa",
        "split": graph_config.fanoutqa.split,
        "tasks": len(graph_raw),
        "graphptc": graph_scores,
        "fewshot_ptc": baseline_scores,
        "difference": {
            "loose_accuracy": (
                graph_scores["official_local"]["accuracy"]["loose"]
                - baseline_scores["official_local"]["accuracy"]["loose"]
            ),
            "strict_accuracy": (
                graph_scores["official_local"]["accuracy"]["strict"]
                - baseline_scores["official_local"]["accuracy"]["strict"]
            ),
            "mimo_judge_accuracy": (
                graph_scores["mimo_judge"]["accuracy"]
                - baseline_scores["mimo_judge"]["accuracy"]
            ),
            "paired_loose_wins": wins,
            "paired_loose_losses": losses,
            "paired_loose_ties": len(graph_raw) - wins - losses,
        },
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return report


def _run_one(config: ExperimentConfig, question: Any, api_key: str) -> dict[str, Any]:
    task_id = str(question.id)
    started = time.time()
    tools: FanOutWikiTools | None = None
    controller: GoalGraphAdaptation | None = None
    result = None
    error: str | None = None
    try:
        tools = FanOutWikiTools(
            default_results=config.fanoutqa.search_results,
            cache_dir=config.fanoutqa.wiki_cache_dir,
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
                task=str(question.question),
                expose_graph_api=False,
            )
            hooks = GraphAgentHooks.from_controller(controller).agent_kwargs()
        else:
            hooks = {"runtime_functions": tuple(functions.values())}
        model = OpenAIChatModel(config.model, api_key)
        if config.fanoutqa.prompt_variant == "fanoutqa-direct-tools-v1":
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
                user_prompt_template=OFFICIAL_OPENBOOK_PROMPT,
                finalize_prompt=FINALIZE_PROMPT,
                ptc_tool_spec=_ptc_spec(config),
                demonstration_messages=_demo_messages(
                    graph_enabled=config.runtime.graph_adaptation_mode == "generic"
                ),
                **hooks,
            )
        result = agent.run(str(question.question))
        answer = result.answer.strip()
        if result.status != "success" or not answer:
            error = result.error or "agent returned no answer"
    except BaseException as exc:
        answer = ""
        error = f"{type(exc).__name__}: {exc}"
    finally:
        if controller is not None:
            controller.finish(answered=bool(result is not None and result.answer.strip()))

    artifact = config.fanoutqa.artifact_dir / task_id
    artifact.mkdir(parents=True, exist_ok=True)
    if result is not None:
        (artifact / "execution.json").write_text(
            json.dumps(
                {
                    "agent": result.to_dict(),
                    "wiki_calls": [] if tools is None else tools.calls,
                },
                ensure_ascii=False,
                indent=2,
                default=repr,
            ),
            encoding="utf-8",
        )
    if controller is not None:
        config.fanoutqa.graph_dir.mkdir(parents=True, exist_ok=True)
        (config.fanoutqa.graph_dir / f"{task_id}.json").write_text(
            json.dumps(
                controller.graph_artifact(), ensure_ascii=False, indent=2, default=repr
            ),
            encoding="utf-8",
        )
    return {
        "schema_version": 1,
        "benchmark": "fanoutqa",
        "split": config.fanoutqa.split,
        "setting": config.fanoutqa.setting,
        "id": task_id,
        "answer": answer,
        "status": "success" if error is None else "failed",
        "error": error,
        "model": config.model.model,
        "agent": None if result is None else result.to_dict(),
        "wiki_calls": [] if tools is None else tools.calls,
        "graph_telemetry": None if controller is None else controller.telemetry(),
        "duration_seconds": time.time() - started,
    }


def _official_dev_scores(
    config: ExperimentConfig,
    questions: Sequence[Any],
    answers: list[dict[str, str]],
) -> dict[str, Any]:
    os.environ["FANOUTQA_OPENAI_API_KEY"] = config.require_api_key(
        config.grader.api_key_env
    )
    os.environ["FANOUTQA_OPENAI_API_BASE"] = str(config.grader.base_url or "")
    os.environ["FANOUTQA_JUDGE_MODEL"] = config.grader.model
    scorer_module = importlib.import_module("fanoutqa.eval.scorer")
    scorer = scorer_module.Scorer(list(questions), answers, only_score_answered=False)
    accuracy, raw_accuracy = scorer.score_accuracy()
    rouge, _ = scorer.score_rouge()
    return {
        "accuracy": dataclasses.asdict(accuracy),
        "raw_accuracy": raw_accuracy,
        "rouge": dataclasses.asdict(rouge),
        "bleurt": "not_run",
        "implementation": "fanoutqa.eval.Scorer",
    }


def _score_with_mimo(
    config: ExperimentConfig, questions: Sequence[Any], answers: list[dict[str, str]]
) -> dict[str, Any]:
    by_id = {str(question.id): question for question in questions}
    existing = _load_records(config.fanoutqa.grades_path)
    pending = [
        answer
        for answer in answers
        if answer["id"] not in existing
        or existing[answer["id"]].get("answer") != answer["answer"]
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
    config.fanoutqa.grades_path.parent.mkdir(parents=True, exist_ok=True)
    if pending:
        with config.fanoutqa.grades_path.open("a", encoding="utf-8", newline="\n") as handle:
            with ThreadPoolExecutor(max_workers=min(config.grader.workers, len(pending))) as executor:
                futures = {
                    executor.submit(
                        _judge_one,
                        model_config,
                        api_key,
                        by_id[answer["id"]],
                        answer["answer"],
                    ): answer["id"]
                    for answer in pending
                }
                for future in as_completed(futures):
                    grade = future.result()
                    existing[grade["id"]] = grade
                    handle.write(json.dumps(grade, ensure_ascii=False) + "\n")
                    handle.flush()
    grades = [existing[answer["id"]] for answer in answers]
    return {
        "model": config.grader.model,
        "accuracy": sum(float(grade["score"]) for grade in grades) / len(questions),
        "scored": len(grades),
        "invalid_labels": sum(grade.get("letter") not in "ABCDEF" for grade in grades),
        "label_counts": {
            letter: sum(grade["letter"] == letter for grade in grades) for letter in "ABCDEF"
        },
        "positive_labels": ["B", "C", "E"],
    }


def _judge_one(
    model_config: ModelConfig, api_key: str, question: Any, answer: str
) -> dict[str, Any]:
    prompt = JUDGE_PROMPT.format(
        question=question.question,
        reference=_str_answer(question.answer),
        answer=answer[:4000],
    )
    turn = OpenAIChatModel(model_config, api_key).create_turn(
        system=JUDGE_SYSTEM_PROMPT,
        messages=[{"role": "user", "content": prompt}],
        tools=[],
    )
    stripped = turn.text.strip().upper()
    letter = stripped[-1] if stripped and stripped[-1] in "ABCDEF" else ""
    return {
        "id": str(question.id),
        "answer": answer,
        "letter": letter,
        "score": 1.0 if letter in {"B", "C", "E"} else 0.0,
        "raw_response": turn.text,
        "usage": dataclasses.asdict(turn.usage),
    }


def _str_answer(answer: Any) -> str:
    if isinstance(answer, list):
        return "\n".join(_str_answer(item) for item in answer)
    if isinstance(answer, dict):
        return "\n".join(f"{key} - {_str_answer(value)}" for key, value in answer.items())
    if isinstance(answer, bool):
        return "yes" if answer else "no"
    return "" if answer is None else str(answer)


def _demo_messages(*, graph_enabled: bool) -> tuple[dict[str, Any], ...]:
    arguments: dict[str, Any] = {
        "code": (
            "import re\n"
            "root = wiki_search(query='Synthetic Atlas group', results=1)[0]\n"
            "root_text = wiki_content(doc=root)\n"
            "members = ['Aster', 'Birch']  # parsed mechanically from root_text\n"
            "values = {}\n"
            "for member in members:\n"
            "    hit = wiki_search(query=f'{member} Synthetic Atlas', results=1)[0]\n"
            "    page = wiki_content(doc=hit)\n"
            "    values[member] = int(re.search(r'recorded value: (\\d+)', page).group(1))\n"
            "print(values)"
        )
    }
    if graph_enabled:
        arguments.update(
            {
                "action": "CONTINUE",
                "target": "task",
                "expected_change": "retrieve and aggregate the synthetic member values",
            }
        )
    return (
        {
            "role": "user",
            "content": (
                "PTC organization example using entirely synthetic pages: identify the qualifying "
                "members of a hypothetical group and return their recorded values."
            ),
        },
        {
            "role": "assistant",
            "content": "I will retrieve the member list, fan out over its pages, and aggregate compactly.",
            "tool_calls": [
                {
                    "id": "fanoutqa_demo_1",
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
            "tool_call_id": "fanoutqa_demo_1",
            "content": "{'Aster': 7, 'Birch': 11}",
        },
        {
            "role": "assistant",
            "content": "Aster: 7\nBirch: 11",
        },
    )


def _ptc_spec(config: ExperimentConfig) -> dict[str, Any]:
    if config.runtime.graph_adaptation_mode == "off":
        return copy.deepcopy(PTC_SPEC)
    return extend_ptc_spec_with_graph_control(
        PTC_SPEC,
        include_input_artifacts=False,
        target_description="Use task for this FanOutQA question.",
    )


def _configure_wikipedia(config: ExperimentConfig) -> None:
    os.environ["FANOUTQA_WIKIPEDIA_TYPE"] = config.fanoutqa.wikipedia_type
    os.environ["FANOUTQA_KIWIX_BASE"] = config.fanoutqa.kiwix_base
    os.environ["FANOUTQA_KIWIX_ZIMNAME"] = config.fanoutqa.kiwix_zimname
    for key in ("NO_PROXY", "no_proxy"):
        values = [value.strip() for value in os.environ.get(key, "").split(",") if value.strip()]
        for host in ("127.0.0.1", "localhost"):
            if host not in values:
                values.append(host)
        os.environ[key] = ",".join(values)


def _load_questions(split: str) -> list[Any]:
    fanoutqa = importlib.import_module("fanoutqa")
    return fanoutqa.load_dev() if split == "dev" else fanoutqa.load_test()


def _load_records(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    records: dict[str, dict[str, Any]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        records[str(record["id"])] = record
    return records


def _write_submission(path: Path, records: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ordered = sorted(records, key=lambda record: str(record["id"]))
    path.write_text(
        "".join(
            json.dumps({"id": record["id"], "answer": record.get("answer", "")}, ensure_ascii=False)
            + "\n"
            for record in ordered
        ),
        encoding="utf-8",
    )


def _summarize_run(questions: Sequence[Any], records: Sequence[Mapping[str, Any]]) -> FanOutQARunSummary:
    calls = [call for record in records for call in record.get("wiki_calls", [])]
    agents = [record.get("agent") or {} for record in records]
    return FanOutQARunSummary(
        selected=len(questions),
        processed=len(records),
        successful=sum(record.get("status") == "success" for record in records),
        failed=sum(record.get("status") != "success" for record in records),
        wiki_search_calls=sum(call.get("operation") == "wiki_search" for call in calls),
        wiki_content_calls=sum(call.get("operation") == "wiki_content" for call in calls),
        input_tokens=sum(int((agent.get("usage") or {}).get("input_tokens", 0)) for agent in agents),
        output_tokens=sum(int((agent.get("usage") or {}).get("output_tokens", 0)) for agent in agents),
    )


def _validate_config(config: ExperimentConfig) -> None:
    if config.fanoutqa.split not in {"dev", "test"}:
        raise ValueError("fanoutqa.split must be dev or test")
    if config.fanoutqa.setting != "openbook":
        raise ValueError("only the official FanOutQA openbook setting is supported")
    if config.fanoutqa.wikipedia_type != "kiwix":
        raise ValueError("FanOutQA evaluation requires the official local Kiwix snapshot")
    if config.fanoutqa.prompt_variant not in {
        "fanoutqa-ptc-fewshot",
        "fanoutqa-direct-tools-v1",
    }:
        raise ValueError("unsupported FanOutQA prompt variant")
    if config.runtime.graph_adaptation_mode not in {"off", "generic"}:
        raise ValueError("FanOutQA graph adaptation must be off or generic")
    if (
        config.fanoutqa.prompt_variant == "fanoutqa-direct-tools-v1"
        and config.runtime.graph_adaptation_mode != "off"
    ):
        raise ValueError("FanOutQA direct tools requires graph adaptation off")
    version = importlib.metadata.version("fanoutqa")
    if tuple(int(part) for part in version.split(".")[:3]) < (1, 3, 0):
        raise ValueError(f"FanOutQA >=1.3.0 is required, found {version}")


def _validate_matched_configs(
    graph_config: ExperimentConfig, baseline_config: ExperimentConfig
) -> None:
    graph = dataclasses.asdict(graph_config)
    baseline = dataclasses.asdict(baseline_config)
    graph["runtime"].pop("graph_adaptation_mode")
    baseline["runtime"].pop("graph_adaptation_mode")
    path_keys = (
        "wiki_cache_dir",
        "results_path",
        "submission_path",
        "grades_path",
        "report_path",
        "artifact_dir",
        "graph_dir",
    )
    for key in path_keys:
        graph["fanoutqa"].pop(key)
        baseline["fanoutqa"].pop(key)
    if graph != baseline:
        raise ValueError("FanOutQA GraphPTC and baseline configs are not matched")
    if graph_config.runtime.graph_adaptation_mode != "generic":
        raise ValueError("FanOutQA GraphPTC config must use generic graph adaptation")
    if baseline_config.runtime.graph_adaptation_mode != "off":
        raise ValueError("FanOutQA baseline config must disable graph adaptation")


def _content_lock(key: str) -> threading.Lock:
    with _CONTENT_LOCKS_GUARD:
        return _CONTENT_LOCKS.setdefault(key, threading.Lock())
