from __future__ import annotations

import argparse
import hashlib
import json
import re
import sqlite3
import statistics
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from graphptc.browsecomp_plus_benchmark import (
    BROWSECOMP_PLUS_ORIGINAL_PTC_SYSTEM_PROMPT,
    BROWSECOMP_PLUS_ORIGINAL_PTC_TOOL_SPEC,
    BROWSECOMP_PLUS_ORIGINAL_PTC_USER_PROMPT_TEMPLATE,
)
from graphptc.codeact_agent import CodeActPTCAgent
from graphptc.config import ExperimentConfig
from graphptc.local_search import SQLiteCorpusSearchTools
from graphptc.model import ModelTurn, OpenAIChatModel
from graphptc.ptc import extract_result_tag
from graphptc.experiments.phase_planning import PHASE_PLANNING_SUFFIX


CONFIG = Path("configs/browsecomp_plus.original-ptc-v1-turn30-pilot20.toml")
SUITE = Path("data/codeact_validation/heldout12.json")
OUTPUT = Path("runs/ptc_phase_planning_ablation")

@dataclass(frozen=True)
class Variant:
    key: str
    thinking: str
    phase_planning: bool


VARIANTS = (
    Variant("v1_current", "disabled", False),
    Variant("v2_thinking", "enabled", False),
    Variant("v3_phase_plan", "disabled", True),
    Variant("v4_thinking_phase_plan", "enabled", True),
)


class RecordingModel:
    def __init__(self, inner: OpenAIChatModel) -> None:
        self._inner = inner
        self.turns: list[ModelTurn] = []

    def create_turn(self, **kwargs: Any) -> ModelTurn:
        turn = self._inner.create_turn(**kwargs)
        self.turns.append(turn)
        return turn


class ContractSearchTools:
    def __init__(self, inner: SQLiteCorpusSearchTools) -> None:
        self._inner = inner

    @property
    def calls(self) -> list[dict[str, Any]]:
        return self._inner.calls

    def search(self, *, query: str) -> list[dict[str, Any]]:
        return [
            {
                "docid": item["docid"],
                "score": item["score"],
                "snippet": item["snippet"],
            }
            for item in self._inner.search(query=query)
        ]

    def fetch(self, *, docid: str) -> dict[str, Any]:
        item = self._inner.fetch(docid=docid)
        return {"docid": item["docid"], "content": item["content"]}


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the MiMo PTC 2x2 planning ablation.")
    parser.add_argument("--config", type=Path, default=CONFIG)
    parser.add_argument("--suite", type=Path, default=SUITE)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--restart", action="store_true")
    parser.add_argument(
        "--variant",
        action="append",
        choices=[variant.key for variant in VARIANTS],
        help="Run only selected variants; may be repeated.",
    )
    args = parser.parse_args()

    load_dotenv(".env")
    config = ExperimentConfig.from_toml(args.config)
    selected_variants = tuple(
        variant
        for variant in VARIANTS
        if not args.variant or variant.key in args.variant
    )
    suite = json.loads(args.suite.read_text(encoding="utf-8"))
    questions = list(suite["questions"])
    args.output.mkdir(parents=True, exist_ok=True)
    responses_path = args.output / "responses.jsonl"
    report_path = args.output / "report.json"
    index_path = args.output / "corpus.sqlite3"
    if args.restart or not index_path.exists():
        _build_index(index_path, suite["documents"])
    if args.restart:
        responses_path.write_text("", encoding="utf-8")

    completed = _load_completed(responses_path)
    api_key = config.require_api_key(config.model.api_key_env)
    write_lock = threading.Lock()

    for task_index, item in enumerate(questions, 1):
        pending = [
            variant
            for variant in selected_variants
            if (variant.key, item["id"]) not in completed
        ]
        if not pending:
            continue
        with ThreadPoolExecutor(max_workers=len(pending)) as executor:
            futures = {
                executor.submit(
                    _run_one,
                    config,
                    api_key,
                    index_path,
                    suite,
                    item,
                    variant,
                ): variant
                for variant in pending
            }
            for future in as_completed(futures):
                record = future.result()
                with write_lock, responses_path.open(
                    "a", encoding="utf-8", newline="\n"
                ) as handle:
                    handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                print(
                    f"[{task_index}/{len(questions)}] {record['variant']} "
                    f"{record['task_id']} status={record['agent']['status']}",
                    flush=True,
                )

    records = _load_records(responses_path)
    expected = len(questions) * len(selected_variants)
    if len(records) != expected:
        raise RuntimeError(f"Incomplete ablation: {len(records)}/{expected} records")
    report = {
        "schema_version": 1,
        "created_at": datetime.now(UTC).isoformat(),
        "experiment": "mimo_ptc_phase_planning_2x2",
        "config_path": str(args.config),
        "suite_path": str(args.suite),
        "suite_sha256": _sha256(args.suite),
        "tasks": [item["id"] for item in questions],
        "variants": [asdict(variant) for variant in selected_variants],
        "controls": {
            "model": config.model.model,
            "temperature": config.model.temperature,
            "max_completion_tokens": config.model.max_completion_tokens,
            "runtime": asdict(config.runtime),
            "base_prompt_sha256": _text_sha256(
                BROWSECOMP_PLUS_ORIGINAL_PTC_SYSTEM_PROMPT
            ),
            "planning_prompt_sha256": _text_sha256(
                _system_prompt(phase_planning=True)
            ),
            "tool_schema_sha256": _text_sha256(
                json.dumps(
                    BROWSECOMP_PLUS_ORIGINAL_PTC_TOOL_SPEC,
                    sort_keys=True,
                    separators=(",", ":"),
                )
            ),
        },
        "summaries": {
            variant.key: summarize(
                [record for record in records if record["variant"] == variant.key]
            )
            for variant in selected_variants
        },
        "records": records,
    }
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report["summaries"], ensure_ascii=False, indent=2))


def _run_one(
    config: ExperimentConfig,
    api_key: str,
    index_path: Path,
    suite: dict[str, Any],
    item: dict[str, Any],
    variant: Variant,
) -> dict[str, Any]:
    model_config = replace(config.model, thinking=variant.thinking)
    recording_model = RecordingModel(OpenAIChatModel(model_config, api_key))
    inner = SQLiteCorpusSearchTools(
        index_path,
        top_k=5,
        snippet_max_chars=int(suite.get("snippet_max_chars", 512)),
        max_tool_calls=config.browsecomp_plus.max_tool_calls,
    )
    tools = ContractSearchTools(inner)
    agent = CodeActPTCAgent(
        model=recording_model,
        search_tools=tools,  # type: ignore[arg-type]
        runtime=config.runtime,
        system_prompt=_system_prompt(phase_planning=variant.phase_planning),
        user_prompt_template=BROWSECOMP_PLUS_ORIGINAL_PTC_USER_PROMPT_TEMPLATE,
        runtime_functions=(tools.search, tools.fetch),
        persistent=True,
        structured_observation=False,
        ptc_tool_spec=BROWSECOMP_PLUS_ORIGINAL_PTC_TOOL_SPEC,
    )
    result = agent.run(item["question"])
    prediction = extract_result_tag(result.answer) or ""
    return {
        "schema_version": 1,
        "variant": variant.key,
        "thinking": variant.thinking,
        "phase_planning": variant.phase_planning,
        "task_id": item["id"],
        "prediction": prediction,
        "correct": _matches_answer(
            prediction,
            required=item["required_entities"],
            excluded=item["excluded_entities"],
        ),
        "agent": result.to_dict(),
        "model_turns": [
            {
                "text": turn.text,
                "tool_calls": len(turn.tool_calls),
                "stop_reason": turn.stop_reason,
                "reasoning_tokens": turn.usage.reasoning_tokens,
            }
            for turn in recording_model.turns
        ],
    }


def _system_prompt(*, phase_planning: bool) -> str:
    if not phase_planning:
        return BROWSECOMP_PLUS_ORIGINAL_PTC_SYSTEM_PROMPT
    return BROWSECOMP_PLUS_ORIGINAL_PTC_SYSTEM_PROMPT + PHASE_PLANNING_SUFFIX


def summarize(records: list[dict[str, Any]]) -> dict[str, Any]:
    first_blocks = [
        record["agent"]["blocks"][0]
        for record in records
        if record["agent"]["blocks"]
    ]
    all_blocks = [
        block for record in records for block in record["agent"]["blocks"]
    ]
    first_calls = [
        _first_block_calls(record)
        for record in records
        if record["agent"]["blocks"]
    ]
    all_call_groups = [record["agent"]["search_calls"] for record in records]
    turns = [int(record["agent"]["model_requests"]) for record in records]
    first_runtime_calls = [int(block["runtime_calls"]) for block in first_blocks]
    all_runtime_calls = [int(block["runtime_calls"]) for block in all_blocks]
    planning_expected = bool(records and records[0]["phase_planning"])
    same_response_plans = sum(_first_turn_has_phase_plan(record) for record in records)
    return {
        "tasks": len(records),
        "successful": sum(record["agent"]["status"] == "success" for record in records),
        "correct": sum(bool(record["correct"]) for record in records),
        "tasks_with_first_block": len(first_blocks),
        "first_block": {
            "multi_call_rate": _rate(
                sum(value > 1 for value in first_runtime_calls), len(first_blocks)
            ),
            "loop_rate": _analysis_rate(first_blocks, "has_loop"),
            "tool_loop_rate": _tool_loop_rate(first_blocks),
            "filter_rate": _analysis_rate(first_blocks, "has_filter"),
            "aggregation_rate": _analysis_rate(first_blocks, "has_aggregation"),
            "coherent_program_rate": _coherent_program_rate(first_blocks),
            "calls_mean": _mean(first_runtime_calls),
            "calls_median": _median(first_runtime_calls),
            "calls_p90": _p90(first_runtime_calls),
            "repeat_search_rate": _repeat_call_rate(first_calls, "search"),
            "repeat_fetch_rate": _repeat_call_rate(first_calls, "fetch"),
            "repeat_retrieval_rate": _repeat_call_rate(first_calls),
        },
        "all_blocks": {
            "blocks": len(all_blocks),
            "multi_call_rate": _rate(
                sum(value > 1 for value in all_runtime_calls), len(all_blocks)
            ),
            "loop_rate": _analysis_rate(all_blocks, "has_loop"),
            "tool_loop_rate": _tool_loop_rate(all_blocks),
            "filter_rate": _analysis_rate(all_blocks, "has_filter"),
            "aggregation_rate": _analysis_rate(all_blocks, "has_aggregation"),
            "coherent_program_rate": _coherent_program_rate(all_blocks),
            "calls_per_block": _mean(all_runtime_calls),
            "repeat_search_rate": _repeat_call_rate(all_call_groups, "search"),
            "repeat_fetch_rate": _repeat_call_rate(all_call_groups, "fetch"),
            "repeat_retrieval_rate": _repeat_call_rate(all_call_groups),
        },
        "turns": {
            "mean": _mean(turns),
            "median": _median(turns),
            "p90": _p90(turns),
        },
        "phase_plan": {
            "expected": planning_expected,
            "same_response_compliance_rate": (
                _rate(same_response_plans, len(records))
                if planning_expected
                else None
            ),
        },
    }


def _first_block_calls(record: dict[str, Any]) -> list[dict[str, Any]]:
    count = int(record["agent"]["blocks"][0]["runtime_calls"])
    return record["agent"]["search_calls"][:count]


def _first_turn_has_phase_plan(record: dict[str, Any]) -> bool:
    turns = record.get("model_turns", [])
    if not turns:
        return False
    first = turns[0]
    text = str(first.get("text", "")).lower()
    return (
        int(first.get("tool_calls", 0)) > 0
        and "<phase_plan>" in text
        and "stage_goal:" in text
        and "parallel_subgoals:" in text
        and "return_condition:" in text
    )


def _analysis_rate(blocks: list[dict[str, Any]], key: str) -> float | None:
    return _rate(
        sum(bool(block.get("program_analysis", {}).get(key)) for block in blocks),
        len(blocks),
    )


def _tool_loop_rate(blocks: list[dict[str, Any]]) -> float | None:
    return _rate(
        sum(
            int(block.get("program_analysis", {}).get("tool_calls_in_loops", 0)) > 0
            for block in blocks
        ),
        len(blocks),
    )


def _coherent_program_rate(blocks: list[dict[str, Any]]) -> float | None:
    return _rate(
        sum(
            int(block.get("runtime_calls", 0)) > 1
            and int(
                block.get("program_analysis", {}).get("tool_calls_in_loops", 0)
            )
            > 0
            and (
                bool(block.get("program_analysis", {}).get("has_filter"))
                or bool(block.get("program_analysis", {}).get("has_aggregation"))
            )
            for block in blocks
        ),
        len(blocks),
    )


def _repeat_call_rate(
    call_groups: list[list[dict[str, Any]]], operation: str | None = None
) -> float | None:
    total = 0
    repeats = 0
    for calls in call_groups:
        signatures = [
            _call_signature(call)
            for call in calls
            if (operation is None or call.get("operation") == operation)
            and _call_signature(call)
        ]
        total += len(signatures)
        repeats += len(signatures) - len(set(signatures))
    if not total:
        return None
    return repeats / total


def _call_signature(call: dict[str, Any]) -> str:
    operation = str(call.get("operation", "")).strip().lower()
    if operation == "search":
        value = " ".join(str(call.get("query", "")).lower().split())
    elif operation == "fetch":
        value = str(call.get("docid", "")).strip().lower()
    else:
        return ""
    return f"{operation}:{value}" if value else ""


def _matches_answer(
    answer: str, *, required: list[str], excluded: list[str]
) -> bool:
    normalized = " ".join(re.findall(r"[a-z0-9]+", answer.lower()))

    def contains(entity: str) -> bool:
        value = " ".join(re.findall(r"[a-z0-9]+", entity.lower()))
        return value in normalized

    return all(contains(entity) for entity in required) and not any(
        contains(entity) for entity in excluded
    )


def _build_index(path: Path, documents: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        path.unlink()
    connection = sqlite3.connect(path)
    try:
        connection.execute(
            "CREATE TABLE documents (docid TEXT UNIQUE, url TEXT, text TEXT NOT NULL)"
        )
        connection.execute(
            "CREATE VIRTUAL TABLE documents_fts USING fts5(text, content='documents', content_rowid='rowid')"
        )
        connection.execute(
            "CREATE VIRTUAL TABLE documents_vocab USING fts5vocab(documents_fts, 'row')"
        )
        connection.executemany(
            "INSERT INTO documents(docid, url, text) VALUES (?, ?, ?)",
            [(item["docid"], item["url"], item["text"]) for item in documents],
        )
        connection.execute("INSERT INTO documents_fts(documents_fts) VALUES('rebuild')")
        connection.commit()
    finally:
        connection.close()


def _load_completed(path: Path) -> set[tuple[str, str]]:
    return {
        (record["variant"], record["task_id"])
        for record in _load_records(path)
    }


def _load_records(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _rate(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def _mean(values: list[int]) -> float | None:
    return statistics.fmean(values) if values else None


def _median(values: list[int]) -> float | None:
    return statistics.median(values) if values else None


def _p90(values: list[int]) -> int | None:
    if not values:
        return None
    ordered = sorted(values)
    return ordered[max(0, (9 * len(ordered) + 9) // 10 - 1)]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _text_sha256(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


if __name__ == "__main__":
    main()
