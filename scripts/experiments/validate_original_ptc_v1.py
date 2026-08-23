from __future__ import annotations

import json
import re
import sqlite3
from dataclasses import asdict
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
from graphptc.model import OpenAIChatModel
from graphptc.ptc import extract_result_tag


CONFIG = Path("configs/browsecomp_plus.original-ptc-v1-turn30-pilot20.toml")
SUITE = Path("data/codeact_validation/heldout12.json")
OUTPUT = Path("runs/original_ptc_v1/synthetic_high_density/report.json")
TASK_IDS = {
    "batch_filter",
    "set_intersection",
    "numeric_aggregation",
    "group_count",
    "multi_predicate",
}


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
    load_dotenv(".env")
    config = ExperimentConfig.from_toml(CONFIG)
    suite = json.loads(SUITE.read_text(encoding="utf-8"))
    questions = [item for item in suite["questions"] if item["id"] in TASK_IDS]
    if {item["id"] for item in questions} != TASK_IDS:
        raise ValueError("Synthetic high-density task selection is incomplete")

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    index_path = OUTPUT.parent / "corpus.sqlite3"
    _build_index(index_path, suite["documents"])
    api_key = config.require_api_key(config.model.api_key_env)
    records: list[dict[str, Any]] = []

    for item in questions:
        inner = SQLiteCorpusSearchTools(
            index_path,
            top_k=5,
            snippet_max_chars=int(suite.get("snippet_max_chars", 512)),
            max_tool_calls=1000,
        )
        tools = ContractSearchTools(inner)
        agent = CodeActPTCAgent(
            model=OpenAIChatModel(config.model, api_key),
            search_tools=tools,  # type: ignore[arg-type]
            runtime=config.runtime,
            system_prompt=BROWSECOMP_PLUS_ORIGINAL_PTC_SYSTEM_PROMPT,
            user_prompt_template=BROWSECOMP_PLUS_ORIGINAL_PTC_USER_PROMPT_TEMPLATE,
            runtime_functions=(tools.search, tools.fetch),
            persistent=True,
            ptc_tool_spec=BROWSECOMP_PLUS_ORIGINAL_PTC_TOOL_SPEC,
        )
        result = agent.run(item["question"])
        prediction = extract_result_tag(result.answer) or ""
        correct = _matches_synthetic_answer(
            prediction,
            required=item["required_entities"],
            excluded=item["excluded_entities"],
        )
        records.append(
            {
                "task_id": item["id"],
                "prediction": prediction,
                "correct": correct,
                "agent": result.to_dict(),
            }
        )

    blocks = [block for record in records for block in record["agent"]["blocks"]]
    coherent_blocks = [
        block
        for block in blocks
        if int(block["runtime_calls"]) > 1
        and int(block["program_analysis"].get("tool_calls_in_loops", 0)) > 0
        and (
            bool(block["program_analysis"].get("has_filter"))
            or bool(block["program_analysis"].get("has_aggregation"))
        )
    ]
    summary = {
        "tasks": len(records),
        "successful": sum(record["agent"]["status"] == "success" for record in records),
        "correct": sum(bool(record["correct"]) for record in records),
        "ptc_blocks": len(blocks),
        "runtime_calls": sum(int(block["runtime_calls"]) for block in blocks),
        "multi_call_blocks": sum(int(block["runtime_calls"]) > 1 for block in blocks),
        "tool_loop_blocks": sum(
            int(block["program_analysis"].get("tool_calls_in_loops", 0)) > 0
            for block in blocks
        ),
        "filter_blocks": sum(
            bool(block["program_analysis"].get("has_filter")) for block in blocks
        ),
        "aggregation_blocks": sum(
            bool(block["program_analysis"].get("has_aggregation")) for block in blocks
        ),
        "coherent_processing_blocks": len(coherent_blocks),
        "raw_stdout_only": all(
            "PTC_OBSERVATION" not in str(block["stdout"]) for block in blocks
        ),
    }
    passed = (
        summary["successful"] == len(records)
        and summary["correct"] == len(records)
        and summary["multi_call_blocks"] > 0
        and summary["tool_loop_blocks"] > 0
        and summary["aggregation_blocks"] > 0
        and summary["coherent_processing_blocks"] > 0
        and summary["raw_stdout_only"]
    )
    report = {
        "schema_version": 1,
        "created_at": datetime.now(UTC).isoformat(),
        "variant": "original-ptc-v1",
        "config": asdict(config),
        "suite": str(SUITE),
        "task_ids": sorted(TASK_IDS),
        "summary": summary,
        "passed": passed,
        "records": records,
    }
    OUTPUT.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, default=str) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"passed": passed, "summary": summary}, ensure_ascii=False))
    if not passed:
        raise SystemExit(1)


def _matches_synthetic_answer(
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


if __name__ == "__main__":
    main()
