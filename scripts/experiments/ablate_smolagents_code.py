from __future__ import annotations

import argparse
import hashlib
import json
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from .ablate_ptc_phase_planning import (
    ContractSearchTools,
    _build_index,
    _load_completed,
    _load_records,
    _matches_answer,
    summarize,
)
from graphptc.config import ExperimentConfig
from graphptc.local_search import SQLiteCorpusSearchTools
from graphptc.experiments.smolagents_code import SmolagentsCodeRunner


CONFIG = Path("configs/browsecomp_plus.original-ptc-v1-turn30-pilot20.toml")
SUITE = Path("data/codeact_validation/fewshot_eval8.json")
ORIGINAL_REPORT = Path("runs/original_ptc_v1/fewshot_eval8/report.json")
OUTPUT = Path("runs/smolagents_code_ablation")
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare stock smolagents CodeAgent with frozen Original PTC results."
    )
    parser.add_argument("--config", type=Path, default=CONFIG)
    parser.add_argument("--suite", type=Path, default=SUITE)
    parser.add_argument("--original-report", type=Path, default=ORIGINAL_REPORT)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--restart", action="store_true")
    parser.add_argument(
        "--unstructured-search",
        action="store_true",
        help="Reproduce the initial stock-tool control without a search output schema.",
    )
    args = parser.parse_args()
    variant = (
        "smolagents-code-v1"
        if args.unstructured_search
        else "smolagents-structured-v1"
    )

    load_dotenv(".env")
    config = ExperimentConfig.from_toml(args.config)
    suite = json.loads(args.suite.read_text(encoding="utf-8"))
    questions = list(suite["questions"])
    if args.limit is not None:
        questions = questions[: args.limit]
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
    pending = [item for item in questions if (variant, item["id"]) not in completed]
    write_lock = threading.Lock()
    with ThreadPoolExecutor(max_workers=min(args.workers, len(pending) or 1)) as executor:
        futures = {
            executor.submit(
                _run_one,
                config,
                api_key,
                index_path,
                suite,
                item,
                variant,
                not args.unstructured_search,
            ): item
            for item in pending
        }
        for future in as_completed(futures):
            record = future.result()
            with write_lock, responses_path.open(
                "a", encoding="utf-8", newline="\n"
            ) as handle:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            print(
                f"{record['task_id']} status={record['agent']['status']} "
                f"correct={record['correct']}",
                flush=True,
            )

    task_ids = {item["id"] for item in questions}
    records = [
        record
        for record in _load_records(responses_path)
        if record["variant"] == variant and record["task_id"] in task_ids
    ]
    if len(records) != len(questions):
        raise RuntimeError(f"Incomplete experiment: {len(records)}/{len(questions)} records")
    challenger = summarize(records)
    original_report = json.loads(args.original_report.read_text(encoding="utf-8"))
    suite_sha256 = hashlib.sha256(args.suite.read_bytes()).hexdigest()
    if original_report["suite_sha256"] != suite_sha256:
        raise RuntimeError("Original and challenger suite hashes do not match")
    original = original_report["summaries"]["original"]
    full_suite = len(questions) == len(suite["questions"])
    gates = _gates(original, challenger) if full_suite else None
    report = {
        "schema_version": 1,
        "created_at": datetime.now(UTC).isoformat(),
        "experiment": "stock_smolagents_codeagent_vs_frozen_original_ptc",
        "variant": variant,
        "smolagents_version": "1.26.0",
        "config_path": str(args.config),
        "suite_path": str(args.suite),
        "suite_sha256": suite_sha256,
        "tasks": [item["id"] for item in questions],
        "original_report_path": str(args.original_report),
        "original_suite_sha256": original_report["suite_sha256"],
        "comparison": {"original": original, variant: challenger},
        "synthetic_gate": gates,
        "records": records,
    }
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({"summary": challenger, "synthetic_gate": gates}, indent=2))


def _run_one(
    config: ExperimentConfig,
    api_key: str,
    index_path: Path,
    suite: dict[str, Any],
    item: dict[str, Any],
    variant: str,
    structured_search_schema: bool,
) -> dict[str, Any]:
    inner = SQLiteCorpusSearchTools(
        index_path,
        top_k=5,
        snippet_max_chars=int(suite.get("snippet_max_chars", 512)),
        max_tool_calls=config.browsecomp_plus.max_tool_calls,
    )
    tools = ContractSearchTools(inner)
    result = SmolagentsCodeRunner(
        config,
        api_key,
        tools,
        structured_search_schema=structured_search_schema,
    ).run(item["question"])
    prediction = result.answer
    return {
        "schema_version": 1,
        "variant": variant,
        "phase_planning": False,
        "task_id": item["id"],
        "prediction": prediction,
        "correct": _matches_answer(
            prediction,
            required=item["required_entities"],
            excluded=item["excluded_entities"],
        ),
        "agent": result.to_dict(),
    }


def _gates(
    original: dict[str, Any], challenger: dict[str, Any]
) -> dict[str, Any]:
    checks = {
        "correctness_not_lower": challenger["correct"] >= original["correct"],
        "first_multi_call_not_lower": (
            challenger["first_block"]["multi_call_rate"]
            >= original["first_block"]["multi_call_rate"]
        ),
        "first_coherent_program_improved": (
            challenger["first_block"]["coherent_program_rate"]
            > original["first_block"]["coherent_program_rate"]
        ),
        "mean_turns_lower": challenger["turns"]["mean"] < original["turns"]["mean"],
        "repeat_retrieval_not_higher": (
            challenger["all_blocks"]["repeat_retrieval_rate"]
            <= original["all_blocks"]["repeat_retrieval_rate"]
        ),
    }
    return {"passed": all(checks.values()), "checks": checks}


if __name__ == "__main__":
    main()
