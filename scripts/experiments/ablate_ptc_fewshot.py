from __future__ import annotations

import argparse
import hashlib
import json
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from .ablate_ptc_phase_planning import (
    ContractSearchTools,
    RecordingModel,
    _build_index,
    _load_completed,
    _load_records,
    _matches_answer,
    summarize,
)
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
from graphptc.experiments.ptc_fewshot import PTC_FEW_SHOT_MESSAGES


CONFIG = Path("configs/browsecomp_plus.original-ptc-v1-turn30-pilot20.toml")
SUITE = Path("data/codeact_validation/fewshot_eval8.json")
OUTPUT = Path("runs/ptc_positive_fewshot_ablation")


@dataclass(frozen=True)
class Variant:
    key: str
    demonstrations: str


VARIANTS = (
    Variant("original", "none"),
    Variant("fewshot", "positive"),
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the MiMo PTC few-shot ablation.")
    parser.add_argument("--config", type=Path, default=CONFIG)
    parser.add_argument("--suite", type=Path, default=SUITE)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--restart", action="store_true")
    args = parser.parse_args()

    load_dotenv(".env")
    config = ExperimentConfig.from_toml(args.config)
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
            for variant in VARIANTS
            if (variant.key, item["id"]) not in completed
        ]
        with ThreadPoolExecutor(max_workers=len(pending)) as executor:
            futures = {
                executor.submit(
                    _run_one, config, api_key, index_path, suite, item, variant
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
    expected = len(questions) * len(VARIANTS)
    if len(records) != expected:
        raise RuntimeError(f"Incomplete ablation: {len(records)}/{expected} records")
    summaries = {
        variant.key: _summarize_variant(
            [record for record in records if record["variant"] == variant.key]
        )
        for variant in VARIANTS
    }
    report = {
        "schema_version": 1,
        "created_at": datetime.now(UTC).isoformat(),
        "experiment": "mimo_ptc_fewshot_controlled_ablation",
        "config_path": str(args.config),
        "suite_path": str(args.suite),
        "suite_sha256": hashlib.sha256(args.suite.read_bytes()).hexdigest(),
        "variants": [asdict(variant) for variant in VARIANTS],
        "summaries": summaries,
        "records": records,
    }
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summaries, ensure_ascii=False, indent=2))


def _run_one(
    config: ExperimentConfig,
    api_key: str,
    index_path: Path,
    suite: dict[str, Any],
    item: dict[str, Any],
    variant: Variant,
) -> dict[str, Any]:
    model = RecordingModel(OpenAIChatModel(config.model, api_key))
    inner = SQLiteCorpusSearchTools(
        index_path,
        top_k=5,
        snippet_max_chars=int(suite.get("snippet_max_chars", 512)),
        max_tool_calls=config.browsecomp_plus.max_tool_calls,
    )
    tools = ContractSearchTools(inner)
    result = CodeActPTCAgent(
        model=model,
        search_tools=tools,  # type: ignore[arg-type]
        runtime=config.runtime,
        system_prompt=BROWSECOMP_PLUS_ORIGINAL_PTC_SYSTEM_PROMPT,
        user_prompt_template=BROWSECOMP_PLUS_ORIGINAL_PTC_USER_PROMPT_TEMPLATE,
        runtime_functions=(tools.search, tools.fetch),
        persistent=True,
        structured_observation=False,
        ptc_tool_spec=BROWSECOMP_PLUS_ORIGINAL_PTC_TOOL_SPEC,
        demonstration_messages=_demonstrations(variant),
    ).run(item["question"])
    prediction = extract_result_tag(result.answer) or ""
    return {
        "schema_version": 1,
        "variant": variant.key,
        "phase_planning": False,
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
            }
            for turn in model.turns
        ],
    }


def _demonstrations(variant: Variant) -> tuple[dict[str, Any], ...]:
    if variant.demonstrations == "positive":
        return PTC_FEW_SHOT_MESSAGES
    return ()


def _summarize_variant(records: list[dict[str, Any]]) -> dict[str, Any]:
    result = summarize(records)
    total_slots = 0
    new_slots = 0
    for record in records:
        seen: set[str] = set()
        for call in record["agent"]["search_calls"]:
            docids = [str(value) for value in call.get("docids", ())]
            total_slots += len(docids)
            new_slots += sum(docid not in seen for docid in docids)
            seen.update(docids)
    result["all_blocks"]["new_result_slot_rate"] = (
        new_slots / total_slots if total_slots else None
    )
    return result


if __name__ == "__main__":
    main()
