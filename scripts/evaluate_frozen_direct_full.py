from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path

from dotenv import load_dotenv

from graphptc.benchmark import _load_records, _summarize_generation, _write_records
from graphptc.browsecomp_plus import (
    BrowseCompPlusGrade,
    evaluate_browsecomp_plus_predictions,
    load_browsecomp_plus,
    summarize_browsecomp_plus_grades,
)
from graphptc.browsecomp_plus_benchmark import (
    _create_judge,
    _evidence_recall,
    _grade_cache_key,
    _grade_from_record,
    _validate_complete_responses,
)
from graphptc.config import ExperimentConfig


CONFIG = Path("configs/browsecomp_plus.direct-tools-v1-turn30-full.toml")
RUN_DIR = Path("runs/browsecomp_plus/direct-tools-v1-turn30-full")
BASE_RESPONSES = RUN_DIR / "responses.jsonl"
RETRY_RESPONSES = RUN_DIR / "retry-584/responses.jsonl"
MERGED_RESPONSES = RUN_DIR / "responses.with-retry-584.jsonl"
GRADES = RUN_DIR / "grades.jsonl"
REPORT = RUN_DIR / "report.json"
FREEZE = RUN_DIR / "freeze.json"
RETRY_ID = "584"


def main() -> None:
    load_dotenv(".env")
    config = ExperimentConfig.from_toml(CONFIG)
    freeze = json.loads(FREEZE.read_text(encoding="utf-8"))
    base = _load_records(BASE_RESPONSES)
    retry = _load_records(RETRY_RESPONSES)
    _validate_provenance(config, freeze, base, retry)

    retry_record = retry[0]
    original_retry_record = next(
        record for record in base if str(record["example_id"]) == RETRY_ID
    )
    merged = [
        retry_record if str(record["example_id"]) == RETRY_ID else record
        for record in base
    ]
    _write_records(MERGED_RESPONSES, merged)

    examples = load_browsecomp_plus(
        config.benchmark.dataset_path,
        expected_examples=config.browsecomp_plus.expected_examples,
    )
    _validate_complete_responses(examples, merged)
    predictions = {
        str(record["example_id"]): str(record.get("prediction", ""))
        for record in merged
    }
    cache_keys = {
        example.example_id: _grade_cache_key(
            example, predictions.get(example.example_id, ""), config.grader
        )
        for example in examples
    }

    cached: dict[str, BrowseCompPlusGrade] = {}
    retained: list[dict[str, object]] = []
    for record in _load_records(GRADES):
        example_id = str(record["example_id"])
        if (
            example_id in cache_keys
            and record.get("cache_key") == cache_keys[example_id]
            and record.get("status") in {"valid", "empty_model_response"}
        ):
            cached[example_id] = _grade_from_record(record)
            retained.append(record)
    _write_records(GRADES, retained)

    pending = [example for example in examples if example.example_id not in cached]
    new: dict[str, BrowseCompPlusGrade] = {}
    if pending:
        judge = _create_judge(config)

        def persist(grade: BrowseCompPlusGrade) -> None:
            record = {
                "cache_key": cache_keys[grade.example_id],
                "grader_model": config.grader.model,
                **grade.to_dict(),
            }
            with GRADES.open("a", encoding="utf-8", newline="\n") as handle:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                handle.flush()
            new[grade.example_id] = grade

        evaluate_browsecomp_plus_predictions(
            pending,
            predictions,
            judge,
            max_workers=config.grader.workers,
            on_grade=persist,
        )

    grade_index = {**cached, **new}
    grades = tuple(grade_index[example.example_id] for example in examples)
    _write_records(
        GRADES,
        [
            {
                "cache_key": cache_keys[grade.example_id],
                "grader_model": config.grader.model,
                **grade.to_dict(),
            }
            for grade in grades
        ],
    )
    summary = summarize_browsecomp_plus_grades(
        grades,
        candidate_retrieval_recall=_evidence_recall(
            examples,
            merged,
            config.browsecomp_plus.qrels_evidence_path,
            record_field="candidate_docids",
        ),
        fetched_evidence_recall=_evidence_recall(
            examples,
            merged,
            config.browsecomp_plus.qrels_evidence_path,
            record_field="fetched_docids",
        ),
    )
    report = {
        "schema_version": 1,
        "benchmark": "browsecomp_plus",
        "run_signature": freeze["run_signature"],
        "variant": "direct-tools-v1",
        "model": config.model.model,
        "grader_model": config.grader.model,
        "created_at": datetime.now(UTC).isoformat(),
        "generation_provenance": {
            "base_responses": str(BASE_RESPONSES),
            "base_records": len(base),
            "base_run_signature": freeze["run_signature"],
            "replacement_example_id": RETRY_ID,
            "original_error": original_retry_record.get("error"),
            "retry_responses": str(RETRY_RESPONSES),
            "retry_run_signature": retry_record["run_signature"],
            "merged_responses": str(MERGED_RESPONSES),
            "frozen_hashes_verified": {
                "config": freeze["sha256"]["config"],
                "dataset": freeze["sha256"]["dataset"],
                "direct_tool_agent": freeze["sha256"]["direct_tool_agent"],
            },
        },
        "grader_configuration": asdict(config.grader),
        "generation": _summarize_generation(merged),
        "summary": summary.to_dict(),
    }
    REPORT.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary.to_dict(), ensure_ascii=False, indent=2))


def _validate_provenance(
    config: ExperimentConfig,
    freeze: dict[str, object],
    base: list[dict[str, object]],
    retry: list[dict[str, object]],
) -> None:
    if len(base) != 830 or len({str(record["example_id"]) for record in base}) != 830:
        raise ValueError("Base responses must contain 830 unique examples")
    frozen_signature = str(freeze["run_signature"])
    if {record.get("run_signature") for record in base} != {frozen_signature}:
        raise ValueError("Base responses do not match the frozen run signature")
    if len(retry) != 1 or str(retry[0].get("example_id")) != RETRY_ID:
        raise ValueError("Retry responses must contain only qid 584")
    if retry[0].get("status") != "success" or not retry[0].get("prediction"):
        raise ValueError("qid 584 retry did not produce a valid prediction")
    paths = {
        "config": CONFIG,
        "dataset": config.benchmark.dataset_path,
        "direct_tool_agent": Path("src/graphptc/direct_tool_agent.py"),
    }
    expected = freeze["sha256"]
    assert isinstance(expected, dict)
    mismatches = {
        name: (_sha256(path), expected[name])
        for name, path in paths.items()
        if _sha256(path) != expected[name]
    }
    if mismatches:
        raise ValueError(f"Frozen Direct inputs changed: {mismatches}")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


if __name__ == "__main__":
    main()
