from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

from dotenv import load_dotenv

from graphptc.browsecomp_plus import (
    BrowseCompPlusEvaluationResult,
    evaluate_browsecomp_plus_predictions,
    load_browsecomp_plus,
    summarize_browsecomp_plus_grades,
)
from graphptc.browsecomp_plus_benchmark import (
    _create_judge,
    _evidence_recall,
    _load_records,
    _response_retriever_metadata,
    _run_signature,
    _run_signature_payload,
    _select_examples,
    _summarize_generation,
    _validate_complete_responses,
    _write_records,
)
from graphptc.config import ExperimentConfig


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Grade a signed BrowseComp-Plus response subset."
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--limit", type=int, required=True)
    args = parser.parse_args()

    load_dotenv(".env")
    config = ExperimentConfig.from_toml(args.config)
    all_examples = load_browsecomp_plus(
        config.benchmark.dataset_path,
        expected_examples=config.browsecomp_plus.expected_examples,
    )
    examples = _select_examples(all_examples, limit=args.limit, example_ids=None)
    records = _load_records(config.benchmark.responses_path)
    retriever_metadata = _response_retriever_metadata(records)
    expected_signature = _run_signature(config, retriever_metadata)
    signatures = {record.get("run_signature") for record in records}
    if signatures != {expected_signature}:
        raise ValueError("Responses do not match the current signed configuration")
    _validate_complete_responses(examples, records)

    predictions = {
        record["example_id"]: str(record.get("prediction", ""))
        for record in records
    }
    grades = evaluate_browsecomp_plus_predictions(
        examples,
        predictions,
        _create_judge(config),
        max_workers=config.grader.workers,
    )
    candidate_recall = _evidence_recall(
        examples,
        records,
        config.browsecomp_plus.qrels_evidence_path,
        record_field="candidate_docids",
    )
    fetched_recall = _evidence_recall(
        examples,
        records,
        config.browsecomp_plus.qrels_evidence_path,
        record_field="fetched_docids",
    )
    summary = summarize_browsecomp_plus_grades(
        grades,
        candidate_retrieval_recall=candidate_recall,
        fetched_evidence_recall=fetched_recall,
    )
    result = BrowseCompPlusEvaluationResult(grades=grades, summary=summary)
    _write_records(
        config.benchmark.grades_path,
        [
            {
                "grader_model": config.grader.model,
                **grade.to_dict(),
            }
            for grade in grades
        ],
    )
    report = {
        "schema_version": 1,
        "benchmark": "browsecomp_plus",
        "evaluation_scope": "fixed_subset",
        "evaluation_example_ids": [example.example_id for example in examples],
        "run_signature": expected_signature,
        "run_configuration": _run_signature_payload(config, retriever_metadata),
        "grader_model": config.grader.model,
        "created_at": datetime.now(UTC).isoformat(),
        "generation": _summarize_generation(records),
        "summary": summary.to_dict(),
    }
    config.benchmark.report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(result.summary.to_dict(), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
