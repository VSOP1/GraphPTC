from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from graphptc.browsecomp_plus import (
    BrowseCompPlusExample,
    load_browsecomp_plus,
    summarize_browsecomp_plus_grades,
)
from graphptc.browsecomp_plus_benchmark import (
    _evidence_recall,
    _grade_from_record,
    _load_records,
    _summarize_generation,
)


ROOT = Path("runs/browsecomp_plus")
OUTPUT = ROOT / "pilot100-comparison/report.json"
QRELS = Path("data/browsecomp_plus/qrel_evidence.txt")

VARIANTS = {
    "original_ptc": (
        ROOT / "original-ptc-v1-turn30-pilot20",
        ROOT / "original-ptc-v1-turn30-extra80",
    ),
    "positive_fewshot": (
        ROOT / "fewshot-ptc-v1-turn30-pilot20",
        ROOT / "fewshot-ptc-v1-turn30-extra80",
    ),
}
DIRECT = ROOT / "direct-tools-v1-turn30-full"


def main() -> None:
    pilot = load_browsecomp_plus(
        "data/browsecomp_plus/pilot20.questions.jsonl", expected_examples=20
    )
    extra = load_browsecomp_plus(
        "data/browsecomp_plus/extra80.questions.jsonl", expected_examples=80
    )
    examples = [*pilot, *extra]
    ids = [example.example_id for example in examples]
    if len(ids) != 100 or len(set(ids)) != 100:
        raise ValueError("Expected 100 unique pilot20 + extra80 query IDs")

    summaries: dict[str, Any] = {}
    for name, run_dirs in VARIANTS.items():
        grades = _merge_records(run_dirs, "grades.jsonl", ids)
        if name == "original_ptc":
            summaries[name] = _summarize_from_components(grades, run_dirs)
            summaries[name]["artifact_note"] = (
                "The frozen pilot20 grades/report are complete, but its responses.jsonl "
                "was truncated by a later interrupted restart. Score is merged from 100 "
                "grades; retrieval and generation metrics are combined from component reports."
            )
        else:
            responses = _merge_records(run_dirs, "responses.jsonl", ids)
            summaries[name] = _summarize(examples, responses, grades)
        summaries[name]["components"] = [
            _component(run_dir) for run_dir in run_dirs
        ]
        summaries[name]["split_scores"] = {
            "pilot20": _score_only(_load_records(run_dirs[0] / "grades.jsonl")),
            "extra80": _score_only(_load_records(run_dirs[1] / "grades.jsonl")),
            "combined100": _score_only(grades),
        }

    direct_responses = _filter_records(
        _load_records(DIRECT / "responses.jsonl"), set(ids)
    )
    direct_grades = _filter_records(
        _load_records(DIRECT / "grades.jsonl"), set(ids)
    )
    _validate_ids(direct_responses, ids, "direct responses")
    _validate_ids(direct_grades, ids, "direct grades")
    summaries["direct_tools"] = _summarize(
        examples, direct_responses, direct_grades
    )
    summaries["direct_tools"]["components"] = [_component(DIRECT)]
    direct_grade_index = {
        str(record["example_id"]): record
        for record in _load_records(DIRECT / "grades.jsonl")
    }
    summaries["direct_tools"]["split_scores"] = {
        "pilot20": _score_only(
            [direct_grade_index[example.example_id] for example in pilot]
        ),
        "extra80": _score_only(
            [direct_grade_index[example.example_id] for example in extra]
        ),
        "combined100": _score_only(direct_grades),
    }

    report = {
        "schema_version": 1,
        "benchmark": "browsecomp_plus",
        "scope": "frozen_pilot100",
        "created_at": datetime.now(UTC).isoformat(),
        "selection": {
            "count": len(ids),
            "query_ids": ids,
            "pilot20_manifest_sha256": _sha256(
                Path("data/browsecomp_plus/pilot20.manifest.json")
            ),
            "extra80_manifest_sha256": _sha256(
                Path("data/browsecomp_plus/extra80.manifest.json")
            ),
        },
        "summaries": summaries,
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summaries, ensure_ascii=False, indent=2))


def _summarize(
    examples: list[BrowseCompPlusExample],
    responses: list[dict[str, Any]],
    grade_records: list[dict[str, Any]],
) -> dict[str, Any]:
    grades = tuple(_grade_from_record(record) for record in grade_records)
    summary = summarize_browsecomp_plus_grades(
        grades,
        candidate_retrieval_recall=_evidence_recall(
            examples, responses, QRELS, record_field="candidate_docids"
        ),
        fetched_evidence_recall=_evidence_recall(
            examples, responses, QRELS, record_field="fetched_docids"
        ),
    )
    return {
        "summary": summary.to_dict(),
        "generation": _summarize_generation(responses),
    }


def _summarize_from_components(
    grade_records: list[dict[str, Any]], run_dirs: tuple[Path, ...]
) -> dict[str, Any]:
    reports = [
        json.loads((run_dir / "report.json").read_text(encoding="utf-8"))
        for run_dir in run_dirs
    ]
    grades = tuple(_grade_from_record(record) for record in grade_records)
    total = sum(report["summary"]["total_examples"] for report in reports)

    def weighted(field: str) -> float:
        return sum(
            report["summary"][field] * report["summary"]["total_examples"]
            for report in reports
        ) / total

    summary = summarize_browsecomp_plus_grades(
        grades,
        candidate_retrieval_recall=weighted("candidate_retrieval_recall"),
        fetched_evidence_recall=weighted("fetched_evidence_recall"),
    )
    return {
        "summary": summary.to_dict(),
        "generation": _combine_generation(
            [report["generation"] for report in reports]
        ),
    }


def _combine_generation(generations: list[dict[str, Any]]) -> dict[str, Any]:
    summed_fields = (
        "total_records",
        "successful",
        "failed",
        "model_requests",
        "ptc_blocks",
        "runtime_calls",
        "direct_tool_rounds",
        "direct_model_tool_calls",
        "repeated_exact_search_queries",
        "repeated_result_slots",
        "total_result_slots",
        "blocks_with_tool_calls_in_loops",
        "blocks_with_filtering",
        "blocks_with_aggregation",
    )
    combined = {
        field: sum(int(generation.get(field, 0)) for generation in generations)
        for field in summed_fields
    }
    blocks = combined["ptc_blocks"]
    combined["mean_runtime_calls_per_ptc_block"] = (
        combined["runtime_calls"] / blocks if blocks else None
    )
    slots = combined["total_result_slots"]
    combined["repeated_result_slot_rate"] = (
        combined["repeated_result_slots"] / slots if slots else None
    )
    combined["usage"] = {
        field: sum(
            int(generation.get("usage", {}).get(field, 0))
            for generation in generations
        )
        for field in (
            "input_tokens",
            "output_tokens",
            "cached_input_tokens",
            "reasoning_tokens",
        )
    }
    return combined


def _score_only(grade_records: list[dict[str, Any]]) -> dict[str, Any]:
    grades = tuple(_grade_from_record(record) for record in grade_records)
    summary = summarize_browsecomp_plus_grades(grades)
    return {
        "total_examples": summary.total_examples,
        "valid_examples": summary.valid_examples,
        "correct": summary.correct,
        "accuracy": summary.accuracy,
    }


def _merge_records(
    run_dirs: tuple[Path, ...], filename: str, expected_ids: list[str]
) -> list[dict[str, Any]]:
    records = [
        record
        for run_dir in run_dirs
        for record in _load_records(run_dir / filename)
    ]
    _validate_ids(records, expected_ids, filename)
    return records


def _filter_records(
    records: list[dict[str, Any]], selected_ids: set[str]
) -> list[dict[str, Any]]:
    return [record for record in records if str(record["example_id"]) in selected_ids]


def _validate_ids(
    records: list[dict[str, Any]], expected_ids: list[str], label: str
) -> None:
    actual = [str(record["example_id"]) for record in records]
    if len(actual) != len(set(actual)):
        raise ValueError(f"Duplicate IDs in {label}")
    if set(actual) != set(expected_ids):
        missing = sorted(set(expected_ids) - set(actual))
        extra = sorted(set(actual) - set(expected_ids))
        raise ValueError(f"ID mismatch in {label}: missing={missing}, extra={extra}")


def _component(run_dir: Path) -> dict[str, Any]:
    report = json.loads((run_dir / "report.json").read_text(encoding="utf-8"))
    return {
        "run_dir": str(run_dir),
        "run_signature": report["run_signature"],
        "examples": report["summary"]["total_examples"],
    }


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


if __name__ == "__main__":
    main()
