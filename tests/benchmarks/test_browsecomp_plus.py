from __future__ import annotations

import json
from pathlib import Path

import pytest

import graphptc.browsecomp_plus as browsecomp_plus
from graphptc.browsecomp_plus import (
    BrowseCompPlusExample,
    evaluate_browsecomp_plus_predictions,
    load_browsecomp_plus,
    parse_browsecomp_plus_judgement,
    summarize_browsecomp_plus_grades,
)


def test_loader_validates_jsonl_shape(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "questions.jsonl"
    rows = [
        {"query_id": "2", "query": "question two", "answer": "answer two"},
        {"query_id": "9", "query": "question nine", "answer": "answer nine"},
    ]
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )
    monkeypatch.setattr(browsecomp_plus, "BROWSECOMP_PLUS_EXPECTED_EXAMPLES", 2)

    assert load_browsecomp_plus(path) == [
        BrowseCompPlusExample("2", "question two", "answer two"),
        BrowseCompPlusExample("9", "question nine", "answer nine"),
    ]


def test_official_style_judgement_parser_and_accuracy() -> None:
    assert parse_browsecomp_plus_judgement(
        "extracted_final_answer: x\nreasoning: match\ncorrect: yes\nconfidence: 80%"
    ) == (True, 80.0)
    assert parse_browsecomp_plus_judgement("correct: maybe") is None

    examples = [
        BrowseCompPlusExample("1", "q1", "a1"),
        BrowseCompPlusExample("2", "q2", "a2"),
    ]

    class Judge:
        def judge(self, prompt: str) -> str:
            return "correct: yes\nconfidence: 100"

    grades = evaluate_browsecomp_plus_predictions(
        examples, {"1": "a1"}, Judge(), max_workers=1
    )
    summary = summarize_browsecomp_plus_grades(grades)

    assert summary.valid_examples == 1
    assert summary.empty_model_responses == 1
    assert summary.accuracy == 0.5

