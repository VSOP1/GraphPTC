from __future__ import annotations

import base64
import csv
from pathlib import Path

from graphptc.browsecomp import (
    BrowseCompExample,
    BrowseCompPrediction,
    derive_key,
    evaluate_browsecomp_predictions,
    load_browsecomp,
    parse_browsecomp_grader_letter,
)


def _encrypt(value: str, password: str) -> str:
    raw = value.encode()
    key = derive_key(password, len(raw))
    return base64.b64encode(
        bytes(a ^ b for a, b in zip(raw, key, strict=True))
    ).decode()


def test_loader_decrypts_official_browsecomp_shape(tmp_path: Path) -> None:
    path = tmp_path / "browsecomp.csv"
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=("problem", "answer", "problem_topic", "canary"),
        )
        writer.writeheader()
        writer.writerow(
            {
                "problem": _encrypt("Which person?", "canary-value"),
                "answer": _encrypt("Ada", "canary-value"),
                "problem_topic": "People",
                "canary": "canary-value",
            }
        )

    examples = load_browsecomp(path, verify_checksum=False)

    assert examples == [
        BrowseCompExample("0", "Which person?", "Ada", "People")
    ]


def test_grader_letter_parser_is_strict() -> None:
    assert parse_browsecomp_grader_letter("A") == "A"
    assert parse_browsecomp_grader_letter("  c\n") == "C"
    assert parse_browsecomp_grader_letter("answer: c") is None
    assert parse_browsecomp_grader_letter("correct: yes") is None


def test_browsecomp_accuracy_includes_empty_and_invalid_responses() -> None:
    examples = [
        BrowseCompExample(str(index), f"question-{index}", "truth", "Other")
        for index in range(5)
    ]
    predictions = [
        BrowseCompPrediction("0", "right"),
        BrowseCompPrediction("1", "wrong"),
        BrowseCompPrediction("2", "unsure"),
        BrowseCompPrediction("3", "invalid"),
    ]

    class FakeJudge:
        def judge(self, prompt: str) -> str:
            if "<sample_answer>right</sample_answer>" in prompt:
                return "A"
            if "<sample_answer>wrong</sample_answer>" in prompt:
                return "B"
            if "<sample_answer>unsure</sample_answer>" in prompt:
                return "C"
            return "not a letter"

    result = evaluate_browsecomp_predictions(
        examples, predictions, FakeJudge(), max_workers=2
    )

    assert result.summary.total_examples == 5
    assert result.summary.valid_examples == 3
    assert result.summary.correct == 1
    assert result.summary.incorrect == 1
    assert result.summary.abstained == 1
    assert result.summary.invalid_auto_rater_responses == 1
    assert result.summary.empty_model_responses == 1
    assert result.summary.accuracy == 0.2
