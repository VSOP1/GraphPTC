from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path

import pytest

import graphptc.deepsearchqa as dsqa
from graphptc.deepsearchqa import (
    DatasetValidationError,
    DeepSearchQAExample,
    GeminiJudge,
    JudgeResponseError,
    MissingAPIKeyError,
    OpenAICompatibleJudge,
    Prediction,
    PredictionValidationError,
    build_judge_prompt,
    download_deepsearchqa,
    evaluate_predictions,
    load_deepsearchqa,
    load_predictions,
    parse_judge_response,
    save_predictions,
)


def _dataset_bytes(*, include_example_id: bool = True) -> bytes:
    columns = ["problem", "problem_category", "answer", "answer_type"]
    if include_example_id:
        columns.insert(0, "example_id")

    output = io.StringIO(newline="")
    import csv

    writer = csv.DictWriter(output, fieldnames=columns, lineterminator="\n")
    writer.writeheader()
    for index in range(900):
        row = {
            "problem": f"Question {index}?",
            "problem_category": "Science",
            "answer": f"Answer {index}",
            "answer_type": "Set Answer" if index % 2 else "Single Answer",
        }
        if include_example_id:
            row["example_id"] = str(index)
        writer.writerow(row)
    return output.getvalue().encode()


@pytest.mark.parametrize("include_example_id", [True, False])
def test_loads_kaggle_and_hugging_face_csv_shapes(
    tmp_path: Path, include_example_id: bool
) -> None:
    path = tmp_path / "DSQA-full.csv"
    path.write_bytes(_dataset_bytes(include_example_id=include_example_id))

    examples = load_deepsearchqa(path)

    assert len(examples) == 900
    assert examples[0].example_id == "0"
    assert examples[-1].answer == "Answer 899"


def test_rejects_incomplete_dataset(tmp_path: Path) -> None:
    path = tmp_path / "DSQA-full.csv"
    lines = _dataset_bytes().decode().splitlines()
    path.write_text("\n".join(lines[:-1]), encoding="utf-8")

    with pytest.raises(DatasetValidationError, match="must contain 900 rows"):
        load_deepsearchqa(path)


def test_download_validates_checksum_before_writing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    data = _dataset_bytes()
    monkeypatch.setattr(dsqa, "urlopen", lambda *args, **kwargs: io.BytesIO(data))
    path = tmp_path / "nested" / "DSQA-full.csv"

    with pytest.raises(DatasetValidationError, match="checksum mismatch"):
        download_deepsearchqa(path, url="https://example.test/data.csv")

    assert not path.exists()

    result = download_deepsearchqa(
        path,
        url="https://example.test/data.csv",
        expected_sha256=hashlib.sha256(data).hexdigest(),
    )
    assert result == path
    assert path.read_bytes() == data


def test_missing_local_dataset_error_explains_download_option(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="download_if_missing=True"):
        load_deepsearchqa(tmp_path / "missing.csv")


def test_prediction_jsonl_round_trip(tmp_path: Path) -> None:
    path = tmp_path / "runs" / "predictions.jsonl"
    expected = [
        Prediction(example_id="0", prediction="New Zealand"),
        Prediction(example_id="1", prediction="M\u00fcnchen"),
    ]

    save_predictions(path, expected)

    assert load_predictions(path) == expected
    assert path.read_text(encoding="utf-8").count("\n") == 2


def test_prediction_jsonl_rejects_duplicate_ids(tmp_path: Path) -> None:
    path = tmp_path / "predictions.jsonl"
    predictions = [
        Prediction(example_id="0", prediction="first"),
        Prediction(example_id="0", prediction="second"),
    ]

    with pytest.raises(PredictionValidationError, match="Duplicate"):
        save_predictions(path, predictions)


def test_build_judge_prompt_uses_official_fields() -> None:
    example = DeepSearchQAExample(
        example_id="7",
        problem="Which items?",
        problem_category="Other",
        answer="A, B",
        answer_type="Set Answer",
    )

    prompt = build_judge_prompt(example, "A and C")

    assert "Your task is to evaluate whether" in prompt
    assert "Prompt Type: Set Answer" in prompt
    assert "<prompt>\nWhich items?\n</prompt>" in prompt
    assert "<answer>\nA, B\n</answer>" in prompt
    assert "<response>\nA and C\n</response>" in prompt


def test_parse_judge_response_accepts_official_fenced_json() -> None:
    raw = """```json
{"Answer Correctness": {"Explanation": "partial", "Correctness Details": {"A": true, "B": false}}}
```"""

    parsed = parse_judge_response(raw)

    assert parsed.explanation == "partial"
    assert parsed.correctness_details == {"A": True, "B": False}
    assert parsed.excessive_answers == []


def test_parse_judge_response_rejects_non_boolean_details() -> None:
    raw = json.dumps(
        {
            "Answer Correctness": {
                "Explanation": "bad",
                "Correctness Details": {"A": "true"},
                "Excessive Answers": [],
            }
        }
    )

    with pytest.raises(JudgeResponseError, match="Correctness Details"):
        parse_judge_response(raw)


class ScriptedJudge:
    def judge(self, prompt: str) -> str:
        if "<response>\nperfect\n</response>" in prompt:
            return json.dumps(
                {
                    "Answer Correctness": {
                        "Explanation": "all found",
                        "Correctness Details": {"A": True, "B": True},
                        "Excessive Answers": [],
                    }
                }
            )
        if "<response>\npartial\n</response>" in prompt:
            return json.dumps(
                {
                    "Answer Correctness": {
                        "Explanation": "one found and one extra",
                        "Correctness Details": {"A": True, "B": False},
                        "Excessive Answers": ["C"],
                    }
                }
            )
        return "not JSON"


def test_evaluation_macro_averages_only_valid_examples() -> None:
    examples = [
        DeepSearchQAExample(str(index), f"Q{index}", "Other", "A, B", "Set Answer")
        for index in range(4)
    ]
    predictions = [
        Prediction("0", "perfect"),
        Prediction("1", "partial"),
        Prediction("3", "invalid"),
    ]

    result = evaluate_predictions(
        examples,
        predictions,
        ScriptedJudge(),
        max_workers=2,
    )

    assert result.summary.total_examples == 4
    assert result.summary.valid_examples == 2
    assert result.summary.empty_model_responses == 1
    assert result.summary.invalid_auto_rater_responses == 1
    assert result.summary.precision == pytest.approx(0.75)
    assert result.summary.recall == pytest.approx(0.75)
    assert result.summary.f1_score == pytest.approx(0.75)
    assert [grade.status for grade in result.grades] == [
        "valid",
        "valid",
        "empty_model_response",
        "invalid_auto_rater_response",
    ]


def test_missing_gemini_key_is_explicit(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("MISSING_GEMINI_KEY", raising=False)

    with pytest.raises(MissingAPIKeyError, match="MISSING_GEMINI_KEY"):
        GeminiJudge(api_key_env="MISSING_GEMINI_KEY")


def test_openai_compatible_judge_requests_json_without_thinking() -> None:
    request: dict[str, object] = {}

    class FakeCompletions:
        def create(self, **kwargs):
            request.update(kwargs)
            message = type("Message", (), {"content": '{"ok": true}'})()
            choice = type("Choice", (), {"message": message})()
            return type("Response", (), {"choices": [choice]})()

    judge = object.__new__(OpenAICompatibleJudge)
    judge._client = type(
        "Client",
        (),
        {"chat": type("Chat", (), {"completions": FakeCompletions()})()},
    )()
    judge._model = "mimo-v2.5"
    judge._max_completion_tokens = 8000
    judge._thinking = "disabled"

    assert judge.judge("grade") == '{"ok": true}'
    assert request["response_format"] == {"type": "json_object"}
    assert request["extra_body"] == {"thinking": {"type": "disabled"}}
