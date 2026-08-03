from __future__ import annotations

import csv
import hashlib
import io
import json
import os
import random
import statistics
import textwrap
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Protocol
from urllib.request import urlopen

from openai import OpenAI


DEEPSEARCHQA_DATASET_URL = (
    "https://www.kaggle.com/api/v1/datasets/download/deepmind/deepsearchqa/"
    "DSQA-full.csv?datasetVersionNumber=4"
)
DEEPSEARCHQA_DATASET_SHA256 = (
    "cc4394663f2fa9af042327d9c6d53767df1ed85c9aaef9ed11fe6458b6133368"
)
DEEPSEARCHQA_EXPECTED_ROWS = 900
DEEPSEARCHQA_JUDGE_MODEL = "gemini-2.5-flash"

_DATASET_COLUMNS = ("problem", "problem_category", "answer", "answer_type")
_KAGGLE_DATASET_COLUMNS = ("example_id", *_DATASET_COLUMNS)
_ANSWER_TYPES = {"Single Answer", "Set Answer"}


DEEPSEARCH_QA_PROMPT = textwrap.dedent("""\
Your task is to evaluate whether a given "AI Response" for a specific "User Prompt" arrived at the correct answer.

**Answer Correctness Task**

*   **Purpose:** Assess whether the AI response provides the correct answer(s) based on the provided "Correct Answer" and "Prompt Type".
*   **Process:**
    *   Identify the "Prompt Type": "<prompt_type>".
    *   Refer to the "Correct Answer": "<answer>".
    *   Based on the "Prompt Type", determine if the "AI Response" contains the expected answer(s).
        *   **'Single Answer'**: Check if the response provides the answer that addresses the user's question. It does not have to match the exact wording of the provided answer.
        *   **'Set Answer'**: Check if the response includes *each* item from the provided ground truth answers. The order might not matter unless specified otherwise. The response might include more answers than the list. Determine the correctness *only* based on the list first and then check if the response includes answers not in the list.
    *   **Explanation:** Provide a brief explanation justifying your assessment of answer correctness, referencing specific parts of the AI response and the correct answer.
    *   **Correctness Details:** Provide a dictionary, one key for each expected answer part, and value is a boolean indicating whether each expected answer part was found.
        *   For 'Set Answer', this will be a list of attributes, one for each item/part in the "Correct Answer". Each key will be a string indicating the expected answer part, and the value will be a boolean indicating whether that part was found in the response.
    *   **Excessive Answers:** Provide a list of strings, each indicating an excessive answer part. If the response provides answers that are **not** in the "Correct Answer" list, add these answers as excessive answers. Return an empty list when there's no excessive answers in the response.


**Output Format:**

Your evaluation *must* be structured as a nested JSON dictionary with the following top-level keys: `"Answer Correctness"`. Please return NULL if any of "Prompt", "AI Response" or "Correct Answer" is empty.
The value for `"Answer Correctness"` should be a dictionary containing `"Explanation"` (a string), `"Correctness Details"` (a dictionary where each key is the expected correct answer, and the value is a boolean indicating whether the response contains the correct answer), and `"Excessive Answers"` (a list of strings indicating the excessive answers).

Make sure you return a valid JSON string. Pay special attention to quotes, commas and special characters in the JSON string. Make sure to escape all special characters and quotes in the JSON string.


""")

GRADER_RATING_OUTPUT_EXAMPLE = r"""**Example (Partial):**

"```json
{{
  "Answer Correctness": {{
    "Explanation": "The response correctly identified Belgium and France but also includes an excessive answer, Italy.",
    "Correctness Details": {{
      "Belgium": true,
      "France": true,
    }},
    "Excessive Answers": [ "Italy" ]
  }}
}}
```"

**Now, proceed with the evaluation using the provided User Prompt, AI Response, and Correct Answer.**

User Prompt (Wrapped in <prompt> and </prompt>):
<prompt>
{prompt}
</prompt>
--------------------
**  Correct Answer (Wrapped in <answer> and </answer>):
Prompt Type: {prompt_type}
<answer>
{answer}
</answer>
--------------------
AI assistant response (Wrapped in <response> and </response>):
<response>
{response}
</response>

--------------------
Rating:"""


class DeepSearchQAError(RuntimeError):
    """Base error for DeepSearchQA data and evaluation failures."""


class DatasetValidationError(DeepSearchQAError):
    pass


class PredictionValidationError(DeepSearchQAError):
    pass


class JudgeResponseError(DeepSearchQAError):
    pass


class MissingAPIKeyError(DeepSearchQAError):
    pass


@dataclass(frozen=True)
class DeepSearchQAExample:
    example_id: str
    problem: str
    problem_category: str
    answer: str
    answer_type: str


@dataclass(frozen=True)
class Prediction:
    example_id: str
    prediction: str

    def to_dict(self) -> dict[str, str]:
        return asdict(self)


@dataclass(frozen=True)
class ParsedJudgeResponse:
    explanation: str
    correctness_details: dict[str, bool]
    excessive_answers: list[str]


@dataclass(frozen=True)
class ExampleGrade:
    example_id: str
    status: str
    precision: float | None = None
    recall: float | None = None
    f1_score: float | None = None
    explanation: str | None = None
    correctness_details: dict[str, bool] | None = None
    excessive_answers: list[str] | None = None
    raw_judge_response: str = ""
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class EvaluationSummary:
    total_examples: int
    valid_examples: int
    empty_model_responses: int
    empty_auto_rater_responses: int
    invalid_auto_rater_responses: int
    judge_errors: int
    precision: float | None
    recall: float | None
    f1_score: float | None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class EvaluationResult:
    grades: tuple[ExampleGrade, ...]
    summary: EvaluationSummary

    def to_dict(self) -> dict[str, Any]:
        return {
            "summary": self.summary.to_dict(),
            "grades": [grade.to_dict() for grade in self.grades],
        }


class Judge(Protocol):
    def judge(self, prompt: str) -> str: ...


class OpenAICompatibleJudge:
    """JSON judge for MiMo during iteration and other Chat Completions APIs."""

    def __init__(
        self,
        api_key: str,
        *,
        model: str,
        base_url: str | None = None,
        max_retries: int = 5,
        max_completion_tokens: int = 8_000,
        thinking: str | None = None,
        timeout_seconds: float = 120.0,
    ) -> None:
        self._client = OpenAI(
            api_key=api_key,
            base_url=base_url,
            max_retries=max_retries,
            timeout=timeout_seconds,
        )
        self._model = model
        self._max_completion_tokens = max_completion_tokens
        self._thinking = thinking

    def judge(self, prompt: str) -> str:
        request: dict[str, Any] = {
            "model": self._model,
            "messages": [{"role": "user", "content": prompt}],
            "max_completion_tokens": self._max_completion_tokens,
            "response_format": {"type": "json_object"},
        }
        if self._thinking:
            request["extra_body"] = {"thinking": {"type": self._thinking}}
        response = self._client.chat.completions.create(**request)
        return response.choices[0].message.content or ""


class GeminiJudge:
    """Gemini autorater configured like the official DeepSearchQA starter."""

    def __init__(
        self,
        api_key: str | None = None,
        *,
        api_key_env: str = "GOOGLE_API_KEY",
        model: str = DEEPSEARCHQA_JUDGE_MODEL,
        max_retries: int = 5,
    ) -> None:
        resolved_key = api_key or os.getenv(api_key_env)
        if not resolved_key:
            raise MissingAPIKeyError(
                f"Missing Gemini API key: pass api_key or set {api_key_env}"
            )
        if max_retries < 1:
            raise ValueError("max_retries must be at least 1")

        try:
            from google import genai
        except ImportError as exc:  # pragma: no cover - packaging failure
            raise DeepSearchQAError(
                "Gemini judging requires the google-genai package"
            ) from exc

        self._client = genai.Client(api_key=resolved_key)
        self._model = model
        self._max_retries = max_retries

    def judge(self, prompt: str) -> str:
        for attempt in range(self._max_retries):
            try:
                response = self._client.models.generate_content(
                    model=self._model,
                    contents=prompt,
                )
                return response.text or ""
            except Exception as exc:
                if attempt == self._max_retries - 1:
                    raise DeepSearchQAError(
                        f"Gemini judge failed after {self._max_retries} attempts: {exc}"
                    ) from exc
                time.sleep(1 + (2 ** (attempt + random.random())))
        raise AssertionError("unreachable")


def download_deepsearchqa(
    destination: str | Path,
    *,
    url: str = DEEPSEARCHQA_DATASET_URL,
    expected_sha256: str | None = DEEPSEARCHQA_DATASET_SHA256,
) -> Path:
    """Download and validate the public Kaggle v4 CSV without Kaggle credentials."""
    destination_path = Path(destination)
    try:
        with urlopen(url, timeout=60) as response:
            data = response.read()
    except Exception as exc:
        raise DeepSearchQAError(f"Unable to download DeepSearchQA from {url}: {exc}") from exc

    if expected_sha256 is not None:
        actual_sha256 = hashlib.sha256(data).hexdigest()
        if actual_sha256 != expected_sha256:
            raise DatasetValidationError(
                "DeepSearchQA checksum mismatch: "
                f"expected {expected_sha256}, got {actual_sha256}"
            )

    _parse_dataset_bytes(data, source=url)
    destination_path.parent.mkdir(parents=True, exist_ok=True)
    destination_path.write_bytes(data)
    return destination_path


def load_deepsearchqa(
    path: str | Path,
    *,
    download_if_missing: bool = False,
    verify_checksum: bool = False,
) -> list[DeepSearchQAExample]:
    """Load a complete official dataset, downloading Kaggle v4 when requested."""
    dataset_path = Path(path)
    if not dataset_path.exists():
        if not download_if_missing:
            raise FileNotFoundError(
                f"DeepSearchQA dataset not found: {dataset_path}. "
                "Pass download_if_missing=True to fetch the public Kaggle v4 CSV."
            )
        download_deepsearchqa(dataset_path)

    data = dataset_path.read_bytes()
    if verify_checksum:
        actual_sha256 = hashlib.sha256(data).hexdigest()
        if actual_sha256 != DEEPSEARCHQA_DATASET_SHA256:
            raise DatasetValidationError(
                "DeepSearchQA checksum mismatch: "
                f"expected {DEEPSEARCHQA_DATASET_SHA256}, got {actual_sha256}"
            )
    return _parse_dataset_bytes(data, source=str(dataset_path))


def save_predictions(path: str | Path, predictions: Iterable[Prediction]) -> Path:
    """Write one stable, UTF-8 JSON object per prediction."""
    records = list(predictions)
    _index_predictions(records)
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    content = "".join(
        json.dumps(record.to_dict(), ensure_ascii=False) + "\n" for record in records
    )
    output_path.write_text(content, encoding="utf-8")
    return output_path


def load_predictions(path: str | Path) -> list[Prediction]:
    predictions: list[Prediction] = []
    with Path(path).open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                raw = json.loads(line)
            except json.JSONDecodeError as exc:
                raise PredictionValidationError(
                    f"Invalid prediction JSON on line {line_number}: {exc}"
                ) from exc
            if not isinstance(raw, dict):
                raise PredictionValidationError(
                    f"Prediction on line {line_number} must be a JSON object"
                )
            example_id = raw.get("example_id")
            prediction = raw.get("prediction")
            if not isinstance(example_id, str) or not isinstance(prediction, str):
                raise PredictionValidationError(
                    f"Prediction on line {line_number} requires string "
                    "example_id and prediction fields"
                )
            predictions.append(Prediction(example_id=example_id, prediction=prediction))
    _index_predictions(predictions)
    return predictions


def build_judge_prompt(example: DeepSearchQAExample, prediction: str) -> str:
    return DEEPSEARCH_QA_PROMPT + GRADER_RATING_OUTPUT_EXAMPLE.format(
        prompt=example.problem.strip(),
        prompt_type=example.answer_type.strip(),
        answer=example.answer.strip(),
        response=prediction.strip(),
    )


def parse_judge_response(response: str) -> ParsedJudgeResponse:
    """Parse exactly the JSON shape accepted by the official starter."""
    json_text = response.strip()
    start_marker = "```json"
    start_index = json_text.find(start_marker)
    if start_index != -1:
        json_text = json_text[start_index + len(start_marker) :].strip()
        end_index = json_text.rfind("```")
        if end_index != -1:
            json_text = json_text[:end_index].strip()

    try:
        parsed = json.loads(json_text)
    except json.JSONDecodeError as exc:
        raise JudgeResponseError(f"Invalid JSON response from autorater: {exc}") from exc

    if not isinstance(parsed, dict):
        raise JudgeResponseError("Autorater response must be a JSON object")
    answer_correctness = parsed.get("Answer Correctness")
    if not isinstance(answer_correctness, dict):
        raise JudgeResponseError("Missing or malformed 'Answer Correctness' node")

    explanation = answer_correctness.get("Explanation")
    if not isinstance(explanation, str):
        raise JudgeResponseError("Missing or malformed 'Explanation'")

    details = answer_correctness.get("Correctness Details")
    if not isinstance(details, dict) or not all(
        isinstance(key, str) and isinstance(value, bool)
        for key, value in details.items()
    ):
        raise JudgeResponseError("Invalid 'Correctness Details'")

    excessive_answers = answer_correctness.get("Excessive Answers", [])
    if not isinstance(excessive_answers, list) or not all(
        isinstance(item, str) for item in excessive_answers
    ):
        raise JudgeResponseError("Invalid 'Excessive Answers'")

    return ParsedJudgeResponse(
        explanation=explanation,
        correctness_details=dict(details),
        excessive_answers=list(excessive_answers),
    )


def evaluate_predictions(
    examples: Iterable[DeepSearchQAExample],
    predictions: Iterable[Prediction],
    judge: Judge,
    *,
    max_workers: int = 5,
    on_grade: Callable[[ExampleGrade], None] | None = None,
) -> EvaluationResult:
    """Judge predictions and macro-average per-example precision, recall, and F1."""
    example_list = list(examples)
    if not example_list:
        raise ValueError("At least one DeepSearchQA example is required")
    if max_workers < 1:
        raise ValueError("max_workers must be at least 1")
    example_ids = [example.example_id for example in example_list]
    if len(example_ids) != len(set(example_ids)):
        raise DatasetValidationError("DeepSearchQA example_id values must be unique")
    prediction_index = _index_predictions(predictions)
    unknown_ids = sorted(set(prediction_index) - set(example_ids))
    if unknown_ids:
        raise PredictionValidationError(
            f"Predictions contain unknown example_id values: {unknown_ids[:5]}"
        )

    def evaluate_one(example: DeepSearchQAExample) -> ExampleGrade:
        prediction = prediction_index.get(example.example_id, "").strip()
        if not prediction:
            return ExampleGrade(
                example_id=example.example_id,
                status="empty_model_response",
                error="AI response was empty.",
            )

        prompt = build_judge_prompt(example, prediction)
        try:
            raw_response = judge.judge(prompt)
        except Exception as exc:
            return ExampleGrade(
                example_id=example.example_id,
                status="judge_error",
                error=f"{type(exc).__name__}: {exc}",
            )
        if not raw_response:
            return ExampleGrade(
                example_id=example.example_id,
                status="empty_auto_rater_response",
                error="Auto-rater response was empty.",
            )

        try:
            parsed = parse_judge_response(raw_response)
        except JudgeResponseError as exc:
            return ExampleGrade(
                example_id=example.example_id,
                status="invalid_auto_rater_response",
                raw_judge_response=raw_response,
                error=str(exc),
            )

        correct = sum(parsed.correctness_details.values())
        expected = len(parsed.correctness_details)
        excessive = len(parsed.excessive_answers)
        precision, recall, f1_score = _calculate_metrics(
            true_positives=correct,
            false_positives=excessive,
            false_negatives=expected - correct,
        )
        return ExampleGrade(
            example_id=example.example_id,
            status="valid",
            precision=precision,
            recall=recall,
            f1_score=f1_score,
            explanation=parsed.explanation,
            correctness_details=parsed.correctness_details,
            excessive_answers=parsed.excessive_answers,
            raw_judge_response=raw_response,
        )

    grades_by_id: dict[str, ExampleGrade] = {}
    with ThreadPoolExecutor(max_workers=min(max_workers, len(example_list))) as executor:
        futures = {
            executor.submit(evaluate_one, example): example.example_id
            for example in example_list
        }
        for future in as_completed(futures):
            grade = future.result()
            grades_by_id[grade.example_id] = grade
            if on_grade is not None:
                on_grade(grade)
    grades = tuple(grades_by_id[example_id] for example_id in example_ids)
    summary = summarize_grades(grades)
    return EvaluationResult(grades=grades, summary=summary)


def summarize_grades(
    grades: Iterable[ExampleGrade],
) -> EvaluationSummary:
    grade_list = tuple(grades)
    if not grade_list:
        raise ValueError("At least one DeepSearchQA grade is required")
    valid_grades = [grade for grade in grade_list if grade.status == "valid"]
    precision = _mean_metric(valid_grades, "precision")
    recall = _mean_metric(valid_grades, "recall")
    f1_score = _mean_metric(valid_grades, "f1_score")
    summary = EvaluationSummary(
        total_examples=len(grade_list),
        valid_examples=len(valid_grades),
        empty_model_responses=_count_status(grade_list, "empty_model_response"),
        empty_auto_rater_responses=_count_status(
            grade_list, "empty_auto_rater_response"
        ),
        invalid_auto_rater_responses=_count_status(
            grade_list, "invalid_auto_rater_response"
        ),
        judge_errors=_count_status(grade_list, "judge_error"),
        precision=precision,
        recall=recall,
        f1_score=f1_score,
    )
    return summary


def _parse_dataset_bytes(data: bytes, *, source: str) -> list[DeepSearchQAExample]:
    try:
        text = data.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise DatasetValidationError(f"DeepSearchQA CSV is not UTF-8: {source}") from exc

    reader = csv.DictReader(io.StringIO(text, newline=""))
    fieldnames = tuple(reader.fieldnames or ())
    if fieldnames not in (_DATASET_COLUMNS, _KAGGLE_DATASET_COLUMNS):
        raise DatasetValidationError(
            f"Unexpected DeepSearchQA columns in {source}: {list(fieldnames)}"
        )

    examples: list[DeepSearchQAExample] = []
    for index, row in enumerate(reader):
        values: dict[str, str] = {}
        for column in _DATASET_COLUMNS:
            value = row.get(column)
            if not isinstance(value, str) or not value.strip():
                raise DatasetValidationError(
                    f"Empty {column!r} in DeepSearchQA row {index} from {source}"
                )
            values[column] = value.strip()

        if values["answer_type"] not in _ANSWER_TYPES:
            raise DatasetValidationError(
                f"Invalid answer_type in DeepSearchQA row {index}: "
                f"{values['answer_type']!r}"
            )
        example_id = row.get("example_id", str(index))
        if not isinstance(example_id, str) or not example_id.strip():
            raise DatasetValidationError(
                f"Empty 'example_id' in DeepSearchQA row {index} from {source}"
            )
        examples.append(
            DeepSearchQAExample(
                example_id=example_id.strip(),
                problem=values["problem"],
                problem_category=values["problem_category"],
                answer=values["answer"],
                answer_type=values["answer_type"],
            )
        )

    if len(examples) != DEEPSEARCHQA_EXPECTED_ROWS:
        raise DatasetValidationError(
            f"DeepSearchQA must contain {DEEPSEARCHQA_EXPECTED_ROWS} rows; "
            f"found {len(examples)} in {source}"
        )
    ids = [example.example_id for example in examples]
    if len(ids) != len(set(ids)):
        raise DatasetValidationError(f"Duplicate DeepSearchQA example_id in {source}")
    return examples


def _index_predictions(predictions: Iterable[Prediction]) -> dict[str, str]:
    result: dict[str, str] = {}
    for prediction in predictions:
        if not isinstance(prediction, Prediction):
            raise PredictionValidationError("Predictions must be Prediction objects")
        if not prediction.example_id:
            raise PredictionValidationError("Prediction example_id must not be empty")
        if not isinstance(prediction.prediction, str):
            raise PredictionValidationError("Prediction text must be a string")
        if prediction.example_id in result:
            raise PredictionValidationError(
                f"Duplicate prediction example_id: {prediction.example_id}"
            )
        result[prediction.example_id] = prediction.prediction
    return result


def _calculate_metrics(
    *, true_positives: int, false_positives: int, false_negatives: int
) -> tuple[float, float, float]:
    precision = (
        true_positives / (true_positives + false_positives)
        if true_positives + false_positives
        else 0.0
    )
    recall = (
        true_positives / (true_positives + false_negatives)
        if true_positives + false_negatives
        else 0.0
    )
    f1_score = (
        2 * precision * recall / (precision + recall)
        if precision + recall
        else 0.0
    )
    return precision, recall, f1_score


def _mean_metric(grades: list[ExampleGrade], name: str) -> float | None:
    values = [getattr(grade, name) for grade in grades]
    return statistics.fmean(value for value in values if value is not None) if values else None


def _count_status(grades: tuple[ExampleGrade, ...], status: str) -> int:
    return sum(grade.status == status for grade in grades)
