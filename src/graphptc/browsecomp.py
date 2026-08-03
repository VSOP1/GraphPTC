from __future__ import annotations

import base64
import csv
import hashlib
import io
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Protocol
from urllib.request import urlopen

from openai import OpenAI


BROWSECOMP_DATASET_URL = (
    "https://openaipublic.blob.core.windows.net/simple-evals/"
    "browse_comp_test_set.csv"
)
BROWSECOMP_DATASET_SHA256 = (
    "7b24471cd5b3eb2a46830a14802b5c029ea62f488ff75a0f88af7923d1454abf"
)
BROWSECOMP_EXPECTED_ROWS = 1_266
_DATASET_COLUMNS = ("problem", "answer", "problem_topic", "canary")


BROWSECOMP_GRADER_PROMPT = """I'm going to show you a question, a ground-truth answer, and a sample answer. Tell me whether the sample answer is correct relative to the ground truth. The wording need not match exactly.
<question>{question}</question>
<sample_answer>{sample_answer}</sample_answer>
<ground_truth>{ground_truth}</ground_truth>

Consider these statements about the sample answer:
 (A) It matches the ground-truth answer.
 (B) It does not match the ground-truth answer.
 (C) It says something like "I'm not sure" or "I don't know".

Respond with exactly one letter (A, B, or C) and nothing else."""

_GRADER_LETTER_RE = re.compile(r"\s*([ABC])\s*", re.IGNORECASE)


class BrowseCompError(RuntimeError):
    """Base error for BrowseComp data and evaluation failures."""


class BrowseCompDatasetError(BrowseCompError):
    pass


@dataclass(frozen=True)
class BrowseCompExample:
    example_id: str
    problem: str
    answer: str
    problem_topic: str


@dataclass(frozen=True)
class BrowseCompPrediction:
    example_id: str
    prediction: str


@dataclass(frozen=True)
class BrowseCompGrade:
    example_id: str
    status: str
    grader_letter: str | None = None
    accuracy: float = 0.0
    raw_judge_response: str = ""
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class BrowseCompEvaluationSummary:
    total_examples: int
    valid_examples: int
    correct: int
    incorrect: int
    abstained: int
    empty_model_responses: int
    invalid_auto_rater_responses: int
    judge_errors: int
    accuracy: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class BrowseCompEvaluationResult:
    grades: tuple[BrowseCompGrade, ...]
    summary: BrowseCompEvaluationSummary


class BrowseCompJudge(Protocol):
    def judge(self, prompt: str) -> str: ...


class OpenAICompatibleBrowseCompJudge:
    """Single-letter BrowseComp judge for MiMo and compatible APIs."""

    def __init__(
        self,
        api_key: str,
        *,
        model: str,
        base_url: str | None = None,
        max_retries: int = 2,
        max_completion_tokens: int = 8,
        thinking: str | None = "disabled",
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
            "max_completion_tokens": self._max_completion_tokens,
            "messages": [{"role": "user", "content": prompt}],
        }
        if self._thinking:
            request["extra_body"] = {"thinking": {"type": self._thinking}}
        response = self._client.chat.completions.create(**request)
        return response.choices[0].message.content or ""


def derive_key(password: str, length: int) -> bytes:
    digest = hashlib.sha256(password.encode()).digest()
    return digest * (length // len(digest)) + digest[: length % len(digest)]


def decrypt(ciphertext_b64: str, password: str) -> str:
    encrypted = base64.b64decode(ciphertext_b64, validate=True)
    key = derive_key(password, len(encrypted))
    return bytes(a ^ b for a, b in zip(encrypted, key, strict=True)).decode()


def download_browsecomp(path: str | Path) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with urlopen(BROWSECOMP_DATASET_URL, timeout=120) as response:
        data = response.read()
    _verify_checksum(data)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_bytes(data)
    temporary.replace(destination)
    return destination


def load_browsecomp(
    path: str | Path,
    *,
    download_if_missing: bool = False,
    verify_checksum: bool = True,
) -> list[BrowseCompExample]:
    dataset_path = Path(path)
    if not dataset_path.exists():
        if not download_if_missing:
            raise FileNotFoundError(dataset_path)
        download_browsecomp(dataset_path)

    data = dataset_path.read_bytes()
    if verify_checksum:
        _verify_checksum(data)
    try:
        text = data.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise BrowseCompDatasetError("BrowseComp CSV must be UTF-8") from exc

    reader = csv.DictReader(io.StringIO(text))
    if tuple(reader.fieldnames or ()) != _DATASET_COLUMNS:
        raise BrowseCompDatasetError(
            f"Unexpected BrowseComp columns: {reader.fieldnames}"
        )

    examples: list[BrowseCompExample] = []
    for index, row in enumerate(reader):
        try:
            problem = decrypt(row["problem"], row["canary"])
            answer = decrypt(row["answer"], row["canary"])
        except Exception as exc:
            raise BrowseCompDatasetError(
                f"Could not decrypt BrowseComp row {index}"
            ) from exc
        if not problem.strip() or not answer.strip():
            raise BrowseCompDatasetError(f"BrowseComp row {index} is empty")
        examples.append(
            BrowseCompExample(
                example_id=str(index),
                problem=problem.strip(),
                answer=answer.strip(),
                problem_topic=row["problem_topic"].strip(),
            )
        )

    if verify_checksum and len(examples) != BROWSECOMP_EXPECTED_ROWS:
        raise BrowseCompDatasetError(
            f"Expected {BROWSECOMP_EXPECTED_ROWS} BrowseComp rows, got {len(examples)}"
        )
    return examples


def build_browsecomp_grader_prompt(
    example: BrowseCompExample, prediction: str
) -> str:
    return BROWSECOMP_GRADER_PROMPT.format(
        question=example.problem,
        sample_answer=prediction.strip(),
        ground_truth=example.answer,
    )


def parse_browsecomp_grader_letter(response: str) -> str | None:
    match = _GRADER_LETTER_RE.fullmatch(response)
    return match.group(1).upper() if match else None


def evaluate_browsecomp_predictions(
    examples: Iterable[BrowseCompExample],
    predictions: Iterable[BrowseCompPrediction],
    judge: BrowseCompJudge,
    *,
    max_workers: int = 5,
    on_grade: Callable[[BrowseCompGrade], None] | None = None,
) -> BrowseCompEvaluationResult:
    example_list = list(examples)
    if not example_list:
        raise ValueError("At least one BrowseComp example is required")
    prediction_index = _prediction_index(predictions)
    example_ids = {example.example_id for example in example_list}
    unknown_ids = sorted(set(prediction_index) - example_ids)
    if unknown_ids:
        raise ValueError(f"Unknown BrowseComp prediction IDs: {unknown_ids[:5]}")

    def evaluate_one(example: BrowseCompExample) -> BrowseCompGrade:
        prediction = prediction_index.get(example.example_id, "").strip()
        if not prediction:
            return BrowseCompGrade(
                example_id=example.example_id,
                status="empty_model_response",
                error="Model response was empty.",
            )
        try:
            raw_response = judge.judge(
                build_browsecomp_grader_prompt(example, prediction)
            )
        except Exception as exc:
            return BrowseCompGrade(
                example_id=example.example_id,
                status="judge_error",
                error=f"{type(exc).__name__}: {exc}",
            )
        letter = parse_browsecomp_grader_letter(raw_response)
        if letter is None:
            return BrowseCompGrade(
                example_id=example.example_id,
                status="invalid_auto_rater_response",
                raw_judge_response=raw_response,
                error="BrowseComp judge did not return A, B, or C.",
            )
        return BrowseCompGrade(
            example_id=example.example_id,
            status="valid",
            grader_letter=letter,
            accuracy=1.0 if letter == "A" else 0.0,
            raw_judge_response=raw_response,
        )

    grades_by_id: dict[str, BrowseCompGrade] = {}
    workers = min(max_workers, len(example_list))
    if workers < 1:
        raise ValueError("max_workers must be at least 1")
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(evaluate_one, example): example for example in example_list}
        for future in as_completed(futures):
            grade = future.result()
            grades_by_id[grade.example_id] = grade
            if on_grade is not None:
                on_grade(grade)

    grades = tuple(grades_by_id[example.example_id] for example in example_list)
    return BrowseCompEvaluationResult(grades, summarize_browsecomp_grades(grades))


def summarize_browsecomp_grades(
    grades: Iterable[BrowseCompGrade],
) -> BrowseCompEvaluationSummary:
    grade_list = tuple(grades)
    if not grade_list:
        raise ValueError("At least one BrowseComp grade is required")
    correct = sum(grade.grader_letter == "A" for grade in grade_list)
    incorrect = sum(grade.grader_letter == "B" for grade in grade_list)
    abstained = sum(grade.grader_letter == "C" for grade in grade_list)
    total = len(grade_list)
    return BrowseCompEvaluationSummary(
        total_examples=total,
        valid_examples=sum(grade.status == "valid" for grade in grade_list),
        correct=correct,
        incorrect=incorrect,
        abstained=abstained,
        empty_model_responses=sum(
            grade.status == "empty_model_response" for grade in grade_list
        ),
        invalid_auto_rater_responses=sum(
            grade.status == "invalid_auto_rater_response" for grade in grade_list
        ),
        judge_errors=sum(grade.status == "judge_error" for grade in grade_list),
        accuracy=correct / total,
    )


def _verify_checksum(data: bytes) -> None:
    actual = hashlib.sha256(data).hexdigest()
    if actual != BROWSECOMP_DATASET_SHA256:
        raise BrowseCompDatasetError(
            f"BrowseComp SHA256 mismatch: expected {BROWSECOMP_DATASET_SHA256}, got {actual}"
        )


def _prediction_index(
    predictions: Iterable[BrowseCompPrediction],
) -> dict[str, str]:
    result: dict[str, str] = {}
    for prediction in predictions:
        if prediction.example_id in result:
            raise ValueError(f"Duplicate prediction ID: {prediction.example_id}")
        result[prediction.example_id] = prediction.prediction
    return result
