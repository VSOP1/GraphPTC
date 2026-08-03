from __future__ import annotations

import hashlib
import json
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator, Protocol
from urllib.request import Request, urlopen

from openai import OpenAI

from .browsecomp import download_browsecomp, load_browsecomp
from .config import BrowseCompPlusConfig
from .local_search import build_sqlite_fts_index, index_document_count


BROWSECOMP_PLUS_EXPECTED_EXAMPLES = 830
BROWSECOMP_PLUS_EXPECTED_DOCUMENTS = 100_195
BROWSECOMP_PLUS_CORPUS_REVISION = "b27b02bc3e45511b8b82a13e6f90ce761df726f6"
BROWSECOMP_PLUS_UPSTREAM_REVISION = "046949032b0328319cc9a02663a759ec601d9402"

_CORPUS_SHA256 = (
    "7c07f9e23b1ca548110fd831714cadc67d44db5223bace6e45fcaa795d3153d0",
    "e92d8202e0f656a85b262153dbcd22ecf80ea2d0c96d9884f9c8e25480b869ab",
    "0e4113a4503342527258d8f2c49877747435f3e65bfe1f7306b4f488c8d225fe",
    "0ceea5e703332a2e3ce700f641273400d84583fad84b659d3248ed06d3a9fef3",
    "15b62914ddc3de6946893c770f07d5d84d29646e833ca1447955668f2b57940c",
    "a9a75708ad37c522e93a774e5a968a3129e12b0559971c8f950a5628e0201df0",
    "290062b60c1a6ebba7d5469a37a431f0a2596e68788295284b1b2d35db07b62c",
)
_QREL_SHA256 = {
    "qrel_golds.txt": "b875af4a745712bee7a94f464ed989232f8c77977c31824428470e11dcb28c73",
    "qrel_evidence.txt": "a6f594975be57339de9e4e9f67f13c044f647feda77c0b84c45a1581e3041bd1",
}

BROWSECOMP_PLUS_GRADER_PROMPT = """Judge whether the following [response] to [question] is correct or not based on the precise and unambiguous [correct_answer] below.

[question]: {question}

[response]: {response}

Your judgement must be in the format and criteria specified below:

extracted_final_answer: The final exact answer extracted from the [response]. Put the extracted answer as 'None' if there is no exact, final answer to extract from the response.

[correct_answer]: {correct_answer}

reasoning: Explain why the extracted_final_answer is correct or incorrect based on [correct_answer], focusing only on if there are meaningful differences between [correct_answer] and the extracted_final_answer. Do not comment on any background to the problem, do not attempt to solve the problem, do not argue for any answer different than [correct_answer], focus only on whether the answers match.

correct: Answer 'yes' if extracted_final_answer matches the [correct_answer] given above, or is within a small margin of error for numerical problems. Answer 'no' otherwise, i.e. if there is any inconsistency, ambiguity, non-equivalency, or if the extracted answer is incorrect.

confidence: The extracted confidence score between 0% and 100% from [response]. Put 100 if there is no confidence score available."""

_CORRECT_RE = re.compile(r"(?im)^correct\s*:\s*(yes|no)\s*$")
_CONFIDENCE_RE = re.compile(r"(?im)^confidence\s*:\s*(\d+(?:\.\d+)?)\s*%?\s*$")


class BrowseCompPlusError(RuntimeError):
    pass


@dataclass(frozen=True)
class BrowseCompPlusExample:
    example_id: str
    question: str
    answer: str


@dataclass(frozen=True)
class BrowseCompPlusGrade:
    example_id: str
    status: str
    correct: bool | None = None
    confidence: float | None = None
    accuracy: float = 0.0
    raw_judge_response: str = ""
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class BrowseCompPlusEvaluationSummary:
    total_examples: int
    valid_examples: int
    correct: int
    incorrect: int
    empty_model_responses: int
    invalid_auto_rater_responses: int
    judge_errors: int
    accuracy: float
    candidate_retrieval_recall: float | None = None
    fetched_evidence_recall: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class BrowseCompPlusEvaluationResult:
    grades: tuple[BrowseCompPlusGrade, ...]
    summary: BrowseCompPlusEvaluationSummary


class BrowseCompPlusJudge(Protocol):
    def judge(self, prompt: str) -> str: ...


class OpenAICompatibleBrowseCompPlusJudge:
    def __init__(
        self,
        api_key: str,
        *,
        model: str,
        base_url: str | None = None,
        max_retries: int = 2,
        max_completion_tokens: int = 1_024,
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


def download_browsecomp_plus(
    config: BrowseCompPlusConfig,
    dataset_path: str | Path,
    browsecomp_path: str | Path,
    *,
    progress: Callable[[str], None] | None = None,
) -> Path:
    source_path = Path(browsecomp_path)
    if not source_path.is_file():
        download_browsecomp(source_path)

    for name, destination in (
        ("qrel_golds.txt", config.qrels_gold_path),
        ("qrel_evidence.txt", config.qrels_evidence_path),
    ):
        url = (
            "https://raw.githubusercontent.com/texttron/BrowseComp-Plus/"
            f"{BROWSECOMP_PLUS_UPSTREAM_REVISION}/topics-qrels/{name}"
        )
        _download_verified(url, destination, _QREL_SHA256[name], progress=progress)

    qrel_ids = sorted(
        load_qrels(config.qrels_evidence_path), key=lambda value: int(value)
    )
    if len(qrel_ids) != BROWSECOMP_PLUS_EXPECTED_EXAMPLES:
        raise BrowseCompPlusError(
            f"Expected {BROWSECOMP_PLUS_EXPECTED_EXAMPLES} query IDs, got {len(qrel_ids)}"
        )
    source = load_browsecomp(source_path)
    examples = [
        BrowseCompPlusExample(
            example_id=query_id,
            question=source[int(query_id) - 1].problem,
            answer=source[int(query_id) - 1].answer,
        )
        for query_id in qrel_ids
    ]
    _write_examples(Path(dataset_path), examples)

    config.corpus_dir.mkdir(parents=True, exist_ok=True)
    for index, expected_hash in enumerate(_CORPUS_SHA256):
        name = f"train-{index:05d}-of-00007.parquet"
        url = (
            "https://huggingface.co/datasets/Tevatron/browsecomp-plus-corpus/resolve/"
            f"{BROWSECOMP_PLUS_CORPUS_REVISION}/data/{name}?download=true"
        )
        _download_verified(
            url, config.corpus_dir / name, expected_hash, progress=progress
        )

    count = 0
    if config.index_path.is_file():
        try:
            count = index_document_count(config.index_path)
        except (OSError, ValueError):
            count = 0
    if count != BROWSECOMP_PLUS_EXPECTED_DOCUMENTS:
        if progress is not None:
            progress("building SQLite FTS5 index")
        count = build_sqlite_fts_index(
            config.index_path, _iter_parquet_documents(config.corpus_dir)
        )
    elif progress is not None:
        progress("verified existing SQLite FTS5 index")
    if count != BROWSECOMP_PLUS_EXPECTED_DOCUMENTS:
        raise BrowseCompPlusError(
            f"Expected {BROWSECOMP_PLUS_EXPECTED_DOCUMENTS} documents, got {count}"
        )
    return Path(dataset_path)


def load_browsecomp_plus(
    path: str | Path,
    *,
    expected_examples: int | None = None,
) -> list[BrowseCompPlusExample]:
    expected_count = (
        BROWSECOMP_PLUS_EXPECTED_EXAMPLES
        if expected_examples is None
        else expected_examples
    )
    dataset_path = Path(path)
    if not dataset_path.is_file():
        raise FileNotFoundError(dataset_path)
    examples: list[BrowseCompPlusExample] = []
    seen: set[str] = set()
    with dataset_path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            try:
                row = json.loads(line)
                example = BrowseCompPlusExample(
                    example_id=str(row["query_id"]),
                    question=str(row["query"]).strip(),
                    answer=str(row["answer"]).strip(),
                )
            except (json.JSONDecodeError, KeyError, TypeError) as exc:
                raise BrowseCompPlusError(
                    f"Invalid BrowseComp-Plus JSONL line {line_number}"
                ) from exc
            if not example.question or not example.answer:
                raise BrowseCompPlusError(
                    f"Empty BrowseComp-Plus example on line {line_number}"
                )
            if example.example_id in seen:
                raise BrowseCompPlusError(
                    f"Duplicate BrowseComp-Plus query ID: {example.example_id}"
                )
            seen.add(example.example_id)
            examples.append(example)
    if len(examples) != expected_count:
        raise BrowseCompPlusError(
            f"Expected {expected_count} examples, got {len(examples)}"
        )
    return examples


def load_qrels(path: str | Path) -> dict[str, set[str]]:
    result: dict[str, set[str]] = {}
    with Path(path).open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            fields = line.split()
            if len(fields) != 4:
                raise BrowseCompPlusError(f"Invalid qrel line {line_number}: {path}")
            query_id, _, docid, relevance = fields
            if int(relevance) > 0:
                result.setdefault(query_id, set()).add(docid)
    return result


def build_browsecomp_plus_grader_prompt(
    example: BrowseCompPlusExample, prediction: str
) -> str:
    return BROWSECOMP_PLUS_GRADER_PROMPT.format(
        question=example.question,
        response=prediction.strip(),
        correct_answer=example.answer,
    )


def parse_browsecomp_plus_judgement(response: str) -> tuple[bool, float | None] | None:
    match = _CORRECT_RE.search(response)
    if match is None:
        return None
    confidence_match = _CONFIDENCE_RE.search(response)
    confidence = float(confidence_match.group(1)) if confidence_match else None
    if confidence is not None and not 0 <= confidence <= 100:
        return None
    return match.group(1).lower() == "yes", confidence


def evaluate_browsecomp_plus_predictions(
    examples: Iterable[BrowseCompPlusExample],
    predictions: dict[str, str],
    judge: BrowseCompPlusJudge,
    *,
    max_workers: int = 5,
    on_grade: Callable[[BrowseCompPlusGrade], None] | None = None,
) -> tuple[BrowseCompPlusGrade, ...]:
    example_list = list(examples)
    if not example_list:
        raise ValueError("At least one BrowseComp-Plus example is required")
    if max_workers < 1:
        raise ValueError("max_workers must be at least 1")

    def evaluate_one(example: BrowseCompPlusExample) -> BrowseCompPlusGrade:
        prediction = predictions.get(example.example_id, "").strip()
        if not prediction:
            return BrowseCompPlusGrade(
                example_id=example.example_id,
                status="empty_model_response",
                error="Model response was empty.",
            )
        try:
            raw = judge.judge(
                build_browsecomp_plus_grader_prompt(example, prediction)
            )
        except Exception as exc:
            return BrowseCompPlusGrade(
                example_id=example.example_id,
                status="judge_error",
                error=f"{type(exc).__name__}: {exc}",
            )
        parsed = parse_browsecomp_plus_judgement(raw)
        if parsed is None:
            return BrowseCompPlusGrade(
                example_id=example.example_id,
                status="invalid_auto_rater_response",
                raw_judge_response=raw,
                error="Judge response has no valid correct: yes/no field.",
            )
        correct, confidence = parsed
        return BrowseCompPlusGrade(
            example_id=example.example_id,
            status="valid",
            correct=correct,
            confidence=confidence,
            accuracy=1.0 if correct else 0.0,
            raw_judge_response=raw,
        )

    by_id: dict[str, BrowseCompPlusGrade] = {}
    with ThreadPoolExecutor(max_workers=min(max_workers, len(example_list))) as pool:
        futures = [pool.submit(evaluate_one, example) for example in example_list]
        for future in as_completed(futures):
            grade = future.result()
            by_id[grade.example_id] = grade
            if on_grade is not None:
                on_grade(grade)
    return tuple(by_id[example.example_id] for example in example_list)


def summarize_browsecomp_plus_grades(
    grades: Iterable[BrowseCompPlusGrade],
    *,
    candidate_retrieval_recall: float | None = None,
    fetched_evidence_recall: float | None = None,
) -> BrowseCompPlusEvaluationSummary:
    values = tuple(grades)
    if not values:
        raise ValueError("At least one BrowseComp-Plus grade is required")
    correct = sum(grade.correct is True for grade in values)
    return BrowseCompPlusEvaluationSummary(
        total_examples=len(values),
        valid_examples=sum(grade.status == "valid" for grade in values),
        correct=correct,
        incorrect=sum(grade.correct is False for grade in values),
        empty_model_responses=sum(
            grade.status == "empty_model_response" for grade in values
        ),
        invalid_auto_rater_responses=sum(
            grade.status == "invalid_auto_rater_response" for grade in values
        ),
        judge_errors=sum(grade.status == "judge_error" for grade in values),
        accuracy=correct / len(values),
        candidate_retrieval_recall=candidate_retrieval_recall,
        fetched_evidence_recall=fetched_evidence_recall,
    )


def verify_browsecomp_plus_assets(config: BrowseCompPlusConfig) -> None:
    for path in (config.qrels_gold_path, config.qrels_evidence_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    for index, expected_hash in enumerate(_CORPUS_SHA256):
        path = config.corpus_dir / f"train-{index:05d}-of-00007.parquet"
        if _sha256_file(path) != expected_hash:
            raise BrowseCompPlusError(f"Corpus shard checksum mismatch: {path}")
    if index_document_count(config.index_path) != BROWSECOMP_PLUS_EXPECTED_DOCUMENTS:
        raise BrowseCompPlusError("Local corpus index document count is invalid")


def _write_examples(path: Path, examples: Iterable[BrowseCompPlusExample]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        for example in examples:
            handle.write(
                json.dumps(
                    {
                        "query_id": example.example_id,
                        "query": example.question,
                        "answer": example.answer,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
    temporary.replace(path)


def _iter_parquet_documents(
    corpus_dir: Path,
) -> Iterator[tuple[str, str, str]]:
    try:
        import pyarrow.parquet as parquet
    except ImportError as exc:
        raise BrowseCompPlusError(
            "Building the corpus index requires the browsecomp-plus extra"
        ) from exc
    for path in sorted(corpus_dir.glob("train-*-of-00007.parquet")):
        source = parquet.ParquetFile(path)
        for batch in source.iter_batches(
            batch_size=128, columns=("docid", "text", "url")
        ):
            values = batch.to_pydict()
            yield from zip(
                map(str, values["docid"]),
                map(str, values["text"]),
                map(str, values["url"]),
                strict=True,
            )


def _download_verified(
    url: str,
    destination: Path,
    expected_sha256: str,
    *,
    progress: Callable[[str], None] | None,
) -> None:
    if destination.is_file() and _sha256_file(destination) == expected_sha256:
        if progress is not None:
            progress(f"verified existing {destination.name}")
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_suffix(destination.suffix + ".part")
    if partial.is_file() and _sha256_file(partial) == expected_sha256:
        partial.replace(destination)
        if progress is not None:
            progress(f"verified {destination.name}")
        return
    offset = partial.stat().st_size if partial.exists() else 0
    headers = {"User-Agent": "graphptc/0.1"}
    if offset:
        headers["Range"] = f"bytes={offset}-"
    if progress is not None:
        progress(f"downloading {destination.name} from byte {offset}")
    with urlopen(Request(url, headers=headers), timeout=120) as response:
        append = offset > 0 and response.status == 206
        with partial.open("ab" if append else "wb") as output:
            while chunk := response.read(1024 * 1024):
                output.write(chunk)
    if _sha256_file(partial) != expected_sha256:
        raise BrowseCompPlusError(f"SHA256 mismatch for {destination.name}")
    partial.replace(destination)
    if progress is not None:
        progress(f"verified {destination.name}")


def _sha256_file(path: Path) -> str:
    if not path.is_file():
        return ""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()
