from __future__ import annotations

import base64
import csv
import hashlib
import io
from dataclasses import dataclass
from pathlib import Path
from urllib.request import urlopen


BROWSECOMP_DATASET_URL = (
    "https://openaipublic.blob.core.windows.net/simple-evals/"
    "browse_comp_test_set.csv"
)
BROWSECOMP_DATASET_SHA256 = (
    "7b24471cd5b3eb2a46830a14802b5c029ea62f488ff75a0f88af7923d1454abf"
)
BROWSECOMP_EXPECTED_ROWS = 1_266
_DATASET_COLUMNS = ("problem", "answer", "problem_topic", "canary")


class BrowseCompDatasetError(RuntimeError):
    """Raised when the encrypted source dataset is missing or malformed."""


@dataclass(frozen=True)
class BrowseCompExample:
    example_id: str
    problem: str
    answer: str
    problem_topic: str


def derive_key(password: str, length: int) -> bytes:
    digest = hashlib.sha256(password.encode()).digest()
    return digest * (length // len(digest)) + digest[: length % len(digest)]


def decrypt(ciphertext_b64: str, password: str) -> str:
    encrypted = base64.b64decode(ciphertext_b64, validate=True)
    key = derive_key(password, len(encrypted))
    return bytes(a ^ b for a, b in zip(encrypted, key, strict=True)).decode()


def download_browsecomp(path: str | Path) -> Path:
    """Download the encrypted source rows required to construct BrowseComp-Plus."""

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
    """Load source questions and answers used by the BrowseComp-Plus qrels."""

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
        raise BrowseCompDatasetError("BrowseComp source CSV must be UTF-8") from exc

    reader = csv.DictReader(io.StringIO(text))
    if tuple(reader.fieldnames or ()) != _DATASET_COLUMNS:
        raise BrowseCompDatasetError(
            f"Unexpected BrowseComp source columns: {reader.fieldnames}"
        )

    examples: list[BrowseCompExample] = []
    for index, row in enumerate(reader):
        try:
            problem = decrypt(row["problem"], row["canary"])
            answer = decrypt(row["answer"], row["canary"])
        except Exception as exc:
            raise BrowseCompDatasetError(
                f"Could not decrypt BrowseComp source row {index}"
            ) from exc
        if not problem.strip() or not answer.strip():
            raise BrowseCompDatasetError(f"BrowseComp source row {index} is empty")
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
            f"Expected {BROWSECOMP_EXPECTED_ROWS} source rows, got {len(examples)}"
        )
    return examples


def _verify_checksum(data: bytes) -> None:
    actual = hashlib.sha256(data).hexdigest()
    if actual != BROWSECOMP_DATASET_SHA256:
        raise BrowseCompDatasetError(
            f"BrowseComp source SHA256 mismatch: "
            f"expected {BROWSECOMP_DATASET_SHA256}, got {actual}"
        )
