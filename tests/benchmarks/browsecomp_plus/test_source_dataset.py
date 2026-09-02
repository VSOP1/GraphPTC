from __future__ import annotations

import base64
import csv
from pathlib import Path

from graphptc.benchmarks.browsecomp_plus.source_dataset import (
    BrowseCompExample,
    derive_key,
    load_browsecomp,
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
