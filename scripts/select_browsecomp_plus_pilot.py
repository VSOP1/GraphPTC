from __future__ import annotations

import argparse
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


DEFAULT_EXCLUDED_IDS = ("1", "3", "5", "6", "7", "8", "10", "11", "12", "15")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source",
        type=Path,
        default=Path("data/browsecomp_plus/questions.jsonl"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/browsecomp_plus/pilot20.questions.jsonl"),
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("data/browsecomp_plus/pilot20.manifest.json"),
    )
    parser.add_argument("--count", type=int, default=20)
    parser.add_argument("--seed", default="baseline-v2-pilot20-2026-08-01")
    parser.add_argument(
        "--exclude-id",
        action="append",
        default=list(DEFAULT_EXCLUDED_IDS),
    )
    args = parser.parse_args()

    records = _load_jsonl(args.source)
    excluded = set(args.exclude_id)
    eligible = [
        record for record in records if str(record.get("query_id")) not in excluded
    ]
    ranked = sorted(
        eligible,
        key=lambda record: _selection_key(args.seed, str(record["query_id"])),
    )
    selected = ranked[: args.count]
    if len(selected) != args.count:
        raise ValueError(
            f"Requested {args.count} examples but only {len(selected)} are eligible"
        )

    _write_jsonl(args.output, selected)
    manifest = {
        "schema_version": 1,
        "created_at": datetime.now(UTC).isoformat(),
        "method": "sha256(seed + ':' + query_id), ascending",
        "seed": args.seed,
        "source": str(args.source),
        "source_sha256": _sha256(args.source),
        "source_examples": len(records),
        "count": len(selected),
        "excluded_ids": sorted(excluded, key=lambda value: int(value)),
        "selected_ids": [str(record["query_id"]) for record in selected],
        "output": str(args.output),
        "output_sha256": _sha256(args.output),
    }
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


def _selection_key(seed: str, example_id: str) -> str:
    return hashlib.sha256(f"{seed}:{example_id}".encode()).hexdigest()


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    records = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            if not isinstance(record, dict) or "query_id" not in record:
                raise ValueError(f"Invalid record on line {line_number}")
            records.append(record)
    return records


def _write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    content = "".join(
        json.dumps(record, ensure_ascii=False) + "\n" for record in records
    )
    path.write_text(content, encoding="utf-8")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


if __name__ == "__main__":
    main()
