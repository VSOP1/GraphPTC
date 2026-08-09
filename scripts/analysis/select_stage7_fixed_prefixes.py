from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from pathlib import Path
from typing import Any


PREFIX = "GRAPH_PROGRESS_SNAPSHOT "


def main() -> None:
    parser = argparse.ArgumentParser(description="Select frozen Stage 7.5b prefix pairs.")
    parser.add_argument("gate_path", type=Path)
    parser.add_argument("archive_dir", type=Path)
    parser.add_argument("output_path", type=Path)
    args = parser.parse_args()
    gate = _json(args.gate_path)
    by_example: dict[str, list[dict[str, Any]]] = {}
    for path in args.archive_dir.glob("*/*.json.gz"):
        payload = _gzip_json(path)
        example_id = str(payload["example_id"])
        selected = _prefix_metadata(path, payload)
        if selected is not None:
            by_example.setdefault(example_id, []).append(selected)

    prefixes: list[dict[str, Any]] = []
    for example_id in gate["capture_example_ids"]:
        candidates = sorted(by_example.get(example_id, []), key=lambda item: item["next_turn"])
        if not candidates:
            raise ValueError(f"no captured capsule prefix for {example_id}")
        chosen = [("earliest_capsule", candidates[0])]
        stagnation = next(
            (
                item
                for item in candidates
                if any(item["capsule"][field] > 0 for field in gate["selection"]["stagnation_fields"])
            ),
            None,
        )
        if stagnation is not None:
            chosen.append(("earliest_positive_stagnation_capsule", stagnation))
        seen: set[str] = set()
        for reason, item in chosen:
            if item["archive_sha256"] in seen:
                continue
            seen.add(item["archive_sha256"])
            prefix_id = f"{example_id}:turn:{item['next_turn']}"
            prefixes.append({"prefix_id": prefix_id, "selection_reason": reason, **item})

    report = {
        "schema_version": 1,
        "stage": "7.5b",
        "mode": "fixed-prefix-selection-manifest",
        "gate_sha256": _sha256(args.gate_path),
        "archive_dir": str(args.archive_dir),
        "captured_examples": sorted(by_example, key=int),
        "prefixes": prefixes,
    }
    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    args.output_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"captured_examples": len(by_example), "paired_prefixes": len(prefixes)}))


def _prefix_metadata(path: Path, payload: dict[str, Any]) -> dict[str, Any] | None:
    messages = payload["messages"]
    indexes = [
        index
        for index, message in enumerate(messages)
        if message.get("role") == "user" and str(message.get("content", "")).startswith(PREFIX)
    ]
    if not indexes:
        return None
    index = indexes[-1]
    content = str(messages[index]["content"])
    return {
        "example_id": str(payload["example_id"]),
        "next_turn": int(payload["next_turn"]),
        "archive_path": str(path),
        "archive_sha256": _sha256(path),
        "message_count": len(messages),
        "capsule_message_index": index,
        "capsule_chars": len(content),
        "capsule": {
            key: value
            for key, value in json.loads(content.removeprefix(PREFIX)).items()
            if key != "padding"
        },
    }


def _gzip_json(path: Path) -> dict[str, Any]:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        return json.load(handle)


def _json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


if __name__ == "__main__":
    main()
