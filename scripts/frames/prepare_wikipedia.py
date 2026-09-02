from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from tfrecord.reader import tfrecord_loader


GCS_LIST_URL = (
    "https://storage.googleapis.com/storage/v1/b/tfds-data/o?"
    "prefix=datasets/wikipedia/20230601.en/1.0.0/&maxResults=1000"
)
GCS_DOWNLOAD_BASE = "https://storage.googleapis.com/tfds-data/"
EXPECTED_DOCUMENTS = 6_672_479
REPO_ROOT = Path(__file__).resolve().parents[2]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        type=Path,
        default=REPO_ROOT / "data" / "frames" / "wikipedia-20230601",
    )
    parser.add_argument("--download-workers", type=int, default=6)
    parser.add_argument("--convert-workers", type=int, default=8)
    parser.add_argument("--index-threads", type=int, default=8)
    args = parser.parse_args()

    tfrecord_dir = args.root / "tfrecord"
    json_dir = args.root / "json"
    index_dir = args.root / "index"
    tfrecord_dir.mkdir(parents=True, exist_ok=True)
    json_dir.mkdir(parents=True, exist_ok=True)

    objects = _tfrecord_objects()
    with ThreadPoolExecutor(max_workers=args.download_workers) as executor:
        list(executor.map(lambda item: _download(item, tfrecord_dir), objects))

    shards = sorted(tfrecord_dir.glob("*.tfrecord-*"))
    with ThreadPoolExecutor(max_workers=args.convert_workers) as executor:
        counts = list(executor.map(lambda path: _convert(path, json_dir), shards))
    document_count = sum(counts)
    if document_count != EXPECTED_DOCUMENTS:
        raise ValueError(f"expected {EXPECTED_DOCUMENTS} articles, converted {document_count}")

    if not index_dir.exists():
        env = dict(os.environ)
        env["JAVA_HOME"] = "/usr/lib/jvm/java-21-openjdk-amd64"
        env["PATH"] = f"{env['JAVA_HOME']}/bin:{env['PATH']}"
        env["JAVA_TOOL_OPTIONS"] = "-Xms1g -Xmx10g"
        subprocess.run(
            [
                sys.executable,
                "-m",
                "pyserini.index.lucene",
                "--collection",
                "JsonCollection",
                "--input",
                os.fspath(json_dir),
                "--index",
                os.fspath(index_dir),
                "--generator",
                "DefaultLuceneDocumentGenerator",
                "--threads",
                str(args.index_threads),
                "--storePositions",
                "--storeDocvectors",
                "--storeRaw",
            ],
            env=env,
            check=True,
        )

    manifest = {
        "corpus_snapshot": "wikipedia/20230601.en",
        "tfds_version": "1.0.0",
        "shards": len(shards),
        "document_count": document_count,
        "index_path": os.fspath(index_dir),
    }
    (args.root / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print(json.dumps(manifest))


def _tfrecord_objects() -> list[dict[str, Any]]:
    with urllib.request.urlopen(GCS_LIST_URL) as response:
        listing = json.load(response)
    objects = [
        item
        for item in listing["items"]
        if "/wikipedia-train.tfrecord-" in item["name"]
    ]
    if len(objects) != 256:
        raise ValueError(f"expected 256 TFRecord shards, found {len(objects)}")
    return sorted(objects, key=lambda item: item["name"])


def _download(item: dict[str, Any], destination_dir: Path) -> Path:
    destination = destination_dir / Path(item["name"]).name
    expected_size = int(item["size"])
    if destination.exists() and destination.stat().st_size == expected_size:
        return destination
    temporary = destination.with_suffix(destination.suffix + ".part")
    url = GCS_DOWNLOAD_BASE + urllib.parse.quote(item["name"], safe="/")
    with urllib.request.urlopen(url) as response, temporary.open("wb") as handle:
        while chunk := response.read(8 * 1024 * 1024):
            handle.write(chunk)
    if temporary.stat().st_size != expected_size:
        raise ValueError(f"incomplete TFRecord shard: {destination.name}")
    temporary.replace(destination)
    return destination


def _convert(source: Path, destination_dir: Path) -> int:
    destination = destination_dir / f"{source.name}.jsonl"
    count_dir = destination_dir.parent / "counts"
    count_dir.mkdir(parents=True, exist_ok=True)
    count_path = count_dir / f"{source.name}.count"
    if destination.exists() and count_path.exists():
        return int(count_path.read_text(encoding="utf-8"))
    shard = source.name.split(".tfrecord-", 1)[1].split("-of-", 1)[0]
    temporary = destination.with_suffix(destination.suffix + ".part")
    count = 0
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        for index, record in enumerate(
            tfrecord_loader(
                os.fspath(source),
                None,
                description={"title": "byte", "text": "byte"},
            )
        ):
            title = record["title"].decode("utf-8")
            text = record["text"].decode("utf-8")
            handle.write(
                json.dumps(
                    {
                        "id": f"{shard}-{index:06d}",
                        "title": title,
                        "contents": f"{title}\n{text}",
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
            count += 1
    temporary.replace(destination)
    count_path.write_text(str(count), encoding="utf-8")
    return count


if __name__ == "__main__":
    main()
