from __future__ import annotations

import argparse
import json
import platform
import subprocess
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

from pyserini.pyclass import autoclass


JSimpleSearcher = autoclass("io.anserini.search.SimpleSearcher")


class FramesRetriever:
    def __init__(
        self,
        index_path: Path,
        *,
        top_k: int,
        snippet_max_chars: int,
        corpus_snapshot: str,
        document_count: int,
    ) -> None:
        self._index_path = index_path
        self._top_k = top_k
        self._snippet_max_chars = snippet_max_chars
        self._corpus_snapshot = corpus_snapshot
        self._searcher = JSimpleSearcher(str(index_path))
        self._searcher.set_bm25(0.9, 0.4)
        self._document_count = int(self._searcher.get_total_num_docs())
        if self._document_count != document_count:
            raise ValueError(
                f"expected {document_count} indexed articles, found {self._document_count}"
            )

    def search(self, query: str) -> list[dict[str, Any]]:
        query = query.strip()
        if not query:
            raise ValueError("Search query must not be empty")
        results: list[dict[str, Any]] = []
        for hit in self._searcher.search(query, self._top_k):
            raw = json.loads(hit.lucene_document.get("raw"))
            text = str(raw["contents"])
            results.append(
                {
                    "docid": hit.docid,
                    "title": str(raw["title"]),
                    "score": hit.score,
                    "snippet": text[: self._snippet_max_chars],
                }
            )
        return results

    def fetch(self, docid: str) -> dict[str, str]:
        docid = docid.strip()
        if not docid:
            raise ValueError("Document ID must not be empty")
        document = self._searcher.doc(docid)
        if document is None:
            raise KeyError(f"Unknown docid: {docid}")
        raw = json.loads(document.get("raw"))
        return {
            "docid": docid,
            "title": str(raw["title"]),
            "content": str(raw["contents"]),
        }

    def metadata(self) -> dict[str, Any]:
        return {
            "backend": "frames_pyserini_bm25",
            "corpus_snapshot": self._corpus_snapshot,
            "source": "TensorFlow Datasets wikipedia/20230601.en/1.0.0",
            "document_count": self._document_count,
            "top_k": self._top_k,
            "snippet_max_chars": self._snippet_max_chars,
            "bm25_parameters": {"k1": 0.9, "b": 0.4},
            "packages": {"pyserini": _package_version("pyserini")},
            "python": platform.python_version(),
            "java": _java_version(),
        }


def _package_version(name: str) -> str:
    try:
        return version(name)
    except PackageNotFoundError:
        return "unknown"


def _java_version() -> str:
    result = subprocess.run(
        ["java", "-version"], capture_output=True, text=True, check=False
    )
    return (result.stderr or result.stdout).splitlines()[0]


def _handler(retriever: FramesRetriever) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            if self.path == "/health":
                self._send(200, {"status": "ok"})
            elif self.path == "/metadata":
                self._send(200, retriever.metadata())
            else:
                self._send(404, {"error": "Unknown endpoint"})

        def do_POST(self) -> None:
            try:
                payload = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                if self.path == "/search":
                    result = retriever.search(str(payload.get("query", "")))
                elif self.path == "/fetch":
                    result = retriever.fetch(str(payload.get("docid", "")))
                else:
                    self._send(404, {"error": "Unknown endpoint"})
                    return
                self._send(200, result)
            except (KeyError, ValueError, json.JSONDecodeError) as exc:
                self._send(400, {"error": str(exc)})
            except Exception as exc:
                self._send(500, {"error": f"{type(exc).__name__}: {exc}"})

        def log_message(self, format: str, *args: Any) -> None:
            return

        def _send(self, status: int, value: Any) -> None:
            body = json.dumps(value, ensure_ascii=False).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    return Handler


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--index-path", type=Path, required=True)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8890)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--snippet-max-chars", type=int, default=1200)
    parser.add_argument("--corpus-snapshot", default="wikipedia/20230601.en")
    parser.add_argument("--document-count", type=int, default=6_672_479)
    args = parser.parse_args()
    retriever = FramesRetriever(
        args.index_path,
        top_k=args.top_k,
        snippet_max_chars=args.snippet_max_chars,
        corpus_snapshot=args.corpus_snapshot,
        document_count=args.document_count,
    )
    ThreadingHTTPServer((args.host, args.port), _handler(retriever)).serve_forever()


if __name__ == "__main__":
    main()
