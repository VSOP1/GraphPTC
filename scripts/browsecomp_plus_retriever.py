from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

from pyserini.search.lucene import LuceneSearcher
from transformers import AutoTokenizer


class OfficialRetriever:
    def __init__(
        self,
        index_path: Path,
        tokenizer_path: Path,
        *,
        top_k: int,
        snippet_max_tokens: int,
        index_revision: str,
        tokenizer_revision: str,
    ) -> None:
        self._index_path = index_path
        self._tokenizer_path = tokenizer_path
        self._top_k = top_k
        self._snippet_max_tokens = snippet_max_tokens
        self._index_revision = index_revision
        self._tokenizer_revision = tokenizer_revision
        self._searcher = LuceneSearcher(str(index_path))
        self._tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_path,
            local_files_only=True,
        )

    def search(self, query: str) -> list[dict[str, Any]]:
        query = query.strip()
        if not query:
            raise ValueError("Search query must not be empty")
        results: list[dict[str, Any]] = []
        for hit in self._searcher.search(query, self._top_k):
            raw = json.loads(hit.lucene_document.get("raw"))
            text = str(raw["contents"])
            tokens = self._tokenizer.encode(text, add_special_tokens=False)
            snippet = self._tokenizer.decode(
                tokens[: self._snippet_max_tokens], skip_special_tokens=True
            )
            results.append(
                {"docid": hit.docid, "score": hit.score, "snippet": snippet}
            )
        return results

    def fetch(self, docid: str) -> dict[str, str]:
        docid = docid.strip()
        if not docid:
            raise ValueError("Document ID must not be empty")
        document = self._searcher.doc(docid)
        if document is None:
            raise KeyError(f"Unknown docid: {docid}")
        raw = json.loads(document.raw())
        return {"docid": docid, "content": str(raw["contents"])}

    def metadata(self) -> dict[str, Any]:
        files = [
            (path.name, path.stat().st_size)
            for path in sorted(self._index_path.iterdir())
            if path.is_file()
        ]
        serialized = json.dumps(files, separators=(",", ":")).encode()
        return {
            "backend": "browsecomp_plus_official_bm25",
            "index_revision": self._index_revision,
            "index_manifest_sha256": hashlib.sha256(serialized).hexdigest(),
            "tokenizer": "Qwen/Qwen3-0.6B",
            "tokenizer_revision": self._tokenizer_revision,
            "tokenizer_manifest_sha256": _directory_sha256(self._tokenizer_path),
            "top_k": self._top_k,
            "snippet_max_tokens": self._snippet_max_tokens,
            "packages": {
                name: _package_version(name)
                for name in ("pyserini", "transformers", "huggingface-hub")
            },
            "python": platform.python_version(),
            "platform": platform.platform(),
            "java": _java_version(),
            "implementation_sha256": hashlib.sha256(
                Path(__file__).read_bytes()
            ).hexdigest(),
        }


def _package_version(name: str) -> str:
    try:
        return version(name)
    except PackageNotFoundError:
        return "unknown"


def _directory_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    for file_path in sorted(item for item in path.rglob("*") if item.is_file()):
        if ".cache" in file_path.parts:
            continue
        digest.update(file_path.relative_to(path).as_posix().encode())
        digest.update(file_path.read_bytes())
    return digest.hexdigest()


def _java_version() -> str:
    result = subprocess.run(
        ["java", "-version"], capture_output=True, text=True, check=False
    )
    return (result.stderr or result.stdout).splitlines()[0]


def _handler(retriever: OfficialRetriever) -> type[BaseHTTPRequestHandler]:
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
                length = int(self.headers.get("Content-Length", "0"))
                if length < 1 or length > 1_000_000:
                    raise ValueError("Invalid request size")
                payload = json.loads(self.rfile.read(length))
                if not isinstance(payload, dict):
                    raise ValueError("Request body must be an object")
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
    parser.add_argument("--tokenizer-path", type=Path, required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--snippet-max-tokens", type=int, default=512)
    parser.add_argument("--index-revision", required=True)
    parser.add_argument("--tokenizer-revision", required=True)
    args = parser.parse_args()

    retriever = OfficialRetriever(
        args.index_path,
        args.tokenizer_path,
        top_k=args.top_k,
        snippet_max_tokens=args.snippet_max_tokens,
        index_revision=args.index_revision,
        tokenizer_revision=args.tokenizer_revision,
    )
    server = ThreadingHTTPServer((args.host, args.port), _handler(retriever))
    server.serve_forever()


if __name__ == "__main__":
    main()
