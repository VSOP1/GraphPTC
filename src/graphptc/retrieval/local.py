from __future__ import annotations

import json
import re
import sqlite3
import threading
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from .web import ToolBudgetExceeded


@dataclass(frozen=True)
class LocalSearchCall:
    sequence: int
    operation: str
    item_count: int
    duration_ms: float
    success: bool
    query: str | None = None
    docid: str | None = None
    docids: tuple[str, ...] = ()
    output_chars: int = 0
    error: str | None = None


class SQLiteCorpusSearchTools:
    """Metered, deterministic search over a fixed SQLite FTS5 corpus."""

    def __init__(
        self,
        index_path: str | Path,
        *,
        top_k: int = 5,
        snippet_max_chars: int = 2_048,
        max_tool_calls: int = 1_000,
    ) -> None:
        self._index_path = Path(index_path)
        if not self._index_path.is_file():
            raise FileNotFoundError(self._index_path)
        self._top_k = _positive(top_k, "top_k")
        self._snippet_max_chars = _positive(
            snippet_max_chars, "snippet_max_chars"
        )
        self._max_tool_calls = _positive(max_tool_calls, "max_tool_calls")
        self._calls: list[LocalSearchCall] = []
        self._consumed = 0
        self._lock = threading.Lock()

    @property
    def calls(self) -> list[dict[str, Any]]:
        with self._lock:
            return [asdict(call) for call in self._calls]

    @property
    def consumed(self) -> int:
        with self._lock:
            return self._consumed

    def search(self, *, query: str) -> list[dict[str, Any]]:
        """Search the frozen corpus and return top documents with short snippets."""
        sequence = self._reserve(1)
        started = time.perf_counter()
        try:
            with _read_connection(self._index_path) as connection:
                fts_query = _fts_query(connection, query)
                rows = connection.execute(
                    """
                    SELECT d.docid, d.url, d.text, bm25(documents_fts) AS rank
                    FROM documents_fts
                    JOIN documents AS d ON d.rowid = documents_fts.rowid
                    WHERE documents_fts MATCH ?
                    ORDER BY rank
                    LIMIT ?
                    """,
                    (fts_query, self._top_k),
                ).fetchall()
            results = [
                {
                    "docid": str(row[0]),
                    "url": row[1] or "",
                    "score": -float(row[3]),
                    "snippet": _truncate(row[2] or "", self._snippet_max_chars),
                }
                for row in rows
            ]
            self._record(
                sequence,
                "search",
                1,
                started,
                True,
                query=query,
                docids=tuple(item["docid"] for item in results),
                output_chars=sum(len(item["snippet"]) for item in results),
            )
            return results
        except Exception as exc:
            self._record(
                sequence,
                "search",
                1,
                started,
                False,
                query=query,
                error=str(exc),
            )
            raise

    def fetch(self, *, docid: str) -> dict[str, Any]:
        """Return one complete document from the frozen corpus."""
        sequence = self._reserve(1)
        started = time.perf_counter()
        try:
            with _read_connection(self._index_path) as connection:
                row = connection.execute(
                    "SELECT docid, url, text FROM documents WHERE docid = ?",
                    (docid,),
                ).fetchone()
            if row is None:
                raise KeyError(f"Unknown docid: {docid}")
            result = {
                "docid": str(row[0]),
                "url": row[1] or "",
                "content": row[2] or "",
            }
            self._record(
                sequence,
                "fetch",
                1,
                started,
                True,
                docid=docid,
                docids=(str(row[0]),),
                output_chars=len(result["content"]),
            )
            return result
        except Exception as exc:
            self._record(
                sequence,
                "fetch",
                1,
                started,
                False,
                docid=docid,
                error=str(exc),
            )
            raise

    def _reserve(self, amount: int) -> int:
        with self._lock:
            if self._consumed + amount > self._max_tool_calls:
                raise ToolBudgetExceeded(
                    f"Tool budget exceeded: {self._consumed + amount} > "
                    f"{self._max_tool_calls}"
                )
            self._consumed += amount
            return self._consumed

    def _record(
        self,
        sequence: int,
        operation: str,
        item_count: int,
        started: float,
        success: bool,
        *,
        query: str | None = None,
        docid: str | None = None,
        docids: tuple[str, ...] = (),
        output_chars: int = 0,
        error: str | None = None,
    ) -> None:
        call = LocalSearchCall(
            sequence=sequence,
            operation=operation,
            item_count=item_count,
            duration_ms=(time.perf_counter() - started) * 1_000,
            success=success,
            query=query,
            docid=docid,
            docids=docids,
            output_chars=output_chars,
            error=error,
        )
        with self._lock:
            self._calls.append(call)


class OfficialCorpusSearchTools:
    """Metered client for the pinned BrowseComp-Plus Pyserini service."""

    def __init__(
        self,
        base_url: str,
        *,
        max_tool_calls: int = 1_000,
        timeout_seconds: float = 60.0,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        if not self._base_url:
            raise ValueError("Retriever URL must not be empty")
        self._max_tool_calls = _positive(max_tool_calls, "max_tool_calls")
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        self._timeout_seconds = timeout_seconds
        self._calls: list[LocalSearchCall] = []
        self._consumed = 0
        self._lock = threading.Lock()

    @property
    def calls(self) -> list[dict[str, Any]]:
        with self._lock:
            return [asdict(call) for call in self._calls]

    @property
    def consumed(self) -> int:
        with self._lock:
            return self._consumed

    def metadata(self) -> dict[str, Any]:
        value = self._request_json("/metadata", None)
        if not isinstance(value, dict):
            raise ValueError("Retriever metadata must be an object")
        return value

    def search(self, *, query: str) -> list[dict[str, Any]]:
        sequence = self._reserve()
        started = time.perf_counter()
        try:
            value = self._request_json("/search", {"query": query})
            if not isinstance(value, list) or not all(
                isinstance(item, dict) for item in value
            ):
                raise ValueError("Retriever search response must be a list of objects")
            results = list(value)
            self._record(
                LocalSearchCall(
                    sequence=sequence,
                    operation="search",
                    item_count=1,
                    duration_ms=(time.perf_counter() - started) * 1_000,
                    success=True,
                    query=query,
                    docids=tuple(str(item["docid"]) for item in results),
                    output_chars=sum(len(str(item.get("snippet", ""))) for item in results),
                )
            )
            return results
        except Exception as exc:
            self._record(
                LocalSearchCall(
                    sequence=sequence,
                    operation="search",
                    item_count=1,
                    duration_ms=(time.perf_counter() - started) * 1_000,
                    success=False,
                    query=query,
                    error=str(exc),
                )
            )
            raise

    def fetch(self, *, docid: str) -> dict[str, Any]:
        sequence = self._reserve()
        started = time.perf_counter()
        try:
            value = self._request_json("/fetch", {"docid": docid})
            if not isinstance(value, dict) or "content" not in value:
                raise ValueError("Retriever fetch response must contain content")
            result = dict(value)
            self._record(
                LocalSearchCall(
                    sequence=sequence,
                    operation="fetch",
                    item_count=1,
                    duration_ms=(time.perf_counter() - started) * 1_000,
                    success=True,
                    docid=docid,
                    docids=(str(result.get("docid", docid)),),
                    output_chars=len(str(result["content"])),
                )
            )
            return result
        except Exception as exc:
            self._record(
                LocalSearchCall(
                    sequence=sequence,
                    operation="fetch",
                    item_count=1,
                    duration_ms=(time.perf_counter() - started) * 1_000,
                    success=False,
                    docid=docid,
                    error=str(exc),
                )
            )
            raise

    def _reserve(self) -> int:
        with self._lock:
            if self._consumed + 1 > self._max_tool_calls:
                raise ToolBudgetExceeded(
                    f"Tool budget exceeded: {self._consumed + 1} > "
                    f"{self._max_tool_calls}"
                )
            self._consumed += 1
            return self._consumed

    def _record(self, call: LocalSearchCall) -> None:
        with self._lock:
            self._calls.append(call)

    def _request_json(self, path: str, payload: dict[str, Any] | None) -> Any:
        data = None if payload is None else _json_bytes(payload)
        request = Request(
            f"{self._base_url}{path}",
            data=data,
            headers={"Content-Type": "application/json"},
            method="GET" if payload is None else "POST",
        )
        try:
            with urlopen(request, timeout=self._timeout_seconds) as response:
                return _load_json_bytes(response.read())
        except HTTPError as exc:
            body = _load_json_bytes(exc.read())
            detail = body.get("error") if isinstance(body, dict) else str(body)
            raise RuntimeError(f"Retriever HTTP {exc.code}: {detail}") from exc


def build_sqlite_fts_index(
    path: str | Path,
    documents: Iterable[tuple[str, str, str]],
    *,
    batch_size: int = 500,
) -> int:
    """Build an FTS5 index atomically from (docid, text, url) records."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    if temporary.exists():
        temporary.unlink()

    count = 0
    connection = sqlite3.connect(temporary)
    try:
        connection.executescript(
            """
            PRAGMA journal_mode=OFF;
            PRAGMA synchronous=OFF;
            PRAGMA temp_store=MEMORY;
            CREATE TABLE documents (
                docid TEXT NOT NULL UNIQUE,
                text TEXT NOT NULL,
                url TEXT NOT NULL
            );
            CREATE VIRTUAL TABLE documents_fts USING fts5(
                text,
                content='documents',
                content_rowid='rowid',
                tokenize='unicode61 remove_diacritics 2'
            );
            """
        )
        for batch in _batched(documents, batch_size):
            connection.executemany(
                "INSERT INTO documents(docid, text, url) VALUES (?, ?, ?)", batch
            )
            count += len(batch)
            connection.commit()
        connection.execute(
            "INSERT INTO documents_fts(documents_fts) VALUES ('rebuild')"
        )
        connection.execute(
            "CREATE VIRTUAL TABLE documents_vocab USING "
            "fts5vocab(documents_fts, 'row')"
        )
        connection.execute(
            "CREATE TABLE metadata(key TEXT PRIMARY KEY, value TEXT NOT NULL)"
        )
        connection.execute(
            "INSERT INTO metadata(key, value) VALUES ('document_count', ?)",
            (str(count),),
        )
        connection.commit()
    finally:
        connection.close()
    temporary.replace(destination)
    return count


def index_document_count(path: str | Path) -> int:
    with _read_connection(Path(path)) as connection:
        row = connection.execute(
            "SELECT value FROM metadata WHERE key='document_count'"
        ).fetchone()
    if row is None:
        raise ValueError("Local corpus index has no document_count metadata")
    return int(row[0])


def _read_connection(path: Path) -> sqlite3.Connection:
    uri = f"{path.resolve().as_uri()}?mode=ro"
    return sqlite3.connect(uri, uri=True)


def _fts_query(connection: sqlite3.Connection, value: str) -> str:
    tokens = list(
        dict.fromkeys(
            token.casefold()
            for token in re.findall(r"\w+", value, flags=re.UNICODE)
            if len(token) > 1 or token.isdigit()
        )
    )
    if not tokens:
        raise ValueError("Search query must contain at least one word")
    placeholders = ",".join("?" for _ in tokens)
    frequencies = {
        str(term): int(document_count)
        for term, document_count in connection.execute(
            f"SELECT term, doc FROM documents_vocab WHERE term IN ({placeholders})",
            tokens,
        )
    }
    ranked = sorted(
        (token for token in tokens if token in frequencies),
        key=lambda token: (frequencies[token], tokens.index(token)),
    )
    selected = ranked[:12] or tokens[:12]
    return " OR ".join(
        f'"{token.replace(chr(34), chr(34) * 2)}"' for token in selected
    )


def _truncate(value: str, maximum: int) -> str:
    return value if len(value) <= maximum else value[:maximum]


def _json_bytes(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":")).encode()


def _load_json_bytes(value: bytes) -> Any:
    return json.loads(value.decode())


def _positive(value: int, name: str) -> int:
    if value < 1:
        raise ValueError(f"{name} must be at least 1")
    return value


def _batched(
    values: Iterable[tuple[str, str, str]], size: int
) -> Iterator[list[tuple[str, str, str]]]:
    if size < 1:
        raise ValueError("batch_size must be at least 1")
    batch: list[tuple[str, str, str]] = []
    for value in values:
        batch.append(value)
        if len(batch) == size:
            yield batch
            batch = []
    if batch:
        yield batch
