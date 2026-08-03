from __future__ import annotations

import ast
import hashlib
import json
import threading
import uuid
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, Protocol


@dataclass(frozen=True)
class SourceSpan:
    line: int
    column: int
    end_line: int
    end_column: int


@dataclass(frozen=True)
class StaticToolCallSite:
    callsite_id: str
    tool_name: str
    span: SourceSpan


@dataclass(frozen=True)
class ProgramAnalysis:
    code_sha256: str
    callsites: tuple[StaticToolCallSite, ...] = ()
    syntax_error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class ProgramAnalyzer:
    """Extract static tool call candidates without rewriting source code."""

    def analyze(self, code: str, tool_names: set[str]) -> ProgramAnalysis:
        code_hash = hashlib.sha256(code.encode()).hexdigest()
        try:
            tree = ast.parse(code)
        except SyntaxError as exc:
            location = f"line {exc.lineno}, column {exc.offset}"
            return ProgramAnalysis(
                code_sha256=code_hash,
                syntax_error=f"{exc.msg} ({location})",
            )

        callsites: list[StaticToolCallSite] = []
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Name):
                continue
            tool_name = node.func.id
            if tool_name not in tool_names:
                continue
            span = SourceSpan(
                line=node.lineno,
                column=node.col_offset,
                end_line=getattr(node, "end_lineno", node.lineno),
                end_column=getattr(node, "end_col_offset", node.col_offset),
            )
            location = (
                f"{code_hash}:{tool_name}:{span.line}:{span.column}:"
                f"{span.end_line}:{span.end_column}"
            )
            callsites.append(
                StaticToolCallSite(
                    callsite_id=f"callsite_{hashlib.sha256(location.encode()).hexdigest()[:20]}",
                    tool_name=tool_name,
                    span=span,
                )
            )
        callsites.sort(key=lambda value: (value.span.line, value.span.column))
        return ProgramAnalysis(code_sha256=code_hash, callsites=tuple(callsites))


@dataclass(frozen=True)
class ExecutionEvent:
    event_id: str
    sequence: int
    kind: str
    occurred_at: str
    episode_id: str
    status: str
    block_id: str | None = None
    tool_call_id: str | None = None
    parent_id: str | None = None
    payload: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class EventSink(Protocol):
    def append(self, event: ExecutionEvent) -> None: ...


class JsonlEventSink:
    """Thread-safe append-only persistence for GraphPTC events."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    def append(self, event: ExecutionEvent) -> None:
        record = json.dumps(event.to_dict(), ensure_ascii=False)
        with self._lock, self.path.open("a", encoding="utf-8", newline="\n") as handle:
            handle.write(record + "\n")


class GraphPTCRuntime:
    """Own episode/block identities and receive ordered execution events."""

    def __init__(self, sink: EventSink | None = None) -> None:
        self._sink = sink
        self._events: list[ExecutionEvent] = []
        self._sequence = 0
        self._block_sequence = 0
        self._lock = threading.Lock()
        self._episode_id: str | None = None
        self._episode_event_id: str | None = None

    @property
    def episode_id(self) -> str | None:
        return self._episode_id

    @property
    def events(self) -> tuple[ExecutionEvent, ...]:
        with self._lock:
            return tuple(self._events)

    def start_episode(self, task: str) -> str:
        if self._episode_id is not None:
            raise RuntimeError("A GraphPTC episode is already active")
        self._episode_id = f"episode_{uuid.uuid4().hex}"
        event = self.emit(
            "episode.started",
            status="running",
            payload={"task": summarize_value(task)},
        )
        self._episode_event_id = event.event_id
        return self._episode_id

    def finish_episode(
        self, *, status: str, error: str | None, duration_ms: float
    ) -> None:
        self.emit(
            "episode.finished",
            status=status,
            parent_id=self._episode_event_id,
            payload={"duration_ms": duration_ms, "error": error},
        )
        self._episode_id = None
        self._episode_event_id = None

    def start_block(self, turn: int, code: str) -> tuple[str, ExecutionEvent]:
        self._require_episode()
        self._block_sequence += 1
        block_id = f"{self._episode_id}:block:{self._block_sequence}"
        event = self.emit(
            "block.started",
            status="running",
            block_id=block_id,
            parent_id=self._episode_event_id,
            payload={"turn": turn, "code": summarize_value(code)},
        )
        return block_id, event

    def record_analysis(
        self,
        block_id: str,
        block_event_id: str,
        analysis: ProgramAnalysis,
    ) -> None:
        self.emit(
            "block.analyzed",
            status="failed" if analysis.syntax_error else "success",
            block_id=block_id,
            parent_id=block_event_id,
            payload=analysis.to_dict(),
        )

    def record_tool_entry(
        self,
        *,
        block_id: str,
        block_event_id: str,
        entry: Any,
        analysis: ProgramAnalysis,
    ) -> None:
        tool_call_id = f"tool_{entry.id}"
        mapping = _callsite_mapping(entry.tool_name, analysis)
        finished_at = _utc_timestamp(entry.timestamp)
        started_at = _utc_timestamp(
            entry.timestamp - timedelta(milliseconds=float(entry.duration_ms))
        )
        started = self.emit(
            "tool.started",
            status="running",
            occurred_at=started_at,
            block_id=block_id,
            tool_call_id=tool_call_id,
            parent_id=block_event_id,
            payload={
                "tool_name": entry.tool_name,
                "invocation_id": entry.invocation_id,
                "arguments": summarize_value(entry.arguments),
                **mapping,
            },
        )
        self.emit(
            "tool.finished",
            status=str(getattr(entry.status, "value", entry.status)),
            occurred_at=finished_at,
            block_id=block_id,
            tool_call_id=tool_call_id,
            parent_id=started.event_id,
            payload={
                "tool_name": entry.tool_name,
                "duration_ms": float(entry.duration_ms),
                "result": summarize_value(entry.result),
                "error": entry.error,
                "exception_type": entry.exception_type,
                **mapping,
            },
        )

    def finish_block(
        self,
        *,
        block_id: str,
        block_event_id: str,
        success: bool,
        duration_ms: float,
        stdout: str,
        invocation_id: str | None,
    ) -> None:
        self.emit(
            "block.finished",
            status="success" if success else "failed",
            block_id=block_id,
            parent_id=block_event_id,
            payload={
                "duration_ms": duration_ms,
                "stdout": summarize_value(stdout),
                "invocation_id": invocation_id,
            },
        )

    def emit(
        self,
        kind: str,
        *,
        status: str,
        occurred_at: str | None = None,
        block_id: str | None = None,
        tool_call_id: str | None = None,
        parent_id: str | None = None,
        payload: dict[str, Any] | None = None,
    ) -> ExecutionEvent:
        episode_id = self._require_episode()
        with self._lock:
            self._sequence += 1
            event = ExecutionEvent(
                event_id=f"{episode_id}:event:{self._sequence}",
                sequence=self._sequence,
                kind=kind,
                occurred_at=occurred_at or datetime.now(UTC).isoformat(),
                episode_id=episode_id,
                status=status,
                block_id=block_id,
                tool_call_id=tool_call_id,
                parent_id=parent_id,
                payload=payload or {},
            )
            self._events.append(event)
        if self._sink is not None:
            self._sink.append(event)
        return event

    def _require_episode(self) -> str:
        if self._episode_id is None:
            raise RuntimeError("No active GraphPTC episode")
        return self._episode_id


def summarize_value(value: Any, *, preview_chars: int = 500) -> dict[str, Any]:
    try:
        serialized = json.dumps(value, ensure_ascii=False, sort_keys=True, default=repr)
    except (TypeError, ValueError):
        serialized = repr(value)
    return {
        "type": type(value).__name__,
        "chars": len(serialized),
        "sha256": hashlib.sha256(serialized.encode()).hexdigest(),
        "preview": serialized[:preview_chars],
        "truncated": len(serialized) > preview_chars,
    }


def _callsite_mapping(
    tool_name: str, analysis: ProgramAnalysis
) -> dict[str, Any]:
    candidates = [
        callsite.callsite_id
        for callsite in analysis.callsites
        if callsite.tool_name == tool_name
    ]
    if len(candidates) == 1:
        return {
            "source_mapping": "unique_static_candidate",
            "static_callsite_id": candidates[0],
            "candidate_callsite_ids": candidates,
        }
    return {
        "source_mapping": "ambiguous" if candidates else "unmapped",
        "static_callsite_id": None,
        "candidate_callsite_ids": candidates,
    }


def _utc_timestamp(value: datetime) -> str:
    if value.tzinfo is None:
        value = value.astimezone()
    return value.astimezone(UTC).isoformat()
