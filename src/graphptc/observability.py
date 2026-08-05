from __future__ import annotations

import copy
import json
import threading
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Protocol


class EventSink(Protocol):
    def append(self, event: dict[str, Any]) -> None: ...


class InMemoryEventSink:
    def __init__(self) -> None:
        self._events: list[dict[str, Any]] = []
        self._lock = threading.Lock()

    @property
    def events(self) -> list[dict[str, Any]]:
        with self._lock:
            return copy.deepcopy(self._events)

    def append(self, event: dict[str, Any]) -> None:
        with self._lock:
            self._events.append(copy.deepcopy(event))


class JsonlEventSink:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self._lock = threading.Lock()

    def append(self, event: dict[str, Any]) -> None:
        line = json.dumps(event, ensure_ascii=False, separators=(",", ":"))
        with self._lock:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8", newline="\n") as handle:
                handle.write(line + "\n")
                handle.flush()


class FanoutEventSink:
    def __init__(self, *sinks: EventSink) -> None:
        self._sinks = sinks

    def append(self, event: dict[str, Any]) -> None:
        for sink in self._sinks:
            sink.append(event)


class ExecutionObserver:
    """Append-only execution events that cannot affect the observed run."""

    def __init__(self, sink: EventSink, *, episode_id: str, task_id: str) -> None:
        self._sink = sink
        self.episode_id = episode_id
        self.task_id = task_id
        self._sequence = 0
        self._lock = threading.Lock()
        self._errors: list[str] = []

    @property
    def errors(self) -> tuple[str, ...]:
        with self._lock:
            return tuple(self._errors)

    def emit(
        self,
        event_type: str,
        *,
        block_id: str | None = None,
        data: dict[str, Any] | None = None,
    ) -> None:
        with self._lock:
            self._sequence += 1
            event = {
                "schema_version": 1,
                "sequence": self._sequence,
                "type": event_type,
                "episode_id": self.episode_id,
                "task_id": self.task_id,
                "block_id": block_id,
                "recorded_at": datetime.now(UTC).isoformat(),
                "data": data or {},
            }
            try:
                self._sink.append(event)
            except Exception as exc:
                self._errors.append(f"{type(exc).__name__}: {exc}")
