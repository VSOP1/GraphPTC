from __future__ import annotations

import contextlib
import io
import json
import queue
import threading
import time
import traceback
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ToolRequest:
    call_id: str
    name: str
    arguments: dict[str, Any]


@dataclass(frozen=True)
class BlockComplete:
    code: str
    stdout: str
    stdout_chars: int
    stdout_truncated: bool
    success: bool
    duration_ms: float
    calls: tuple[dict[str, Any], ...]
    error_type: str | None = None
    error_message: str | None = None


class Tau3ToolError(RuntimeError):
    pass


class Tau3ProgramRuntime:
    """Suspend one PTC program at tool calls handled by the official orchestrator."""

    def __init__(
        self,
        tool_names: tuple[str, ...],
        *,
        max_stdout_chars: int,
        timeout_seconds: float,
    ) -> None:
        self._tool_names = tuple(tool_names)
        self._max_stdout_chars = max_stdout_chars
        self._timeout_seconds = timeout_seconds
        self._events: queue.Queue[ToolRequest | BlockComplete] = queue.Queue()
        self._responses: queue.Queue[
            tuple[str, bool, bool | None, str | None] | None
        ] = queue.Queue()
        self._thread: threading.Thread | None = None
        self._code = ""
        self._started = 0.0
        self._calls: list[dict[str, Any]] = []
        self._call_count = 0
        self._closed = False

    @property
    def closed(self) -> bool:
        return self._closed

    def start(self, code: str) -> ToolRequest | BlockComplete:
        if self._closed:
            raise RuntimeError("runtime is closed")
        if self._thread is not None and self._thread.is_alive():
            raise RuntimeError("a PTC block is already running")
        if not isinstance(code, str) or not code.strip():
            raise ValueError("PTC code must be a non-empty string")
        self._code = code
        self._started = time.perf_counter()
        self._calls = []
        self._call_count = 0
        self._thread = threading.Thread(target=self._run, name="tau3-ptc", daemon=True)
        self._thread.start()
        return self._next_event()

    def resume(
        self,
        content: str | None,
        *,
        error: bool,
        state_changed: bool | None = None,
        declared_effect: str | None = None,
    ) -> ToolRequest | BlockComplete:
        if self._thread is None or not self._thread.is_alive():
            raise RuntimeError("no PTC block is waiting for a tool result")
        self._responses.put((content or "", error, state_changed, declared_effect))
        return self._next_event()

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._responses.put(None)
        if self._thread is not None:
            self._thread.join(timeout=1)

    def _next_event(self) -> ToolRequest | BlockComplete:
        elapsed = time.perf_counter() - self._started
        remaining = max(0.001, self._timeout_seconds - elapsed)
        try:
            return self._events.get(timeout=remaining)
        except queue.Empty as exc:
            self.close()
            raise TimeoutError("PTC program exceeded its wall-clock timeout") from exc

    def _proxy(self, name: str):
        def call(**arguments: Any) -> Any:
            self._call_count += 1
            call_id = f"tau3-ptc-{self._call_count}"
            record = {
                "id": call_id,
                "name": name,
                "arguments": arguments,
                "success": None,
            }
            self._calls.append(record)
            self._events.put(ToolRequest(call_id, name, arguments))
            response = self._responses.get()
            if response is None:
                record["success"] = False
                raise Tau3ToolError("runtime closed while waiting for a tool result")
            content, error, state_changed, declared_effect = response
            record["success"] = not error
            record["output"] = content
            if declared_effect is not None:
                record["effect"] = declared_effect
            if state_changed is not None:
                record["state_changed"] = state_changed
            if declared_effect is not None or state_changed is not None:
                record["effect_basis"] = "official_tool_metadata_and_db_hash"
            if error:
                raise Tau3ToolError(content or f"{name} failed")
            try:
                return json.loads(content)
            except (TypeError, json.JSONDecodeError):
                return content

        call.__name__ = name
        return call

    def _run(self) -> None:
        output = io.StringIO()
        error_type = None
        error_message = None
        try:
            namespace = {name: self._proxy(name) for name in self._tool_names}
            namespace["__builtins__"] = __builtins__
            with contextlib.redirect_stdout(output):
                exec(  # noqa: S102 - executing the model's PTC program is this runtime's contract
                    compile(self._code, "<tau3-ptc>", "exec"), namespace, namespace
                )
            success = True
        except Exception as exc:  # noqa: BLE001 - program failures become PTC observations
            success = False
            error_type = type(exc).__name__
            error_message = str(exc)
            if not isinstance(exc, Tau3ToolError):
                traceback.print_exc(file=output)
        raw = output.getvalue()
        truncated = len(raw) > self._max_stdout_chars
        shown = raw[: self._max_stdout_chars]
        if truncated:
            shown += "\n...[stdout truncated]"
        self._events.put(
            BlockComplete(
                code=self._code,
                stdout=shown,
                stdout_chars=len(raw),
                stdout_truncated=truncated,
                success=success,
                duration_ms=(time.perf_counter() - self._started) * 1_000,
                calls=tuple(dict(item) for item in self._calls),
                error_type=error_type,
                error_message=error_message,
            )
        )
