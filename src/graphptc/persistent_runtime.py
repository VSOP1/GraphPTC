from __future__ import annotations

import json
import queue
import subprocess
import sys
import threading
import time
from collections.abc import Callable
from typing import Any

from codecell import BaseRuntime, CodeResult, truncate
from codecell.python import PythonValidator


class PersistentIpcRuntime(BaseRuntime):
    """Task-scoped Python subprocess with persistent globals and IPC tools."""

    def __init__(self) -> None:
        super().__init__(PythonValidator())
        self._process: subprocess.Popen[str] | None = None
        self._lines: queue.Queue[str | None] | None = None
        self._lock = threading.Lock()
        self.last_state: dict[str, str] = {}
        self._process_starts = 0
        self._executions = 0
        self._timeouts = 0
        self._protocol_errors = 0
        self._tool_calls = 0
        self._closed = False

    def execute(
        self,
        code: str,
        *,
        namespace: dict[str, Callable[..., Any]] | None = None,
        timeout: float | None = None,
    ) -> CodeResult:
        self._validator.validate(code)
        with self._lock:
            self._executions += 1
            process, lines = self._ensure_process()
            tool_docs = {
                name: getattr(function, "__doc__", None)
                for name, function in (namespace or {}).items()
            }
            self._send(
                process,
                {"type": "execute", "tools": tool_docs, "code": code},
            )
            deadline = time.monotonic() + timeout if timeout else None
            while True:
                remaining = None if deadline is None else deadline - time.monotonic()
                if remaining is not None and remaining <= 0:
                    self._timeouts += 1
                    self._terminate()
                    return CodeResult(return_code=-1, timed_out=True)
                try:
                    line = lines.get(timeout=remaining)
                except queue.Empty:
                    self._timeouts += 1
                    self._terminate()
                    return CodeResult(return_code=-1, timed_out=True)
                if line is None:
                    stderr = process.stderr.read() if process.stderr else ""
                    return_code = process.poll()
                    self._terminate()
                    return CodeResult(
                        stderr=truncate(stderr),
                        return_code=return_code if return_code is not None else -1,
                    )
                message = json.loads(line)
                message_type = message.get("type")
                if message_type == "call":
                    self._dispatch_call(process, message, namespace or {})
                    continue
                if message_type == "done":
                    self.last_state = dict(message.get("state", {}))
                    return CodeResult(
                        stdout=truncate(str(message.get("stdout", ""))),
                        stderr=truncate(str(message.get("stderr", ""))),
                        return_code=int(message.get("rc", 1)),
                    )
                self._terminate()
                self._protocol_errors += 1
                return CodeResult(
                    stderr=f"Persistent runtime protocol error: {message}",
                    return_code=1,
                )

    def close(self) -> None:
        with self._lock:
            process = self._process
            if process is None:
                return
            if process.poll() is None:
                try:
                    self._send(process, {"type": "close"})
                    process.wait(timeout=2)
                except (BrokenPipeError, OSError, subprocess.TimeoutExpired):
                    process.kill()
                    process.wait()
            self._process = None
            self._lines = None
            self._closed = True

    def telemetry(self) -> dict[str, Any]:
        return {
            "persistent": True,
            "process_starts": self._process_starts,
            "executions": self._executions,
            "timeouts": self._timeouts,
            "protocol_errors": self._protocol_errors,
            "tool_calls": self._tool_calls,
            "closed": self._closed,
            "state": dict(self.last_state),
        }

    def _ensure_process(
        self,
    ) -> tuple[subprocess.Popen[str], queue.Queue[str | None]]:
        if self._process is not None and self._process.poll() is None:
            assert self._lines is not None
            return self._process, self._lines
        process = subprocess.Popen(
            [sys.executable, "-m", "graphptc.persistent_worker"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        self._process_starts += 1
        self._closed = False
        lines: queue.Queue[str | None] = queue.Queue()

        def read_lines() -> None:
            assert process.stdout is not None
            for line in process.stdout:
                lines.put(line)
            lines.put(None)

        threading.Thread(target=read_lines, daemon=True).start()
        self._process = process
        self._lines = lines
        return process, lines

    def _dispatch_call(
        self,
        process: subprocess.Popen[str],
        message: dict[str, Any],
        namespace: dict[str, Callable[..., Any]],
    ) -> None:
        self._tool_calls += 1
        tool_name = str(message.get("tool", ""))
        try:
            function = namespace[tool_name]
            value = function(**message.get("kwargs", {}))
            response = {"type": "result", "value": value}
        except Exception as exc:
            response = {"type": "result", "error": f"{type(exc).__name__}: {exc}"}
        self._send(process, response)

    @staticmethod
    def _send(process: subprocess.Popen[str], message: dict[str, Any]) -> None:
        if process.stdin is None:
            raise RuntimeError("Persistent runtime stdin is unavailable")
        process.stdin.write(json.dumps(message, ensure_ascii=True) + "\n")
        process.stdin.flush()

    def _terminate(self) -> None:
        process = self._process
        if process is not None and process.poll() is None:
            process.kill()
            process.wait()
        self._process = None
        self._lines = None
        self.last_state = {}
