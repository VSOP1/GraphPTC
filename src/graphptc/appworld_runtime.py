from __future__ import annotations

import json
import queue
import subprocess
import threading
from collections.abc import Callable, Sequence
from typing import Any

from codecell import BaseRuntime, CodeResult
from codecell.python import PythonValidator


class AppWorldProgramRuntime(BaseRuntime):
    """Task-scoped IPC client whose worker owns one persistent AppWorld shell."""

    def __init__(
        self,
        *,
        worker_command: Sequence[str],
        root: str,
        task_id: str,
        experiment_name: str,
        timeout_seconds: float = 100,
    ) -> None:
        super().__init__(PythonValidator())
        self._worker_command = tuple(worker_command)
        self._root = root
        self._task_id = task_id
        self._experiment_name = experiment_name
        self._timeout_seconds = timeout_seconds
        self._process: subprocess.Popen[str] | None = None
        self._lines: queue.Queue[str | None] | None = None
        self._stderr: list[str] = []
        self._lock = threading.Lock()
        self._closed = False
        self._completed = False
        self._executions = 0
        self._timeouts = 0
        self._metadata: dict[str, Any] = {}
        self.last_execution_trace: dict[str, Any] = {}

    def execute(
        self,
        code: str,
        *,
        namespace: dict[str, Callable[..., Any]] | None = None,
        timeout: float | None = None,
    ) -> CodeResult:
        del namespace
        with self._lock:
            if self._closed:
                raise RuntimeError("AppWorld runtime is closed")
            self._ensure_process()
            self._executions += 1
            self._send({"type": "execute", "code": code})
            try:
                message = self._receive(timeout or self._timeout_seconds)
            except TimeoutError:
                self._timeouts += 1
                self._terminate()
                return CodeResult(return_code=-1, timed_out=True)
            if message.get("type") != "execution":
                self._terminate()
                return CodeResult(
                    stderr=f"AppWorld worker protocol error: {message}",
                    return_code=1,
                )
            output = str(message.get("stdout", ""))
            success = bool(message.get("success", False))
            self._completed = bool(message.get("completed", False))
            api_calls = message.get("api_calls", [])
            self.last_execution_trace = {
                "api_calls": api_calls if isinstance(api_calls, list) else [],
                "external_actions": _external_actions(api_calls, success=success),
                "completed": self._completed,
                "output_directory": self._metadata.get("output_directory"),
            }
            if success:
                return CodeResult(stdout=output, return_code=0)
            return CodeResult(stderr=output, return_code=1)

    @property
    def task_completed(self) -> bool:
        return self._completed

    @property
    def metadata(self) -> dict[str, Any]:
        with self._lock:
            if self._closed:
                raise RuntimeError("AppWorld runtime is closed")
            self._ensure_process()
            return dict(self._metadata)

    def evaluate(self) -> dict[str, Any]:
        with self._lock:
            if self._closed:
                raise RuntimeError("AppWorld runtime is closed")
            self._ensure_process()
            self._send({"type": "evaluate"})
            message = self._receive(self._timeout_seconds)
            if message.get("type") != "evaluation":
                raise RuntimeError(f"AppWorld worker protocol error: {message}")
            evaluation = message.get("evaluation", {})
            if not isinstance(evaluation, dict):
                raise RuntimeError("AppWorld worker returned an invalid evaluation")
            return evaluation

    def close(self) -> None:
        with self._lock:
            process = self._process
            if process is None:
                self._closed = True
                return
            if process.poll() is None:
                try:
                    self._send({"type": "close"})
                    self._receive(5)
                    process.wait(timeout=5)
                except (BrokenPipeError, OSError, TimeoutError, subprocess.TimeoutExpired):
                    process.kill()
                    process.wait()
            self._process = None
            self._lines = None
            self._closed = True

    def telemetry(self) -> dict[str, Any]:
        return {
            "runtime": "appworld",
            "task_id": self._task_id,
            "persistent": True,
            "executions": self._executions,
            "timeouts": self._timeouts,
            "task_completed": self._completed,
            "closed": self._closed,
            "metadata": dict(self._metadata),
        }

    def _ensure_process(self) -> None:
        if self._process is not None and self._process.poll() is None:
            return
        process = subprocess.Popen(
            self._worker_command,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        lines: queue.Queue[str | None] = queue.Queue()

        def read_stdout() -> None:
            assert process.stdout is not None
            for line in process.stdout:
                lines.put(line)
            lines.put(None)

        def read_stderr() -> None:
            assert process.stderr is not None
            for line in process.stderr:
                self._stderr.append(line)

        threading.Thread(target=read_stdout, daemon=True).start()
        threading.Thread(target=read_stderr, daemon=True).start()
        self._process = process
        self._lines = lines
        self._send(
            {
                "type": "initialize",
                "root": self._root,
                "task_id": self._task_id,
                "experiment_name": self._experiment_name,
                "timeout_seconds": self._timeout_seconds,
            }
        )
        message = self._receive(60)
        if message.get("type") != "ready":
            self._terminate()
            raise RuntimeError(f"AppWorld worker failed to initialize: {message}")
        self._metadata = {key: value for key, value in message.items() if key != "type"}

    def _send(self, message: dict[str, Any]) -> None:
        process = self._process
        if process is None or process.stdin is None:
            raise RuntimeError("AppWorld worker is unavailable")
        process.stdin.write(json.dumps(message, ensure_ascii=True) + "\n")
        process.stdin.flush()

    def _receive(self, timeout: float) -> dict[str, Any]:
        if self._lines is None:
            raise RuntimeError("AppWorld worker is unavailable")
        try:
            line = self._lines.get(timeout=timeout)
        except queue.Empty as exc:
            raise TimeoutError("AppWorld worker timed out") from exc
        if line is None:
            stderr = "".join(self._stderr[-20:]).strip()
            raise RuntimeError(f"AppWorld worker exited unexpectedly: {stderr}")
        message = json.loads(line)
        if not isinstance(message, dict):
            raise RuntimeError("AppWorld worker returned a non-object message")
        return message

    def _terminate(self) -> None:
        process = self._process
        if process is not None and process.poll() is None:
            process.kill()
            process.wait()
        self._process = None
        self._lines = None


def _external_actions(value: Any, *, success: bool) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    actions: list[dict[str, Any]] = []
    for call in value:
        if not isinstance(call, dict):
            continue
        method = str(call.get("method", "")).lower()
        url = str(call.get("url", ""))
        actions.append(
            {
                "name": f"{method.upper()} {url}".strip(),
                "arguments": dict(call),
                "effect": "read" if method in {"get", "head", "options"} else "write",
                "success": success,
            }
        )
    return actions
