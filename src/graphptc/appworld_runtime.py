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
        self._close_error: str | None = None
        self._termination_confirmed = False
        self._broken_error: str | None = None
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
            if self._broken_error is not None:
                self._broken_trace(
                    self._broken_error,
                    failure_type="broken_runtime",
                )
                raise RuntimeError(
                    f"AppWorld runtime is unusable after worker failure: {self._broken_error}"
                )
            self.last_execution_trace = {}
            self._ensure_process()
            self._executions += 1
            try:
                self._send({"type": "execute", "code": code})
                execution_timeout = self._timeout_seconds if timeout is None else timeout
                response_slack = min(5.0, max(0.25, execution_timeout * 0.02))
                message = self._receive(execution_timeout + response_slack)
            except TimeoutError:
                self._timeouts += 1
                self._mark_broken("worker timed out", failure_type="timeout")
                return CodeResult(return_code=-1, timed_out=True)
            except (BrokenPipeError, OSError, RuntimeError, json.JSONDecodeError) as exc:
                message_text = f"{type(exc).__name__}: {exc}"
                self._mark_broken(message_text, failure_type="worker_failure")
                return CodeResult(stderr=message_text, return_code=1)
            if message.get("type") != "execution":
                error = f"AppWorld worker protocol error: {message}"
                self._mark_broken(error, failure_type="protocol_error")
                return CodeResult(
                    stderr=error,
                    return_code=1,
                )
            output = str(message.get("stdout", ""))
            success = bool(message.get("success", False))
            self._completed = bool(message.get("completed", False))
            api_calls = _redact_secrets(message.get("api_calls", []))
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
    def fatal_error(self) -> str | None:
        return self._broken_error

    @property
    def metadata(self) -> dict[str, Any]:
        with self._lock:
            if self._closed:
                raise RuntimeError("AppWorld runtime is closed")
            if not self._metadata:
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
                self._termination_confirmed = True
                return
            try:
                if process.poll() is None:
                    self._send({"type": "close"})
                    message = self._receive(5)
                    if message.get("type") != "closed":
                        raise RuntimeError(f"AppWorld worker close protocol error: {message}")
                    process.wait(timeout=5)
                self._termination_confirmed = process.poll() is not None
            except Exception as exc:
                self._close_error = f"{type(exc).__name__}: {exc}"
                self._termination_confirmed = self._terminate()
            finally:
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
            "termination_confirmed": self._termination_confirmed,
            "close_error": self._close_error,
            "broken": self._broken_error is not None,
            "broken_error": self._broken_error,
            "metadata": dict(self._metadata),
        }

    def _ensure_process(self) -> None:
        if self._broken_error is not None:
            raise RuntimeError(
                f"AppWorld runtime is unusable after worker failure: {self._broken_error}"
            )
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

    def _terminate(self) -> bool:
        process = self._process
        try:
            if process is not None and process.poll() is None:
                try:
                    process.kill()
                except (OSError, ProcessLookupError):
                    pass
                try:
                    process.wait(timeout=5)
                except (OSError, subprocess.TimeoutExpired):
                    pass
        finally:
            self._process = None
            self._lines = None
        return process is None or process.poll() is not None

    def _mark_broken(self, message: str, *, failure_type: str) -> None:
        self._broken_error = message
        self._completed = False
        self._broken_trace(message, failure_type=failure_type)
        self._termination_confirmed = self._terminate()

    def _broken_trace(self, message: str, *, failure_type: str) -> None:
        self.last_execution_trace = {
            "api_calls": [],
            "external_actions": [],
            "api_calls_complete": False,
            "effects_unknown": True,
            "completed": False,
            "output_directory": self._metadata.get("output_directory"),
            "failure": {"type": failure_type, "message": message},
        }


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
                "success": True if success else None,
                "outcome_unknown": not success,
                "effect_basis": "http_method",
            }
        )
    return actions


_SENSITIVE_ARGUMENT_KEYS = {
    "api_key",
    "authorization",
    "cookie",
    "password",
    "refresh_token",
    "secret",
    "token",
    "access_token",
}


def _redact_secrets(value: Any) -> Any:
    if isinstance(value, dict):
        redacted: dict[str, Any] = {}
        for key, item in value.items():
            normalized = str(key).strip().lower().replace("-", "_")
            if normalized in _SENSITIVE_ARGUMENT_KEYS or normalized.endswith(
                ("_password", "_secret", "_token", "_api_key")
            ):
                redacted[str(key)] = "<redacted>"
            else:
                redacted[str(key)] = _redact_secrets(item)
        return redacted
    if isinstance(value, list):
        return [_redact_secrets(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_redact_secrets(item) for item in value)
    return value
