from __future__ import annotations

import json
import queue
import subprocess
import threading
from collections.abc import Callable, Mapping, Sequence
from typing import Any

from codecell import BaseRuntime, CodeResult
from codecell.python import PythonValidator


class AgentDiffProgramRuntime(BaseRuntime):
    """Trial-scoped IPC client for Agent-Diff's official Python executor."""

    def __init__(
        self,
        *,
        worker_command: Sequence[str],
        task: Mapping[str, Any],
        trial: int,
        official_commit: str,
        timeout_seconds: float = 30,
    ) -> None:
        super().__init__(PythonValidator())
        self._worker_command = tuple(worker_command)
        self._task = dict(task)
        self._trial = int(trial)
        self._official_commit = official_commit
        self._timeout_seconds = timeout_seconds
        self._process: subprocess.Popen[str] | None = None
        self._lines: queue.Queue[str | None] | None = None
        self._stderr: list[str] = []
        self._lock = threading.Lock()
        self._closed = False
        self._broken_error: str | None = None
        self._close_error: str | None = None
        self._termination_confirmed = False
        self._environment_deleted = False
        self._executions = 0
        self._timeouts = 0
        self._metadata: dict[str, Any] = {}
        self._official_diff: dict[str, Any] = {}
        self._official_result: dict[str, Any] = {}
        self._official_end_run: dict[str, Any] = {}
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
            self._require_usable()
            self._ensure_process()
            self.last_execution_trace = {}
            self._executions += 1
            execution_timeout = self._timeout_seconds if timeout is None else timeout
            try:
                self._send({"type": "execute", "code": code, "timeout": execution_timeout})
                response = self._receive(execution_timeout + min(5.0, max(0.5, execution_timeout * 0.05)))
            except TimeoutError:
                self._timeouts += 1
                self._mark_broken("worker timed out", failure_type="timeout")
                return CodeResult(return_code=-1, timed_out=True)
            except (BrokenPipeError, OSError, RuntimeError, json.JSONDecodeError) as exc:
                message = f"{type(exc).__name__}: {exc}"
                self._mark_broken(message, failure_type="worker_failure")
                return CodeResult(stderr=message, return_code=1)
            if response.get("type") != "execution":
                message = f"Agent-Diff worker protocol error: {response}"
                self._mark_broken(message, failure_type="protocol_error")
                return CodeResult(stderr=message, return_code=1)
            success = response.get("status") == "success" and int(response.get("exit_code", 1)) == 0
            self.last_execution_trace = {
                "external_actions": _redact(response.get("external_actions", [])),
                "state_effects": _bounded_state_effects(response.get("state_effects", [])),
                "python_state_persistent": False,
                "effects_source": "static HTTP call extraction; official diff reserved for final evaluation",
            }
            if not success:
                message = str(response.get("stderr") or response.get("error") or "execution failed")
                self.last_execution_trace["failure"] = {
                    "type": "execution_error",
                    "message": message[:500],
                }
            return CodeResult(
                stdout=str(response.get("stdout", "")),
                stderr=str(response.get("stderr", "")),
                return_code=int(response.get("exit_code", 0 if success else 1)),
            )

    @property
    def task_completed(self) -> bool:
        return False

    @property
    def fatal_error(self) -> str | None:
        return self._broken_error

    @property
    def metadata(self) -> dict[str, Any]:
        with self._lock:
            self._require_usable()
            self._ensure_process()
            return dict(self._metadata)

    @property
    def official_diff(self) -> dict[str, Any]:
        return dict(self._official_diff)

    @property
    def official_result(self) -> dict[str, Any]:
        return dict(self._official_result)

    @property
    def official_end_run(self) -> dict[str, Any]:
        return dict(self._official_end_run)

    def evaluate(self) -> dict[str, Any]:
        with self._lock:
            self._require_usable()
            self._ensure_process()
            self._send({"type": "evaluate"})
            response = self._receive(self._timeout_seconds)
            if response.get("type") != "evaluation":
                raise RuntimeError(f"Agent-Diff worker protocol error: {response}")
            evaluation = response.get("evaluation")
            if not isinstance(evaluation, dict):
                raise RuntimeError("Agent-Diff worker returned an invalid evaluation")
            diff = response.get("official_diff")
            self._official_diff = dict(diff) if isinstance(diff, dict) else {}
            result = response.get("official_result")
            self._official_result = dict(result) if isinstance(result, dict) else {}
            end_run = response.get("official_end_run")
            self._official_end_run = dict(end_run) if isinstance(end_run, dict) else {}
            return dict(evaluation)

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
                    response = self._receive(15)
                    if response.get("type") != "closed":
                        raise RuntimeError(f"Agent-Diff worker close protocol error: {response}")
                    self._environment_deleted = bool(response.get("environment_deleted"))
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
            "runtime": "agent_diff",
            "task_id": str(self._task.get("test_id", "")),
            "trial": self._trial,
            "persistent_python_state": False,
            "persistent_service_state": True,
            "executions": self._executions,
            "timeouts": self._timeouts,
            "closed": self._closed,
            "environment_deleted": self._environment_deleted,
            "termination_confirmed": self._termination_confirmed,
            "close_error": self._close_error,
            "broken": self._broken_error is not None,
            "broken_error": self._broken_error,
            "metadata": dict(self._metadata),
        }

    def _require_usable(self) -> None:
        if self._closed:
            raise RuntimeError("Agent-Diff runtime is closed")
        if self._broken_error is not None:
            self._broken_trace(self._broken_error, failure_type="broken_runtime")
            raise RuntimeError(f"Agent-Diff runtime is unusable: {self._broken_error}")

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
                "task": self._task,
                "trial": self._trial,
                "official_commit": self._official_commit,
                "timeout_seconds": self._timeout_seconds,
            }
        )
        response = self._receive(120)
        if response.get("type") != "ready":
            self._terminate()
            raise RuntimeError(f"Agent-Diff worker failed to initialize: {response}")
        self._metadata = {key: value for key, value in response.items() if key != "type"}

    def _send(self, message: dict[str, Any]) -> None:
        if self._process is None or self._process.stdin is None:
            raise RuntimeError("Agent-Diff worker is unavailable")
        self._process.stdin.write(json.dumps(message, ensure_ascii=True) + "\n")
        self._process.stdin.flush()

    def _receive(self, timeout: float) -> dict[str, Any]:
        if self._lines is None:
            raise RuntimeError("Agent-Diff worker is unavailable")
        try:
            line = self._lines.get(timeout=timeout)
        except queue.Empty as exc:
            raise TimeoutError("Agent-Diff worker timed out") from exc
        if line is None:
            stderr = "".join(self._stderr[-20:]).strip()
            raise RuntimeError(f"Agent-Diff worker exited unexpectedly: {stderr}")
        response = json.loads(line)
        if not isinstance(response, dict):
            raise RuntimeError("Agent-Diff worker returned a non-object message")
        if response.get("type") == "error":
            raise RuntimeError(str(response.get("error")))
        return response

    def _terminate(self) -> bool:
        process = self._process
        try:
            if process is not None and process.poll() is None:
                process.kill()
                process.wait(timeout=5)
        except (OSError, ProcessLookupError, subprocess.TimeoutExpired):
            pass
        finally:
            self._process = None
            self._lines = None
        return process is None or process.poll() is not None

    def _mark_broken(self, message: str, *, failure_type: str) -> None:
        self._broken_error = message
        self._broken_trace(message, failure_type=failure_type)
        self._termination_confirmed = self._terminate()

    def _broken_trace(self, message: str, *, failure_type: str) -> None:
        self.last_execution_trace = {
            "external_actions": [],
            "state_effects": [],
            "effects_unknown": True,
            "failure": {"type": failure_type, "message": message[:500]},
        }


def _bounded_state_effects(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    effects: list[dict[str, Any]] = []
    for item in value[:100]:
        if not isinstance(item, Mapping):
            continue
        effects.append(
            {
                "diff_type": str(item.get("diff_type", ""))[:32],
                "entity": str(item.get("entity", ""))[:120],
                "count": int(item.get("count", 0)),
            }
        )
    return effects


_SENSITIVE_KEYS = {"authorization", "api_key", "password", "secret", "token", "access_token"}


def _redact(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {
            str(key): "<redacted>"
            if str(key).lower().replace("-", "_") in _SENSITIVE_KEYS
            else _redact(item)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [_redact(item) for item in value]
    return value
