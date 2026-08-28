from __future__ import annotations

import json
import os
import queue
import subprocess
import threading
from collections.abc import Callable, Mapping, Sequence
from typing import Any

from codecell import BaseRuntime, CodeResult
from codecell.python import PythonValidator


class DeepPlanningProgramRuntime(BaseRuntime):
    """Task-scoped persistent Python runtime backed by official DeepPlanning tools."""

    def __init__(self, *, worker_command: Sequence[str], request: Mapping[str, Any], timeout_seconds: float) -> None:
        super().__init__(PythonValidator())
        self._command = tuple(worker_command)
        self._request = dict(request)
        self._timeout = timeout_seconds
        self._process: subprocess.Popen[str] | None = None
        self._lines: queue.Queue[str | None] | None = None
        self._stderr: list[str] = []
        self._lock = threading.Lock()
        self._executions = 0
        self._timeouts = 0
        self._closed = False
        self._broken_error: str | None = None
        self._metadata: dict[str, Any] = {}
        self._calls: list[dict[str, Any]] = []
        self.last_execution_trace: dict[str, Any] = {}

    def execute(self, code: str, *, namespace: dict[str, Callable[..., Any]] | None = None, timeout: float | None = None) -> CodeResult:
        del namespace
        self._validator.validate(code)
        with self._lock:
            self._ensure_process()
            self._executions += 1
            self.last_execution_trace = {}
            effective = self._timeout if timeout is None else timeout
            try:
                self._send({"type": "execute", "code": code})
                response = self._receive(effective)
            except TimeoutError:
                self._timeouts += 1
                self._broken_error = "official DeepPlanning worker timed out"
                self._terminate()
                self.last_execution_trace = {"failure": {"type": "timeout", "message": self._broken_error}}
                return CodeResult(return_code=-1, timed_out=True)
            if response.get("type") != "execution":
                self._broken_error = f"official DeepPlanning worker protocol error: {response}"
                self._terminate()
                return CodeResult(stderr=self._broken_error, return_code=1)
            actions = list(response.get("external_actions", []))
            self._calls.extend(actions)
            self.last_execution_trace = {
                "external_actions": actions,
                "state_effects": list(response.get("state_effects", [])),
                "artifacts": list(response.get("artifacts", [])),
                "python_state_persistent": True,
            }
            rc = int(response.get("rc", 1))
            if rc:
                self.last_execution_trace["failure"] = {
                    "type": "execution_error",
                    "message": str(response.get("stderr", ""))[:500],
                }
            return CodeResult(stdout=str(response.get("stdout", "")), stderr=str(response.get("stderr", "")), return_code=rc)

    @property
    def metadata(self) -> dict[str, Any]:
        with self._lock:
            self._ensure_process()
            return dict(self._metadata)

    @property
    def fatal_error(self) -> str | None:
        return self._broken_error

    def telemetry(self) -> dict[str, Any]:
        return {
            "runtime": "deepplanning",
            "persistent": True,
            "executions": self._executions,
            "tool_calls": len(self._calls),
            "failed_tool_calls": sum(not call.get("success", False) for call in self._calls),
            "timeouts": self._timeouts,
            "closed": self._closed,
            "broken_error": self._broken_error,
            "tool_counts": _counts(str(call.get("tool", "")) for call in self._calls),
        }

    def close(self) -> None:
        with self._lock:
            if self._process is None:
                self._closed = True
                return
            try:
                if self._process.poll() is None:
                    self._send({"type": "close"})
                    self._receive(5)
                    self._process.wait(timeout=5)
            except Exception:
                self._terminate()
            self._process = None
            self._lines = None
            self._closed = True

    def _ensure_process(self) -> None:
        if self._process is not None and self._process.poll() is None:
            return
        env = dict(os.environ)
        env["PYTHONUTF8"] = "1"
        process = subprocess.Popen(
            list(self._command), stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            text=True, encoding="utf-8", env=env,
        )
        lines: queue.Queue[str | None] = queue.Queue()
        threading.Thread(target=_read_lines, args=(process.stdout, lines), daemon=True).start()
        threading.Thread(target=_read_stderr, args=(process.stderr, self._stderr), daemon=True).start()
        self._process, self._lines = process, lines
        self._send({"type": "initialize", **self._request})
        response = self._receive(120)
        if response.get("type") != "ready":
            self._terminate()
            raise RuntimeError(f"DeepPlanning worker failed to initialize: {response}; stderr={''.join(self._stderr)[-1000:]}")
        self._metadata = {key: value for key, value in response.items() if key != "type"}

    def _send(self, payload: dict[str, Any]) -> None:
        assert self._process is not None and self._process.stdin is not None
        self._process.stdin.write(json.dumps(payload, ensure_ascii=True) + "\n")
        self._process.stdin.flush()

    def _receive(self, timeout: float) -> dict[str, Any]:
        assert self._lines is not None
        try:
            line = self._lines.get(timeout=timeout)
        except queue.Empty as exc:
            raise TimeoutError from exc
        if line is None:
            raise RuntimeError(f"DeepPlanning worker exited: {''.join(self._stderr)[-1000:]}")
        return json.loads(line)

    def _terminate(self) -> None:
        if self._process is not None and self._process.poll() is None:
            self._process.kill()
            self._process.wait()
        self._process = None
        self._lines = None


def _read_lines(stream: Any, output: queue.Queue[str | None]) -> None:
    if stream is not None:
        for line in stream:
            output.put(line)
    output.put(None)


def _read_stderr(stream: Any, output: list[str]) -> None:
    if stream is not None:
        for line in stream:
            output.append(line)


def _counts(values: Any) -> dict[str, int]:
    result: dict[str, int] = {}
    for value in values:
        result[value] = result.get(value, 0) + 1
    return result
