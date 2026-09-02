from __future__ import annotations

import contextlib
import inspect
import io
import json
import os
import queue
import subprocess
import threading
import time
from collections.abc import Callable, Mapping, Sequence
from typing import Any

from codecell import BaseRuntime, CodeResult
from codecell.python import PythonValidator

from ...runtime.code_execution import execution_timeout, safe_builtins


class _OfficialWorkerClient:
    def __init__(
        self,
        *,
        command: Sequence[str],
        root: str,
        task_id: str,
        timeout_seconds: float,
    ) -> None:
        if not command:
            raise ValueError("APIFlow official worker command is required")
        env = dict(os.environ)
        env["PYTHONUTF8"] = "1"
        self._timeout_seconds = timeout_seconds
        self._process = subprocess.Popen(
            tuple(command),
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
            env=env,
        )
        self._lines: queue.Queue[str | None] = queue.Queue()
        self._stderr: list[str] = []

        def read_stdout() -> None:
            assert self._process.stdout is not None
            for line in self._process.stdout:
                self._lines.put(line)
            self._lines.put(None)

        def read_stderr() -> None:
            assert self._process.stderr is not None
            self._stderr.extend(self._process.stderr)

        threading.Thread(target=read_stdout, daemon=True).start()
        threading.Thread(target=read_stderr, daemon=True).start()
        self.metadata = self.request(
            {"type": "initialize", "root": root, "task_id": task_id},
            timeout=max(30.0, timeout_seconds),
        )

    def request(
        self, payload: Mapping[str, Any], *, timeout: float | None = None
    ) -> dict[str, Any]:
        if self._process.poll() is not None:
            raise RuntimeError(f"APIFlow worker exited: {''.join(self._stderr[-20:])}")
        assert self._process.stdin is not None
        self._process.stdin.write(json.dumps(dict(payload), ensure_ascii=True) + "\n")
        self._process.stdin.flush()
        try:
            line = self._lines.get(timeout=timeout or self._timeout_seconds)
        except queue.Empty as exc:
            raise TimeoutError("APIFlow official worker timed out") from exc
        if line is None:
            raise RuntimeError(f"APIFlow worker exited: {''.join(self._stderr[-20:])}")
        response = json.loads(line)
        if not isinstance(response, dict):
            raise RuntimeError("APIFlow worker returned a non-object response")
        if response.get("type") == "error":
            raise RuntimeError(str(response.get("error", "APIFlow worker error")))
        return response

    def close(self) -> None:
        if self._process.poll() is None:
            try:
                self.request({"type": "close"}, timeout=15.0)
                self._process.wait(timeout=5.0)
            except Exception:
                self._process.kill()
                self._process.wait(timeout=5.0)


class APIFlowProgramRuntime(BaseRuntime):
    """Task-scoped persistent PTC runtime backed by APIFlow's official tools."""

    def __init__(
        self,
        *,
        worker_command: Sequence[str],
        root: str,
        task_id: str,
        timeout_seconds: float = 60.0,
    ) -> None:
        super().__init__(PythonValidator())
        self._client = _OfficialWorkerClient(
            command=worker_command,
            root=root,
            task_id=task_id,
            timeout_seconds=timeout_seconds,
        )
        self._globals: dict[str, Any] = {"__builtins__": safe_builtins()}
        self._calls: list[dict[str, Any]] = []
        self._executions = 0
        self._closed = False
        self._task_completed = False
        self._secret_values: set[str] = set()
        self.last_execution_trace: dict[str, Any] = {}
        self._functions = self._build_functions()

    @property
    def metadata(self) -> dict[str, Any]:
        return dict(self._client.metadata)

    @property
    def functions(self) -> tuple[Callable[..., Any], ...]:
        return tuple(self._functions)

    @property
    def task_completed(self) -> bool:
        return self._task_completed

    @property
    def secret_values(self) -> tuple[str, ...]:
        return tuple(sorted(self._secret_values, key=len, reverse=True))

    @property
    def fatal_error(self) -> None:
        return None

    def execute(
        self,
        code: str,
        *,
        namespace: dict[str, Callable[..., Any]] | None = None,
        timeout: float | None = None,
    ) -> CodeResult:
        if self._closed:
            raise RuntimeError("APIFlow runtime is closed")
        self._executions += 1
        self._globals.update(namespace or {})
        calls_before = len(self._calls)
        stdout = io.StringIO()
        stderr = io.StringIO()
        try:
            with execution_timeout(timeout), contextlib.redirect_stdout(
                stdout
            ), contextlib.redirect_stderr(stderr):
                exec(compile(code, "<apiflow-ptc>", "exec"), self._globals, self._globals)
            result = CodeResult(stdout=stdout.getvalue(), stderr=stderr.getvalue())
        except TimeoutError as exc:
            result = CodeResult(
                stdout=stdout.getvalue(), stderr=str(exc), return_code=-1, timed_out=True
            )
        except BaseException as exc:
            result = CodeResult(
                stdout=stdout.getvalue(),
                stderr=f"{type(exc).__name__}: {exc}",
                return_code=1,
            )
        recent = self._calls[calls_before:]
        self.last_execution_trace = {
            "api_calls": recent,
            "external_actions": [
                {
                    "name": call["name"],
                    "arguments": call["arguments"],
                    "effect": call["effect"],
                    "success": call["success"],
                }
                for call in recent
            ],
        }
        if result.return_code != 0:
            self.last_execution_trace["failure"] = {
                "type": "execution_error",
                "message": result.stderr[:500],
            }
        return result

    def evaluate(self, final_answer: str) -> dict[str, Any]:
        return self._client.request(
            {"type": "evaluate", "final_answer": final_answer},
            timeout=max(30.0, self._client._timeout_seconds),
        )

    def telemetry(self) -> dict[str, Any]:
        return {
            "runtime": "apiflow",
            "persistent": True,
            "executions": self._executions,
            "tool_calls": len(self._calls),
            "tool_counts": _counts(call["name"] for call in self._calls),
            "failed_tool_calls": sum(not call["success"] for call in self._calls),
            "task_completed": self._task_completed,
            "closed": self._closed,
        }

    def close(self) -> None:
        if self._closed:
            return
        self._client.close()
        self._closed = True

    def _build_functions(self) -> list[Callable[..., Any]]:
        signatures = {
            "read": [
                ("entity_ref", inspect.Parameter.empty),
                ("scope", None),
            ],
            "write": [
                ("entity_ref", inspect.Parameter.empty),
                ("content", inspect.Parameter.empty),
                ("scope", None),
            ],
            "edit": [
                ("entity_ref", inspect.Parameter.empty),
                ("patch", inspect.Parameter.empty),
            ],
            "search": [("query", inspect.Parameter.empty), ("kind", None)],
            "execute": [("target", None), ("request", None), ("mode", None)],
            "clarify": [("question", inspect.Parameter.empty)],
            "report_blocked": [("reason", inspect.Parameter.empty)],
        }
        functions = []
        for name, parameters in signatures.items():
            function = self._make_wrapper(name)
            function.__signature__ = inspect.Signature(  # type: ignore[attr-defined]
                inspect.Parameter(
                    parameter,
                    inspect.Parameter.KEYWORD_ONLY,
                    default=default,
                )
                for parameter, default in parameters
            )
            functions.append(function)
        return functions

    def _make_wrapper(self, name: str) -> Callable[..., Any]:
        def invoke(**arguments: Any) -> Any:
            started = time.perf_counter()
            success = False
            response: dict[str, Any] | None = None
            try:
                response = self._client.request(
                    {"type": "call_tool", "name": name, "arguments": arguments}
                )
                success = bool(response.get("success"))
                self._task_completed = self._task_completed or bool(
                    response.get("task_completed")
                )
                if not success:
                    raise RuntimeError(str(response.get("error", "APIFlow tool failed")))
                value = response.get("result")
                _collect_secrets(value, self._secret_values)
                return value
            finally:
                self._calls.append(
                    {
                        "index": len(self._calls) + 1,
                        "name": name,
                        "arguments": _redact(arguments),
                        "effect": (response or {}).get("effect", "write"),
                        "success": success,
                        "duration_ms": (time.perf_counter() - started) * 1_000,
                    }
                )

        invoke.__name__ = name
        invoke.__qualname__ = name
        return invoke


def _counts(values: Any) -> dict[str, int]:
    counts: dict[str, int] = {}
    for value in values:
        counts[str(value)] = counts.get(str(value), 0) + 1
    return counts


_SECRET_KEYS = {"authorization", "password", "secret", "token", "api_key"}


def _redact(value: Any) -> Any:
    if isinstance(value, Mapping):
        named_secret = str(value.get("name", "")).lower().endswith(
            ("token", "password", "secret", "api_key")
        )
        return {
            str(key): (
                "<redacted>"
                if str(key).lower().replace("-", "_") in _SECRET_KEYS
                or str(key).lower().endswith(("_token", "_password", "_secret", "_api_key"))
                or (named_secret and str(key).lower() == "value")
                else _redact(item)
            )
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [_redact(item) for item in value]
    return value


def _collect_secrets(value: Any, output: set[str]) -> None:
    if isinstance(value, Mapping):
        named_secret = str(value.get("name", "")).lower().endswith(
            ("token", "password", "secret", "api_key")
        )
        for key, item in value.items():
            normalized = str(key).lower().replace("-", "_")
            sensitive = (
                normalized in _SECRET_KEYS
                or normalized.endswith(("_token", "_password", "_secret", "_api_key"))
                or (named_secret and normalized == "value")
            )
            if sensitive and isinstance(item, (str, int, float)) and str(item):
                output.add(str(item))
            else:
                _collect_secrets(item, output)
    elif isinstance(value, list):
        for item in value:
            _collect_secrets(item, output)
