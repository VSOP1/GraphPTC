from __future__ import annotations

import contextlib
import ast
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

from .mcpmark_runtime import _execution_timeout, _safe_builtins


_OMITTED = "__GRAPHPTC_TOOLHOP_OMITTED_PARAMETER__"


class _ToolHopWorkerClient:
    def __init__(
        self,
        *,
        command: Sequence[str],
        task_id: str,
        functions: Sequence[str],
        timeout_seconds: float,
    ) -> None:
        if not command:
            raise ValueError("ToolHop official worker command is required")
        self._timeout_seconds = timeout_seconds
        self._stderr: list[str] = []
        self._lines: queue.Queue[str | None] = queue.Queue()
        self._process = subprocess.Popen(
            tuple(command),
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
            env={**os.environ, "PYTHONUTF8": "1"},
        )

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
            {
                "type": "initialize",
                "task_id": task_id,
                "functions": list(functions),
            },
            timeout=max(30.0, timeout_seconds),
        )

    def request(
        self, payload: Mapping[str, Any], *, timeout: float | None = None
    ) -> dict[str, Any]:
        if self._process.poll() is not None:
            raise RuntimeError(f"ToolHop worker exited: {''.join(self._stderr[-20:])}")
        assert self._process.stdin is not None
        self._process.stdin.write(json.dumps(dict(payload), ensure_ascii=True) + "\n")
        self._process.stdin.flush()
        try:
            line = self._lines.get(timeout=timeout or self._timeout_seconds)
        except queue.Empty as exc:
            self.terminate()
            raise TimeoutError("ToolHop official tool call timed out") from exc
        if line is None:
            raise RuntimeError(f"ToolHop worker exited: {''.join(self._stderr[-20:])}")
        response = json.loads(line)
        if not isinstance(response, dict):
            raise RuntimeError("ToolHop worker returned a non-object response")
        if response.get("type") == "error":
            raise RuntimeError(str(response.get("error", "ToolHop worker error")))
        return response

    def terminate(self) -> None:
        if self._process.poll() is None:
            self._process.kill()
            self._process.wait(timeout=5)

    def close(self) -> None:
        if self._process.poll() is not None:
            return
        try:
            self.request({"type": "close"}, timeout=15.0)
            self._process.wait(timeout=5.0)
        except Exception:
            self.terminate()


class ToolHopProgramRuntime(BaseRuntime):
    """Task-scoped persistent PTC namespace backed by ToolHop's official functions."""

    def __init__(
        self,
        *,
        worker_command: Sequence[str],
        task: Mapping[str, Any],
        timeout_seconds: float = 10.0,
    ) -> None:
        super().__init__(PythonValidator())
        self._task_id = str(task["id"])
        self._schemas = list((task.get("tools") or {}).values())
        sources = [str(source) for source in task.get("functions") or ()]
        self._defaults = _function_defaults(sources)
        self._client = _ToolHopWorkerClient(
            command=worker_command,
            task_id=self._task_id,
            functions=sources,
            timeout_seconds=timeout_seconds,
        )
        self._globals: dict[str, Any] = {"__builtins__": _safe_builtins()}
        self._calls: list[dict[str, Any]] = []
        self._executions = 0
        self._closed = False
        self._fatal_error: str | None = None
        self.last_execution_trace: dict[str, Any] = {}
        self._functions = self._build_functions()

    @property
    def metadata(self) -> dict[str, Any]:
        return dict(self._client.metadata)

    @property
    def functions(self) -> tuple[Callable[..., Any], ...]:
        return tuple(self._functions)

    @property
    def calls(self) -> tuple[dict[str, Any], ...]:
        return tuple(self._calls)

    @property
    def task_completed(self) -> bool:
        return False

    @property
    def fatal_error(self) -> str | None:
        return self._fatal_error

    @property
    def last_tool_output(self) -> str | None:
        if not self._calls:
            return None
        return self._calls[-1].get("result_text")

    def execute(
        self,
        code: str,
        *,
        namespace: dict[str, Callable[..., Any]] | None = None,
        timeout: float | None = None,
    ) -> CodeResult:
        if self._closed:
            raise RuntimeError("ToolHop runtime is closed")
        self._executions += 1
        self._globals.update(namespace or {})
        calls_before = len(self._calls)
        stdout = io.StringIO()
        stderr = io.StringIO()
        try:
            with _execution_timeout(timeout), contextlib.redirect_stdout(
                stdout
            ), contextlib.redirect_stderr(stderr):
                exec(compile(code, "<toolhop-ptc>", "exec"), self._globals, self._globals)
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
                    "effect": "read",
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

    def telemetry(self) -> dict[str, Any]:
        return {
            "runtime": "toolhop",
            "task_id": self._task_id,
            "persistent": True,
            "executions": self._executions,
            "tool_calls": len(self._calls),
            "tool_counts": _counts(call["name"] for call in self._calls),
            "failed_tool_calls": sum(not call["success"] for call in self._calls),
            "closed": self._closed,
            "fatal_error": self._fatal_error,
        }

    def close(self) -> None:
        if self._closed:
            return
        self._client.close()
        self._closed = True

    def _build_functions(self) -> list[Callable[..., Any]]:
        by_name: dict[str, Mapping[str, Any]] = {}
        for schema in self._schemas:
            by_name[str(schema["name"])] = schema
        output: list[Callable[..., Any]] = []
        for name, schema in by_name.items():
            function = self._make_wrapper(name)
            parameters = schema.get("parameters") or {}
            properties = parameters.get("properties") or {}
            defaults = self._defaults.get(name, {})
            function.__signature__ = inspect.Signature(  # type: ignore[attr-defined]
                inspect.Parameter(
                    str(parameter),
                    inspect.Parameter.KEYWORD_ONLY,
                    default=defaults.get(parameter, _OMITTED),
                )
                for parameter in properties
            )
            function.__doc__ = str(schema.get("description") or "")
            output.append(function)
        return output

    def _make_wrapper(self, name: str) -> Callable[..., Any]:
        def invoke(**arguments: Any) -> Any:
            arguments = {
                key: value for key, value in arguments.items() if value != _OMITTED
            }
            started = time.perf_counter()
            response: dict[str, Any] | None = None
            success = False
            error: str | None = None
            try:
                response = self._client.request(
                    {"type": "call_tool", "name": name, "arguments": arguments}
                )
                success = bool(response.get("success"))
                if not success:
                    error = (
                        f"{response.get('error_type', 'Error')}: "
                        f"{response.get('error', 'ToolHop tool failed')}"
                    )
                    raise RuntimeError(error)
                return response.get("result")
            except (TimeoutError, BrokenPipeError, OSError) as exc:
                error = f"{type(exc).__name__}: {exc}"
                self._fatal_error = error
                raise
            finally:
                result = (response or {}).get("result")
                self._calls.append(
                    {
                        "index": len(self._calls) + 1,
                        "name": name,
                        "arguments": dict(arguments),
                        "effect": "read",
                        "success": success,
                        "result": result if success else None,
                        "result_text": (
                            json.dumps(result, ensure_ascii=False) if success else None
                        ),
                        "error": error,
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


def _function_defaults(sources: Sequence[str]) -> dict[str, dict[str, Any]]:
    output: dict[str, dict[str, Any]] = {}
    for source in sources:
        tree = ast.parse(source)
        function = next(
            node
            for node in tree.body
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        )
        positional = [*function.args.posonlyargs, *function.args.args]
        values: dict[str, Any] = {}
        if function.args.defaults:
            for argument, default in zip(
                positional[-len(function.args.defaults) :], function.args.defaults
            ):
                values[argument.arg] = ast.literal_eval(default)
        for argument, default in zip(
            function.args.kwonlyargs, function.args.kw_defaults
        ):
            if default is not None:
                values[argument.arg] = ast.literal_eval(default)
        output[function.name] = values
    return output
