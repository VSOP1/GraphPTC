from __future__ import annotations

import contextlib
import importlib.metadata
import io
import json
import platform
import sys
import time
from collections.abc import Mapping
from typing import Any


def _function_name(source: str) -> str:
    return source.split("def", 1)[1].split("(", 1)[0].strip()


def _call_tool(
    sources: list[str], name: str, arguments: Mapping[str, Any]
) -> dict[str, Any]:
    namespace: dict[str, Any] = {}
    stdout = io.StringIO()
    stderr = io.StringIO()
    started = time.perf_counter()
    try:
        with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            for source in sources:
                if _function_name(source) == name:
                    exec(source, namespace, namespace)
            function = namespace[name]
            result = function(**dict(arguments))
            # The official runner serializes feedback before returning it to the model.
            json.dumps(result, ensure_ascii=False)
        return {
            "type": "tool_result",
            "success": True,
            "result": result,
            "stdout": stdout.getvalue(),
            "stderr": stderr.getvalue(),
            "duration_ms": (time.perf_counter() - started) * 1_000,
        }
    except BaseException as exc:
        return {
            "type": "tool_result",
            "success": False,
            "error_type": type(exc).__name__,
            "error": str(exc),
            "stdout": stdout.getvalue(),
            "stderr": stderr.getvalue(),
            "duration_ms": (time.perf_counter() - started) * 1_000,
        }


def _environment() -> dict[str, Any]:
    packages = {}
    for name in (
        "Babel",
        "dicttoxml",
        "holidays",
        "numpy",
        "python-dateutil",
        "pytz",
        "roman",
        "sympy",
    ):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    return {
        "python": platform.python_version(),
        "executable": sys.executable,
        "packages": packages,
    }


def main() -> int:
    sources: list[str] = []
    for line in sys.stdin:
        if not line.strip():
            continue
        try:
            request = json.loads(line)
            request_type = request.get("type")
            if request_type == "inspect":
                response = {"type": "inspection", "environment": _environment()}
            elif request_type == "initialize":
                sources = [str(value) for value in request.get("functions") or []]
                response = {
                    "type": "initialized",
                    "task_id": str(request.get("task_id")),
                    "function_names": [_function_name(source) for source in sources],
                    "environment": _environment(),
                }
            elif request_type == "call_tool":
                response = _call_tool(
                    sources,
                    str(request.get("name")),
                    request.get("arguments") or {},
                )
            elif request_type == "close":
                print(json.dumps({"type": "closed"}), flush=True)
                return 0
            else:
                response = {"type": "error", "error": "unknown request type"}
        except BaseException as exc:
            response = {
                "type": "error",
                "error": f"{type(exc).__name__}: {exc}",
            }
        print(json.dumps(response, ensure_ascii=True, default=repr), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
