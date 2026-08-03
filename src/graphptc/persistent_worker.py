from __future__ import annotations

import builtins
import io
import json
import sys
import traceback
from collections.abc import Callable
from types import ModuleType
from typing import Any


def _state_manifest(namespace: dict[str, Any], tool_names: set[str]) -> dict[str, str]:
    return {
        name: type(value).__name__
        for name, value in sorted(namespace.items())
        if not name.startswith("__")
        and name not in tool_names
        and not callable(value)
        and not isinstance(value, ModuleType)
    }


def main() -> None:
    real_stdout = sys.stdout
    real_stdin = sys.stdin
    namespace: dict[str, Any] = {"__builtins__": builtins.__dict__}

    def send(message: dict[str, Any]) -> None:
        real_stdout.write(json.dumps(message, ensure_ascii=True) + "\n")
        real_stdout.flush()

    def receive() -> dict[str, Any]:
        line = real_stdin.readline()
        if not line:
            raise EOFError("Parent process closed the IPC pipe")
        return json.loads(line)

    while True:
        try:
            request = receive()
        except EOFError:
            return
        if request.get("type") == "close":
            return
        if request.get("type") != "execute":
            send({"type": "protocol_error", "error": "Expected execute message"})
            continue

        tool_docs = request.get("tools", {})
        tool_names = set(tool_docs)

        def make_stub(tool_name: str, tool_doc: str | None) -> Callable[..., Any]:
            def stub(**kwargs: Any) -> Any:
                send({"type": "call", "tool": tool_name, "kwargs": kwargs})
                response = receive()
                if response.get("error"):
                    raise RuntimeError(
                        f"Tool {tool_name} failed: {response['error']}"
                    )
                return response["value"]

            stub.__name__ = tool_name
            stub.__qualname__ = tool_name
            stub.__doc__ = tool_doc
            return stub

        namespace.update(
            {
                name: make_stub(name, doc)
                for name, doc in tool_docs.items()
            }
        )
        captured_stdout = io.StringIO()
        sys.stdout = captured_stdout
        return_code = 0
        stderr = ""
        try:
            exec(compile(request["code"], "<code_execution>", "exec"), namespace)
        except Exception:
            return_code = 1
            stderr = traceback.format_exc()
        finally:
            sys.stdout = real_stdout

        send(
            {
                "type": "done",
                "stdout": captured_stdout.getvalue(),
                "stderr": stderr,
                "rc": return_code,
                "state": _state_manifest(namespace, tool_names),
            }
        )


if __name__ == "__main__":
    main()
