from __future__ import annotations

import builtins
import dis
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


def _call_site(frame: Any) -> dict[str, int] | None:
    return _position_at_offset(frame.f_code, frame.f_lasti)


def _position_at_offset(code: Any, offset: int) -> dict[str, int] | None:
    try:
        instructions = dis.get_instructions(code, show_caches=True)
        current = None
        for instruction in instructions:
            if instruction.offset > offset:
                break
            if instruction.positions.lineno is not None:
                current = instruction
    except Exception:
        return None
    if current is None:
        return None
    positions = current.positions
    if positions.lineno is None or positions.col_offset is None:
        return None
    result = {
        "line": int(positions.lineno),
        "column": int(positions.col_offset),
    }
    if positions.end_lineno is not None:
        result["end_line"] = int(positions.end_lineno)
    if positions.end_col_offset is not None:
        result["end_column"] = int(positions.end_col_offset)
    return result


def _exception_location(exc: Exception) -> dict[str, int] | None:
    location = None
    current = exc.__traceback__
    while current is not None:
        if current.tb_frame.f_code.co_filename == "<code_execution>":
            candidate = _position_at_offset(
                current.tb_frame.f_code,
                current.tb_lasti,
            )
            if candidate is not None:
                location = candidate
        current = current.tb_next
    return location


def _state_tracer() -> tuple[
    Callable[..., Any],
    set[str],
    set[str],
    set[tuple[int, int, int, int]],
]:
    loaded: set[str] = set()
    stored: set[str] = set()
    executed_spans: set[tuple[int, int, int, int]] = set()
    instruction_cache: dict[Any, dict[int, dis.Instruction]] = {}

    def trace(frame: Any, event: str, arg: Any) -> Any:
        try:
            if frame.f_code.co_filename != "<code_execution>":
                return trace if event == "call" else None
            if event == "call":
                frame.f_trace_opcodes = True
                return trace
            if event != "opcode":
                return trace
            instructions = instruction_cache.setdefault(
                frame.f_code,
                {
                    instruction.offset: instruction
                    for instruction in dis.get_instructions(
                        frame.f_code,
                        show_caches=True,
                    )
                },
            )
            instruction = instructions.get(frame.f_lasti)
            if instruction is None:
                return trace
            positions = instruction.positions
            if (
                positions.lineno is not None
                and positions.col_offset is not None
                and positions.end_lineno is not None
                and positions.end_col_offset is not None
            ):
                executed_spans.add(
                    (
                        int(positions.lineno),
                        int(positions.col_offset),
                        int(positions.end_lineno),
                        int(positions.end_col_offset),
                    )
                )
            if not isinstance(instruction.argval, str):
                return trace
            if instruction.opname in {"LOAD_NAME", "LOAD_GLOBAL", "LOAD_DEREF"}:
                loaded.add(instruction.argval)
            elif instruction.opname in {
                "STORE_NAME",
                "STORE_GLOBAL",
                "STORE_DEREF",
            }:
                stored.add(instruction.argval)
        except Exception:
            return trace
        return trace

    return trace, loaded, stored, executed_spans


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
        observe = request.get("observe") is True
        state_before = _state_manifest(namespace, tool_names)

        def make_stub(tool_name: str, tool_doc: str | None) -> Callable[..., Any]:
            def stub(**kwargs: Any) -> Any:
                message = {"type": "call", "tool": tool_name, "kwargs": kwargs}
                if observe:
                    call_site = _call_site(sys._getframe(1))
                    if call_site is not None:
                        message["call_site"] = call_site
                send(message)
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
        trace = None
        loaded_names: set[str] = set()
        stored_names: set[str] = set()
        executed_spans: set[tuple[int, int, int, int]] = set()
        error_location = None
        previous_trace = sys.gettrace()
        try:
            if observe:
                trace, loaded_names, stored_names, executed_spans = _state_tracer()
                sys.settrace(trace)
            exec(compile(request["code"], "<code_execution>", "exec"), namespace)
        except Exception as exc:
            return_code = 1
            if observe:
                error_location = _exception_location(exc)
            stderr = traceback.format_exc()
        finally:
            if observe:
                sys.settrace(previous_trace)
            sys.stdout = real_stdout

        state_after = _state_manifest(namespace, tool_names)

        send(
            {
                "type": "done",
                "stdout": captured_stdout.getvalue(),
                "stderr": stderr,
                "rc": return_code,
                "state": state_after,
                "execution_trace": (
                    {
                        "state_before": state_before,
                        "state_after": state_after,
                        "loaded_names": sorted(loaded_names),
                        "stored_names": sorted(stored_names),
                        "executed_spans": [
                            {
                                "line": line,
                                "column": column,
                                "end_line": end_line,
                                "end_column": end_column,
                            }
                            for line, column, end_line, end_column in sorted(
                                executed_spans
                            )
                        ],
                        "error_location": error_location,
                    }
                    if observe
                    else {}
                ),
            }
        )


if __name__ == "__main__":
    main()
