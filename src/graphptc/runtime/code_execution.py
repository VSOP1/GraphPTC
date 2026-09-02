from __future__ import annotations

import contextlib
import signal
import sys
import threading
import time
from collections.abc import Iterator, Sequence
from typing import Any


_SAFE_MODULES = {
    "collections",
    "csv",
    "datetime",
    "decimal",
    "fractions",
    "functools",
    "itertools",
    "json",
    "math",
    "re",
    "statistics",
    "string",
    "textwrap",
    "urllib.parse",
}


def safe_builtins() -> dict[str, Any]:
    import builtins

    allowed = {
        "abs",
        "all",
        "any",
        "bool",
        "dict",
        "enumerate",
        "filter",
        "float",
        "int",
        "isinstance",
        "len",
        "list",
        "map",
        "max",
        "min",
        "next",
        "print",
        "range",
        "repr",
        "reversed",
        "round",
        "set",
        "slice",
        "sorted",
        "str",
        "sum",
        "tuple",
        "zip",
        "Exception",
        "RuntimeError",
        "ValueError",
        "TypeError",
    }
    values = {name: getattr(builtins, name) for name in allowed}

    def safe_import(
        name: str,
        globals: Any = None,
        locals: Any = None,
        fromlist: Sequence[str] = (),
        level: int = 0,
    ) -> Any:
        del globals, locals
        if level or name not in _SAFE_MODULES:
            raise ImportError(f"module {name!r} is unavailable in this PTC runtime")
        return __import__(name, fromlist=fromlist)

    values["__import__"] = safe_import
    return values


@contextlib.contextmanager
def execution_timeout(seconds: float | None) -> Iterator[None]:
    if seconds is None or seconds <= 0:
        yield
        return

    if (
        not hasattr(signal, "SIGALRM")
        or threading.current_thread() is not threading.main_thread()
    ):
        deadline = time.monotonic() + seconds
        previous = sys.gettrace()

        def check_deadline(frame: Any, event: str, arg: Any) -> Any:
            del frame, event, arg
            if time.monotonic() >= deadline:
                raise TimeoutError(f"PTC block timed out after {seconds:g} seconds")
            return check_deadline

        sys.settrace(check_deadline)
        try:
            yield
        finally:
            sys.settrace(previous)
        return

    def expired(signum: int, frame: Any) -> None:
        del signum, frame
        raise TimeoutError(f"PTC block timed out after {seconds:g} seconds")

    previous = signal.signal(signal.SIGALRM, expired)
    signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous)
