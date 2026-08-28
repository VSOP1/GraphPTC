from __future__ import annotations

import asyncio
import contextlib
import hashlib
import io
import inspect
import json
import keyword
import os
import queue
import re
import signal
import sys
import threading
import time
from collections.abc import Callable, Mapping, Sequence
from contextlib import AsyncExitStack
from typing import Any

from codecell import BaseRuntime, CodeResult
from codecell.python import PythonValidator
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


class MCPClientSession:
    """Own one stdio MCP session on a dedicated asyncio thread."""

    def __init__(
        self,
        *,
        command: str,
        args: Sequence[str],
        env: Mapping[str, str] | None = None,
        timeout_seconds: float = 120.0,
    ) -> None:
        self._command = command
        self._args = list(args)
        self._env = {**os.environ, **dict(env or {})}
        self._timeout_seconds = timeout_seconds
        self._requests: queue.Queue[tuple[str, Any, queue.Queue[Any]]] = queue.Queue()
        self._ready = threading.Event()
        self._thread = threading.Thread(target=self._thread_main, daemon=True)
        self._startup_error: BaseException | None = None
        self._closed = False
        self._thread.start()
        # session.initialize() has the same timeout. Allow its async context a
        # short grace period to terminate the stdio child before reporting a
        # startup failure to the caller.
        if not self._ready.wait(timeout_seconds + 10.0):
            raise TimeoutError("MCP session initialization timed out")
        if self._startup_error is not None:
            raise RuntimeError(
                f"MCP session initialization failed: {type(self._startup_error).__name__}: "
                f"{self._startup_error}"
            ) from self._startup_error

    def list_tools(self) -> list[dict[str, Any]]:
        return self._request("list_tools", None)

    def call_tool(self, name: str, arguments: Mapping[str, Any]) -> dict[str, Any]:
        return self._request("call_tool", (name, dict(arguments)))

    def close(self) -> None:
        if self._closed:
            return
        try:
            self._request("close", None)
        finally:
            self._closed = True
            self._thread.join(timeout=10)

    def _request(self, operation: str, payload: Any) -> Any:
        if self._closed and operation != "close":
            raise RuntimeError("MCP session is closed")
        response: queue.Queue[Any] = queue.Queue(maxsize=1)
        self._requests.put((operation, payload, response))
        try:
            value = response.get(timeout=self._timeout_seconds)
        except queue.Empty as exc:
            raise TimeoutError(f"MCP {operation} timed out") from exc
        if isinstance(value, BaseException):
            raise value
        return value

    def _thread_main(self) -> None:
        try:
            asyncio.run(self._serve())
        except BaseException as exc:  # pragma: no cover - startup/platform failures
            self._startup_error = exc
            self._ready.set()

    async def _serve(self) -> None:
        try:
            from mcp import ClientSession, StdioServerParameters
            from mcp.client.stdio import stdio_client
        except ImportError as exc:
            raise RuntimeError(
                "MCPMark runtime requires the isolated mcp==1.12.1 environment"
            ) from exc
        params = StdioServerParameters(
            command=self._command,
            args=self._args,
            env=self._env,
        )
        async with AsyncExitStack() as stack:
            read, write = await stack.enter_async_context(stdio_client(params))
            session = await stack.enter_async_context(ClientSession(read, write))
            await asyncio.wait_for(session.initialize(), timeout=self._timeout_seconds)
            self._ready.set()
            while True:
                operation, payload, response = await asyncio.to_thread(self._requests.get)
                try:
                    if operation == "close":
                        response.put(None)
                        return
                    if operation == "list_tools":
                        result = await asyncio.wait_for(
                            session.list_tools(), timeout=self._timeout_seconds
                        )
                        response.put(
                            [tool.model_dump(mode="json") for tool in result.tools]
                        )
                    elif operation == "call_tool":
                        name, arguments = payload
                        result = await asyncio.wait_for(
                            session.call_tool(name, arguments),
                            timeout=self._timeout_seconds,
                        )
                        response.put(result.model_dump(mode="json"))
                    else:
                        raise ValueError(f"unknown MCP operation: {operation}")
                except BaseException as exc:
                    response.put(exc)


class MCPMarkProgramRuntime(BaseRuntime):
    """One task-scoped persistent Python namespace backed by dynamic MCP tools."""

    def __init__(
        self,
        client: MCPClientSession,
        tools: Sequence[Mapping[str, Any]],
    ) -> None:
        super().__init__(PythonValidator())
        self._client = client
        self._tools = [dict(tool) for tool in tools]
        self._globals: dict[str, Any] = {"__builtins__": _safe_builtins()}
        self._calls: list[dict[str, Any]] = []
        self._closed = False
        self._executions = 0
        self._wrapper_to_tool: dict[str, str] = {}
        self._functions = self._build_functions()
        self.last_execution_trace: dict[str, Any] = {}

    @property
    def functions(self) -> tuple[Callable[..., Any], ...]:
        return tuple(self._functions)

    @property
    def tool_manifest(self) -> list[dict[str, Any]]:
        return [
            {
                "wrapper": wrapper,
                "mcp_tool": original,
                "description": next(
                    (str(t.get("description", "")) for t in self._tools if t.get("name") == original),
                    "",
                ),
                "input_schema": next(
                    (t.get("inputSchema", {}) for t in self._tools if t.get("name") == original),
                    {},
                ),
                "annotations": next(
                    (t.get("annotations") for t in self._tools if t.get("name") == original),
                    None,
                ),
            }
            for wrapper, original in self._wrapper_to_tool.items()
        ]

    def execute(
        self,
        code: str,
        *,
        namespace: dict[str, Callable[..., Any]] | None = None,
        timeout: float | None = None,
    ) -> CodeResult:
        if self._closed:
            raise RuntimeError("MCPMark runtime is closed")
        self._executions += 1
        self._globals.update(namespace or {})
        calls_before = len(self._calls)
        stdout = io.StringIO()
        stderr = io.StringIO()
        try:
            with _execution_timeout(timeout), contextlib.redirect_stdout(
                stdout
            ), contextlib.redirect_stderr(stderr):
                exec(compile(code, "<mcpmark-ptc>", "exec"), self._globals, self._globals)
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
            "mcp_calls": recent,
            "external_actions": [
                {
                    "name": call["tool"],
                    "arguments": call["arguments"],
                    "effect": call["effect"],
                    "effect_basis": call["effect_basis"],
                    "success": call["success"],
                    "outcome_unknown": not call["success"],
                    "result_sha256": call["result_sha256"],
                }
                for call in recent
            ],
        }
        return result

    def telemetry(self) -> dict[str, Any]:
        return {
            "runtime": "mcpmark",
            "persistent": True,
            "executions": self._executions,
            "mcp_calls": len(self._calls),
            "tool_counts": _counts(call["tool"] for call in self._calls),
            "failed_mcp_calls": sum(not call["success"] for call in self._calls),
            "namespace_names": sorted(
                name
                for name in self._globals
                if name != "__builtins__" and not name.startswith("_")
            ),
            "closed": self._closed,
        }

    def close(self) -> None:
        if self._closed:
            return
        self._client.close()
        self._closed = True

    def _build_functions(self) -> list[Callable[..., Any]]:
        functions: list[Callable[..., Any]] = []
        used: set[str] = set()
        for tool in self._tools:
            original = str(tool["name"])
            wrapper = _python_name(original)
            if wrapper in used:
                raise ValueError(f"MCP tool wrapper name collision: {wrapper}")
            used.add(wrapper)
            self._wrapper_to_tool[wrapper] = original
            annotations = tool.get("annotations") or {}
            read_only = bool(annotations.get("readOnlyHint", False))
            effect = "read" if read_only else "write"
            basis = "mcp_annotation" if "readOnlyHint" in annotations else "conservative_unknown"

            input_schema = tool.get("inputSchema") or {}
            functions.append(
                self._make_wrapper(wrapper, original, effect, basis, input_schema)
            )
        return functions

    def _make_wrapper(
        self,
        wrapper: str,
        tool: str,
        effect: str,
        basis: str,
        input_schema: Mapping[str, Any],
    ) -> Callable[..., Any]:
        def invoke(**arguments: Any) -> Any:
            started = time.perf_counter()
            success = False
            value: dict[str, Any] | None = None
            try:
                value = self._client.call_tool(tool, arguments)
                success = not bool(value.get("isError", False))
                if not success:
                    raise RuntimeError(_tool_error(value))
                return value
            finally:
                serialized = json.dumps(
                    value, ensure_ascii=False, sort_keys=True, default=repr
                )
                self._calls.append(
                    {
                        "index": len(self._calls) + 1,
                        "tool": tool,
                        "arguments": _redact(arguments),
                        "effect": effect,
                        "effect_basis": basis,
                        "success": success,
                        "duration_ms": (time.perf_counter() - started) * 1_000,
                        "result_chars": len(serialized),
                        "result_sha256": hashlib.sha256(serialized.encode()).hexdigest(),
                    }
                )

        invoke.__name__ = wrapper
        invoke.__qualname__ = wrapper
        invoke.__signature__ = _wrapper_signature(input_schema)  # type: ignore[attr-defined]
        return invoke


def create_mcp_client(
    service: str,
    service_config: Mapping[str, Any],
    *,
    timeout_seconds: float,
    commands: Mapping[str, str] | None = None,
    npm_cache_dir: str = "",
    npm_dependency_cutoff: str = "",
    postgres_pip_constraints: str = "",
) -> tuple[MCPClientSession, dict[str, Any]]:
    command, args, env = _official_server_spec(
        service,
        service_config,
        commands=commands,
        npm_cache_dir=npm_cache_dir,
        npm_dependency_cutoff=npm_dependency_cutoff,
        postgres_pip_constraints=postgres_pip_constraints,
    )
    client = MCPClientSession(
        command=command,
        args=args,
        env=env,
        timeout_seconds=timeout_seconds,
    )
    return client, {"command": command, "args": args, "env_keys": sorted(env)}


def _official_server_spec(
    service: str,
    config: Mapping[str, Any],
    *,
    commands: Mapping[str, str] | None = None,
    npm_cache_dir: str = "",
    npm_dependency_cutoff: str = "",
    postgres_pip_constraints: str = "",
) -> tuple[str, list[str], dict[str, str]]:
    executables = {"npx": "npx", "pipx": "pipx", "docker": "docker"}
    executables.update(commands or {})
    npx_env = _command_path_env(executables["npx"])
    if npm_cache_dir:
        npx_env["NPM_CONFIG_CACHE"] = npm_cache_dir
    if npm_dependency_cutoff:
        npx_env["NPM_CONFIG_BEFORE"] = npm_dependency_cutoff
    if service == "notion":
        key = str(config.get("notion_key", ""))
        if not key:
            raise ValueError("Notion API key required")
        return (
            executables["npx"],
            ["-y", "@notionhq/notion-mcp-server@1.9.1"],
            {
                **npx_env,
                "OPENAPI_MCP_HEADERS": json.dumps(
                    {"Authorization": f"Bearer {key}", "Notion-Version": "2022-06-28"}
                )
            },
        )
    if service == "filesystem":
        root = str(config.get("test_directory", ""))
        if not root:
            raise ValueError("Filesystem test directory required")
        return (
            executables["npx"],
            ["-y", "@modelcontextprotocol/server-filesystem@2025.12.18", root],
            npx_env,
        )
    if service in {"playwright", "playwright_webarena"}:
        args = ["-y", "@playwright/mcp@0.0.68"]
        if bool(config.get("headless", True)):
            args.append("--headless")
        args.extend(
            [
                "--isolated",
                "--no-sandbox",
                "--browser",
                str(config.get("browser", "chromium")),
                "--viewport-size",
                f"{int(config.get('viewport_width', 1280))},{int(config.get('viewport_height', 720))}",
            ]
        )
        return executables["npx"], args, npx_env
    if service == "postgres":
        database = config.get("current_database") or config.get("database")
        required = {
            "username": config.get("username"),
            "password": config.get("password"),
            "database": database,
        }
        if not all(required.values()):
            raise ValueError("PostgreSQL username, password, and database required")
        uri = (
            f"postgresql://{required['username']}:{required['password']}@"
            f"{config.get('host', 'localhost')}:{config.get('port', 5432)}/{database}"
        )
        env = {"DATABASE_URI": uri}
        if postgres_pip_constraints:
            env["PIP_CONSTRAINT"] = postgres_pip_constraints
        return (
            executables["pipx"],
            ["run", "postgres-mcp==0.3.0", "--access-mode=unrestricted"],
            env,
        )
    if service == "github":
        token = str(config.get("github_token", ""))
        if not token:
            raise ValueError("GitHub token required")
        return (
            executables["docker"],
            [
                "run",
                "-i",
                "--rm",
                "-e",
                "GITHUB_PERSONAL_ACCESS_TOKEN",
                "ghcr.io/github/github-mcp-server:v0.15.0",
            ],
            {"GITHUB_PERSONAL_ACCESS_TOKEN": token},
        )
    raise ValueError(f"unsupported MCPMark service: {service}")


def _command_path_env(command: str) -> dict[str, str]:
    directory = os.path.dirname(command)
    if not directory:
        return {}
    return {"PATH": directory + os.pathsep + os.environ.get("PATH", "")}


def _python_name(value: str) -> str:
    name = re.sub(r"\W", "_", value)
    if not name or name[0].isdigit() or keyword.iskeyword(name):
        name = f"mcp_{name}"
    return name


def _wrapper_signature(schema: Mapping[str, Any]) -> inspect.Signature:
    properties = schema.get("properties") or {}
    required = set(schema.get("required") or ())
    parameters = []
    for name, property_schema in properties.items():
        default = (
            inspect.Parameter.empty
            if name in required
            else property_schema.get("default")
        )
        parameters.append(
            inspect.Parameter(
                str(name),
                inspect.Parameter.KEYWORD_ONLY,
                default=default,
                annotation=_schema_annotation(property_schema),
            )
        )
    return inspect.Signature(parameters)


def _schema_annotation(schema: Mapping[str, Any]) -> Any:
    value = schema.get("type")
    if isinstance(value, list):
        value = next((item for item in value if item != "null"), None)
    return {
        "string": str,
        "integer": int,
        "number": float,
        "boolean": bool,
        "array": list,
        "object": dict,
    }.get(value, Any)


def _safe_builtins() -> dict[str, Any]:
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
            raise ImportError(f"module {name!r} is unavailable in MCPMark PTC")
        return __import__(name, fromlist=fromlist)

    values["__import__"] = safe_import
    return values


@contextlib.contextmanager
def _execution_timeout(seconds: float | None):
    if seconds is None or seconds <= 0:
        yield
        return

    if not hasattr(signal, "SIGALRM") or threading.current_thread() is not threading.main_thread():
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


def _tool_error(value: Mapping[str, Any]) -> str:
    return json.dumps(value.get("content", value), ensure_ascii=False, default=repr)[:2_000]


def _counts(values: Any) -> dict[str, int]:
    result: dict[str, int] = {}
    for value in values:
        result[str(value)] = result.get(str(value), 0) + 1
    return result


_SECRET_KEYS = {"authorization", "password", "secret", "token", "api_key"}


def _redact(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {
            str(key): (
                "<redacted>"
                if str(key).lower().replace("-", "_") in _SECRET_KEYS
                or str(key).lower().endswith(("_token", "_password", "_secret", "_api_key"))
                else _redact(item)
            )
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [_redact(item) for item in value]
    return value
