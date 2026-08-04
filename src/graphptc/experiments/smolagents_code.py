from __future__ import annotations

import ast
import json
import time
from collections.abc import Callable
from typing import Any

from smolagents import CodeAgent, OpenAIServerModel, Tool
from smolagents.local_python_executor import CodeOutput, PythonExecutor
from smolagents.monitoring import LogLevel

from ..config import ExperimentConfig
from ..persistent_runtime import PersistentIpcRuntime
from ..ptc import AgentResult, PTCBlockTrace, _analyze_program, _truncate


class SearchTool(Tool):
    name = "search"
    description = (
        "Search the frozen local corpus with BM25. Returns zero to five best-first objects with "
        "exactly docid (string), score (number), and snippet (string); there is no title or URL "
        "field. Scores are comparable only within one search call. A snippet is the beginning of "
        "the document rather than a query-centered passage, so a missing term does not prove the "
        "full document is irrelevant. An empty query raises ValueError."
    )
    inputs = {"query": {"type": "string", "description": "The corpus search query."}}
    output_type = "array"
    output_schema = {
        "type": "array",
        "items": {
            "type": "object",
            "properties": {
                "docid": {"type": "string"},
                "score": {"type": "number"},
                "snippet": {"type": "string"},
            },
            "required": ["docid", "score", "snippet"],
            "additionalProperties": False,
        },
    }

    def __init__(
        self,
        function: Callable[..., list[dict[str, Any]]],
        *,
        expose_output_schema: bool = True,
    ) -> None:
        self._function = function
        if not expose_output_schema:
            self.output_schema = None
        super().__init__()

    def forward(self, query: str) -> list[dict[str, Any]]:
        return self._function(query=query)

    def to_code_prompt(self) -> str:
        if self.output_schema is None:
            return super().to_code_prompt()
        schema = json.dumps(self.output_schema, indent=4)
        return (
            "def search(query: string) -> list[dict]:\n"
            f'    """{self.description}\n\n'
            "    Args:\n"
            "        query: The corpus search query.\n\n"
            "    Returns:\n"
            "        list[dict] adhering to this JSON schema:\n"
            f"{schema}\n"
            '    """'
        )


class FetchTool(Tool):
    name = "fetch"
    description = (
        "Fetch one complete document from the frozen local corpus. Accepts a docid previously "
        "returned by search and returns an object with exactly docid (string) and content "
        "(string). An empty docid raises ValueError and an unknown docid raises KeyError."
    )
    inputs = {"docid": {"type": "string", "description": "A docid returned by search."}}
    output_type = "object"
    output_schema = {
        "type": "object",
        "properties": {
            "docid": {"type": "string"},
            "content": {"type": "string"},
        },
        "required": ["docid", "content"],
        "additionalProperties": False,
    }

    def __init__(self, function: Callable[..., dict[str, Any]]) -> None:
        self._function = function
        super().__init__()

    def forward(self, docid: str) -> dict[str, Any]:
        return self._function(docid=docid)


class PersistentSmolExecutor(PythonExecutor):
    """Run smolagents text actions in the existing task-scoped IPC runtime."""

    def __init__(
        self,
        *,
        runtime: PersistentIpcRuntime,
        search_calls: list[dict[str, Any]] | Callable[[], list[dict[str, Any]]],
        timeout_seconds: float,
        max_stdout_chars: int,
    ) -> None:
        self._runtime = runtime
        self._get_search_calls = (
            search_calls if callable(search_calls) else lambda: search_calls
        )
        self._timeout_seconds = timeout_seconds
        self._max_stdout_chars = max_stdout_chars
        self._tools: dict[str, Tool] = {}
        self._variables: dict[str, Any] = {}
        self.blocks: list[PTCBlockTrace] = []
        self.executions = 0

    def send_tools(self, tools: dict[str, Tool]) -> None:
        self._tools = dict(tools)

    def send_variables(self, variables: dict[str, Any]) -> None:
        self._variables.update(variables)

    def __call__(self, code_action: str) -> CodeOutput:
        started = time.perf_counter()
        calls_before = len(self._get_search_calls())
        final: dict[str, Any] = {"called": False, "value": None}

        namespace: dict[str, Callable[..., Any]] = {}
        for name, tool in self._tools.items():
            if name == "final_answer":
                namespace[name] = self._final_answer_wrapper(tool, final)
            else:
                namespace[name] = self._tool_wrapper(tool)

        normalized_code = _keywordize_tool_calls(code_action, self._tools)
        execution = self._runtime.execute(
            normalized_code,
            namespace=namespace,
            timeout=self._timeout_seconds,
        )
        self.executions += 1
        if execution.return_code != 0:
            message = execution.stderr or "Python action failed"
            raise RuntimeError(message.strip())

        stdout = execution.stdout or ""
        stdout_chars = len(stdout)
        visible_stdout = _truncate(stdout, self._max_stdout_chars)
        runtime_calls = len(self._get_search_calls()) - calls_before
        if runtime_calls or not final["called"]:
            self.blocks.append(
                PTCBlockTrace(
                    turn=self.executions,
                    tool_call_id=None,
                    code=code_action,
                    stdout=visible_stdout,
                    stdout_chars=stdout_chars,
                    stdout_truncated=stdout_chars > self._max_stdout_chars,
                    success=True,
                    duration_ms=(time.perf_counter() - started) * 1_000,
                    invocation_id=None,
                    runtime_calls=runtime_calls,
                    program_analysis=_analyze_program(code_action, {"search", "fetch"}),
                )
            )
        return CodeOutput(
            output=final["value"],
            logs=visible_stdout,
            is_final_answer=bool(final["called"]),
        )

    @staticmethod
    def _tool_wrapper(tool: Tool) -> Callable[..., Any]:
        def call(**kwargs: Any) -> Any:
            return tool(**kwargs)

        call.__name__ = tool.name
        call.__doc__ = tool.description
        return call

    @staticmethod
    def _final_answer_wrapper(
        tool: Tool, final: dict[str, Any]
    ) -> Callable[..., Any]:
        def call(**kwargs: Any) -> Any:
            value = tool(**kwargs)
            final.update(called=True, value=value)
            return value

        call.__name__ = tool.name
        call.__doc__ = tool.description
        return call


class SmolagentsCodeRunner:
    """Stock smolagents CodeAgent with the project's model and runtime budgets."""

    def __init__(
        self,
        config: ExperimentConfig,
        api_key: str,
        search_tools: Any,
        *,
        structured_search_schema: bool = True,
    ) -> None:
        self._config = config
        self._search_tools = search_tools
        self._structured_search_schema = structured_search_schema
        thinking = config.model.thinking
        extra_body = {"thinking": {"type": thinking}} if thinking else None
        model_kwargs: dict[str, Any] = {
            "temperature": config.model.temperature,
            "max_completion_tokens": config.model.max_completion_tokens,
        }
        if config.model.top_p is not None:
            model_kwargs["top_p"] = config.model.top_p
        if extra_body is not None:
            model_kwargs["extra_body"] = extra_body
        self._model = OpenAIServerModel(
            model_id=config.model.model,
            api_base=config.model.base_url,
            api_key=api_key,
            client_kwargs={
                "timeout": config.model.timeout_seconds,
                "max_retries": 0,
            },
            **model_kwargs,
        )

    def run(self, task: str) -> AgentResult:
        started = time.perf_counter()
        ipc_runtime = PersistentIpcRuntime()
        executor = PersistentSmolExecutor(
            runtime=ipc_runtime,
            search_calls=lambda: self._search_tools.calls,
            timeout_seconds=self._config.runtime.code_timeout_seconds,
            max_stdout_chars=self._config.runtime.max_stdout_chars,
        )
        agent = CodeAgent(
            tools=[
                SearchTool(
                    self._search_tools.search,
                    expose_output_schema=self._structured_search_schema,
                ),
                FetchTool(self._search_tools.fetch),
            ],
            model=self._model,
            executor=executor,
            max_steps=self._config.runtime.max_turns,
            max_print_outputs_length=self._config.runtime.max_stdout_chars,
            return_full_result=True,
            verbosity_level=LogLevel.OFF,
        )
        try:
            run_result = agent.run(task, return_full_result=True)
            steps = list(run_result.steps)
            usage = run_result.token_usage
            answer = "" if run_result.output is None else str(run_result.output)
            result = AgentResult(
                answer=answer,
                status="success" if run_result.state == "success" else "failed",
                finish_reason=run_result.state,
                duration_ms=(time.perf_counter() - started) * 1_000,
                model_requests=sum(
                    bool(step.get("model_output") is not None) for step in steps
                ),
                ptc_blocks=len(executor.blocks),
                blocks=list(executor.blocks),
                search_calls=list(self._search_tools.calls),
            )
            if usage is not None:
                result.usage = result.usage.__class__(
                    input_tokens=usage.input_tokens,
                    output_tokens=usage.output_tokens,
                )
        except Exception as exc:
            result = AgentResult(
                status="failed",
                error=f"{type(exc).__name__}: {exc}",
                duration_ms=(time.perf_counter() - started) * 1_000,
                ptc_blocks=len(executor.blocks),
                blocks=list(executor.blocks),
                search_calls=list(self._search_tools.calls),
            )
        finally:
            ipc_runtime.close()
        result.runtime_session = ipc_runtime.telemetry()
        return result


class _KeywordToolCalls(ast.NodeTransformer):
    def __init__(self, tool_parameters: dict[str, tuple[str, ...]]) -> None:
        self._tool_parameters = tool_parameters

    def visit_Call(self, node: ast.Call) -> ast.AST:
        self.generic_visit(node)
        if not isinstance(node.func, ast.Name) or not node.args:
            return node
        parameters = self._tool_parameters.get(node.func.id)
        if parameters is None or len(node.args) > len(parameters):
            return node
        existing = {keyword.arg for keyword in node.keywords}
        keywords = [
            ast.keyword(arg=name, value=value)
            for name, value in zip(parameters, node.args, strict=False)
            if name not in existing
        ]
        node.keywords = keywords + node.keywords
        node.args = []
        return node


def _keywordize_tool_calls(code: str, tools: dict[str, Tool]) -> str:
    parameters = {name: tuple(tool.inputs) for name, tool in tools.items()}
    tree = ast.parse(code)
    tree = _KeywordToolCalls(parameters).visit(tree)
    ast.fix_missing_locations(tree)
    return ast.unparse(tree)
