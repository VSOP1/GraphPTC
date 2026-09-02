from __future__ import annotations

import ast
import contextlib
import hashlib
import io
import json
import os
import subprocess
import sys
import time
import traceback
from pathlib import Path
from types import SimpleNamespace
from typing import Any

SRC_ROOT = Path(__file__).resolve().parents[3]
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from graphptc.graph.adaptation import GoalGraphAdaptation


def _backend(value: str):
    from tool_sandbox.common.tool_discovery import ToolBackend

    if value.lower() != "default":
        raise ValueError(f"unsupported ToolSandbox backend: {value!r}")
    return ToolBackend.DEFAULT


def _inspect(request: dict[str, Any]) -> dict[str, Any]:
    from importlib.metadata import version

    from tool_sandbox.cli.utils import resolve_scenarios

    root = Path(request["root"])
    scenarios = resolve_scenarios(None, _backend(request["tool_backend"]))
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=root, text=True, capture_output=True, check=True
    ).stdout.strip()
    return {
        "type": "inspection",
        "version": version("tool-sandbox"),
        "git_commit": commit,
        "scenario_count": len(scenarios),
        "scenario_names": list(scenarios),
        "scenario_categories": {
            name: [str(value) for value in scenario.categories]
            for name, scenario in scenarios.items()
        },
    }


class MiMoUser:
    """Official ToolSandbox user role with only its model transport replaced."""

    def __new__(cls, model_config: dict[str, Any], api_key: str):
        from openai import OpenAI
        from tool_sandbox.roles.openai_api_user import OpenAIAPIUser

        class Role(OpenAIAPIUser):
            model_name = str(model_config["model"])

            def __init__(self) -> None:
                self.openai_client = OpenAI(
                    api_key=api_key,
                    base_url=model_config.get("base_url"),
                    max_retries=int(model_config.get("max_retries", 3)),
                    timeout=float(model_config.get("timeout_seconds", 600.0)),
                )

        role = Role()
        return role


class ToolSandboxAgent:
    def __init__(self, request: dict[str, Any], api_key: str) -> None:
        from openai import OpenAI
        from tool_sandbox.common.execution_context import RoleType

        self.role_type = RoleType.AGENT
        self._config = request["agent_model"]
        self._runtime = request["runtime"]
        self._client = OpenAI(
            api_key=api_key,
            base_url=self._config.get("base_url"),
            max_retries=int(self._config.get("max_retries", 3)),
            timeout=float(self._config.get("timeout_seconds", 600.0)),
        )
        self._overlay_prompt = str(request["system_prompt"])
        self._demos = list(request["demonstration_messages"])
        self._mode = str(request.get("agent_mode", "ptc"))
        if self._mode not in {"ptc", "direct_tools"}:
            raise ValueError(f"unsupported ToolSandbox agent mode: {self._mode!r}")
        self._ptc_spec = request.get("ptc_tool_spec")
        if self._mode == "ptc" and not isinstance(self._ptc_spec, dict):
            raise ValueError("ToolSandbox PTC mode requires ptc_tool_spec")
        self._graph_mode = request["graph_adaptation_mode"] == "generic"
        if self._mode == "direct_tools" and self._graph_mode:
            raise ValueError("ToolSandbox direct-tools mode cannot enable graph control")
        self._controller: GoalGraphAdaptation | None = None
        self._messages: list[dict[str, Any]] = []
        self._official_system = ""
        self._pending: dict[str, dict[str, Any]] = {}
        self._block_count = 0
        self._direct_call_count = 0
        self._direct_trace: list[dict[str, Any]] = []
        self._execution_failures = 0
        self._model_requests = 0
        self._usage = {"input_tokens": 0, "output_tokens": 0}

    def get_messages(self, ending_index: int | None = None):
        from tool_sandbox.roles.base_role import BaseRole

        return BaseRole.get_messages(ending_index=ending_index)

    def add_messages(self, messages):
        from tool_sandbox.roles.base_role import BaseRole

        return BaseRole.add_messages(messages)

    def get_available_tools(self):
        from tool_sandbox.common.execution_context import RoleType, get_current_context

        return {
            name: function
            for name, function in get_current_context()
            .get_available_tools(scrambling_allowed=True)
            .items()
            if RoleType.AGENT in getattr(function, "visible_to", (RoleType.AGENT,))
        }

    def respond(self, ending_index: int | None = None) -> None:
        from tool_sandbox.common.execution_context import RoleType, get_current_context
        from tool_sandbox.common.message_conversion import Message
        from tool_sandbox.common.tool_conversion import convert_to_openai_tools

        visible = [
            message
            for message in self.get_messages(ending_index=ending_index)
            if RoleType.AGENT in message.visible_to
        ]
        incoming = visible[-1]
        if incoming.recipient != RoleType.AGENT:
            raise KeyError("latest visible message is not addressed to the agent")
        if incoming.sender == RoleType.SYSTEM:
            self._official_system = incoming.content
            return

        available_tools = self.get_available_tools()
        context = get_current_context()
        for agent_name, function in available_tools.items():
            context.interactive_console.locals[agent_name] = function
        tool_schemas = convert_to_openai_tools(available_tools)
        if self._controller is None:
            task = incoming.content if incoming.sender == RoleType.USER else "ToolSandbox scenario"
            self._controller = GoalGraphAdaptation(
                {}, {}, task=task, expose_graph_api=False
            ) if self._graph_mode else None
            self._messages = [json.loads(json.dumps(item)) for item in self._demos]
            if self._controller is not None:
                self._messages.append({"role": "user", "content": self._controller.initial_observation()})

        if incoming.sender == RoleType.USER:
            self._messages.append({"role": "user", "content": incoming.content})
        elif incoming.sender == RoleType.EXECUTION_ENVIRONMENT:
            responses = [
                message
                for message in visible
                if message.sender == RoleType.EXECUTION_ENVIRONMENT
                and message.openai_tool_call_id in self._pending
            ]
            if not responses:
                raise RuntimeError("execution response has no pending tool call")
            for response_message in responses:
                self._observe_execution(response_message)
        else:
            raise ValueError(f"unsupported incoming role: {incoming.sender}")

        system = self._official_system + "\n\n" + self._overlay_prompt
        if self._mode == "ptc":
            system += (
                "\n\nScenario function schemas (authoritative for this scenario):\n"
                + json.dumps(tool_schemas, ensure_ascii=False, separators=(",", ":"))
            )
        response = self._client.chat.completions.create(
            model=self._config["model"],
            messages=[{"role": "system", "content": system}, *self._messages],
            tools=tool_schemas if self._mode == "direct_tools" else [self._ptc_spec],
            tool_choice="auto",
            max_tokens=int(self._config.get("max_completion_tokens", 32000)),
            temperature=self._config.get("temperature"),
            top_p=self._config.get("top_p"),
            extra_body=(
                {"thinking": {"type": self._config["thinking"]}}
                if self._config.get("thinking")
                else None
            ),
        )
        self._model_requests += 1
        if response.usage is not None:
            self._usage["input_tokens"] += int(response.usage.prompt_tokens or 0)
            self._usage["output_tokens"] += int(response.usage.completion_tokens or 0)
        message = response.choices[0].message
        calls = list(message.tool_calls or ())
        assistant: dict[str, Any] = {"role": "assistant", "content": message.content}
        if calls:
            if self._mode == "direct_tools":
                self._emit_direct_calls(
                    calls=calls,
                    assistant=assistant,
                    available_tool_names=set(available_tools),
                    context=context,
                    role_type=RoleType,
                    message_type=Message,
                )
                return
            if len(calls) != 1 or calls[0].function.name != "programmatic_tool_call":
                raise ValueError("ToolSandbox PTC requires exactly one programmatic_tool_call per model turn")
            call = calls[0]
            payload = json.loads(call.function.arguments)
            code = payload.get("code")
            if not isinstance(code, str) or not code.strip():
                raise ValueError("programmatic_tool_call.code must be non-empty Python source")
            if self._controller is not None:
                self._controller.prepare_program_action(payload)
            call_id = _code_call_id(code)
            assistant["tool_calls"] = [
                {
                    "id": call_id,
                    "type": "function",
                    "function": {
                        "name": call.function.name,
                        "arguments": call.function.arguments,
                    },
                }
            ]
            self._messages.append(assistant)
            self._pending[call_id] = {
                "id": call_id,
                "code": code,
                "mode": "ptc",
                "started": time.perf_counter(),
                "state_before": _state_snapshot(context),
                "locals_before": _local_types(context.interactive_console.locals),
            }
            self.add_messages(
                [
                    Message(
                        sender=RoleType.AGENT,
                        recipient=RoleType.EXECUTION_ENVIRONMENT,
                        content=code,
                        openai_tool_call_id=call_id,
                        openai_function_name="programmatic_tool_call",
                    )
                ]
            )
            return

        text = (message.content or "").strip()
        if not text:
            raise ValueError("agent model returned neither code nor a user response")
        self._messages.append(assistant)
        self.add_messages([Message(sender=RoleType.AGENT, recipient=RoleType.USER, content=text)])

    def _emit_direct_calls(
        self,
        *,
        calls: list[Any],
        assistant: dict[str, Any],
        available_tool_names: set[str],
        context: Any,
        role_type: Any,
        message_type: Any,
    ) -> None:
        from tool_sandbox.common.message_conversion import (
            openai_tool_call_to_python_code,
        )

        assistant["tool_calls"] = [
            {
                "id": call.id,
                "type": "function",
                "function": {
                    "name": call.function.name,
                    "arguments": call.function.arguments,
                },
            }
            for call in calls
        ]
        outgoing = []
        for call in calls:
            execution_name = context.get_execution_facing_tool_name(call.function.name)
            code = openai_tool_call_to_python_code(
                call,
                available_tool_names,
                execution_facing_tool_name=execution_name,
            )
            self._pending[call.id] = {
                "id": call.id,
                "name": call.function.name,
                "code": code,
                "mode": "direct_tools",
                "started": time.perf_counter(),
                "state_before": _state_snapshot(context),
                "locals_before": _local_types(context.interactive_console.locals),
            }
            outgoing.append(
                message_type(
                    sender=role_type.AGENT,
                    recipient=role_type.EXECUTION_ENVIRONMENT,
                    content=code,
                    openai_tool_call_id=call.id,
                    openai_function_name=call.function.name,
                )
            )
        self._messages.append(assistant)
        self.add_messages(outgoing)

    def _observe_execution(self, incoming: Any) -> None:
        from tool_sandbox.common.execution_context import get_current_context

        call_id = incoming.openai_tool_call_id
        if call_id not in self._pending:
            raise RuntimeError("execution response has no pending tool call")
        pending = self._pending.pop(call_id)
        self._block_count += 1
        raw = incoming.content or ""
        limit = int(self._runtime.get("max_stdout_chars", 8000))
        truncated = len(raw) > limit
        shown = raw[:limit] + ("\n...[stdout truncated]" if truncated else "")
        success = incoming.tool_call_exception is None
        if not success:
            self._execution_failures += 1
        context = get_current_context()
        state_after = _state_snapshot(context)
        locals_after = _local_types(context.interactive_console.locals)
        actions = _external_actions(incoming.tool_trace or (), state_after != pending["state_before"], success)
        content = shown or ("Execution successful." if success else "Execution failed.")
        if pending["mode"] == "direct_tools":
            self._direct_call_count += 1
            self._direct_trace.append(
                {
                    "turn": self._model_requests,
                    "tool_call_id": pending["id"],
                    "tool": pending["name"],
                    "success": success,
                    "duration_ms": (time.perf_counter() - pending["started"]) * 1000,
                    "observation_chars": len(raw),
                    "observation_truncated": truncated,
                    "runtime_calls": len(actions),
                }
            )
            self._messages.append(
                {"role": "tool", "tool_call_id": pending["id"], "content": content}
            )
            return
        trace = SimpleNamespace(
            turn=self._model_requests,
            tool_call_id=pending["id"],
            code=pending["code"],
            stdout=shown,
            stdout_chars=len(raw),
            stdout_truncated=truncated,
            success=success,
            duration_ms=(time.perf_counter() - pending["started"]) * 1000,
            invocation_id=None,
            runtime_calls=len(actions),
            program_analysis=_program_analysis(pending["code"]),
            runtime_trace={
                "external_actions": actions,
                "state_before": pending["locals_before"],
                "state_after": locals_after,
                "loaded_names": _loaded_names(pending["code"]),
                "stored_names": _stored_names(pending["code"]),
                "error_location": "execution_environment" if not success else None,
            },
            error_type="ToolSandboxExecutionError" if not success else None,
            error_message=incoming.tool_call_exception,
        )
        if self._controller is not None:
            content += "\n\n" + self._controller.observe(trace)
        self._messages.append({"role": "tool", "tool_call_id": pending["id"], "content": content})

    def finish(self, answered: bool) -> None:
        if self._controller is not None:
            self._controller.finish(answered=answered)

    def graph_artifact(self) -> dict[str, Any] | None:
        return self._controller.graph_artifact() if self._controller is not None else None

    def telemetry(self) -> dict[str, Any]:
        return {
            "mode": (
                "direct_tool_calling" if self._mode == "direct_tools" else "programmatic_tool_calling"
            ),
            "model_requests": self._model_requests,
            "ptc_blocks": self._block_count,
            "direct_tool_calls": self._direct_call_count,
            "tool_call_trace": list(self._direct_trace),
            "execution_failures": self._execution_failures,
            "usage": dict(self._usage),
            "graph": self._controller.telemetry() if self._controller is not None else None,
        }

    def teardown(self) -> None:
        close = getattr(self._client, "close", None)
        if callable(close):
            close()


class ToolSandboxExecutionEnvironment:
    """Official persistent shell response path for PTC or native tool calls."""

    role_type = None

    def __init__(self) -> None:
        from tool_sandbox.common.execution_context import RoleType
        from tool_sandbox.roles.execution_environment import ExecutionEnvironment

        self.role_type = RoleType.EXECUTION_ENVIRONMENT
        self._official = ExecutionEnvironment()

    def respond(self, ending_index: int | None = None) -> None:
        import polars as pl
        from attrs import evolve
        from tool_sandbox.common.execution_context import (
            DatabaseNamespace,
            get_current_context,
        )
        from tool_sandbox.roles.execution_environment import (
            get_messages_to_process,
            respond_to_messages,
        )

        messages = self._official.get_messages(ending_index=ending_index)
        self._official.messages_validation(messages)
        pending = get_messages_to_process(messages, recipient=self.role_type)
        if not pending:
            raise ValueError("ToolSandbox execution received no tool messages")
        context = get_current_context()
        responses = respond_to_messages(
            interactive_console=context.interactive_console,
            messages=pending,
            role_type=self.role_type,
        )
        traces = context.get_database(DatabaseNamespace.SANDBOX)["tool_trace"][0]
        if traces is not None:
            trace_list = traces.to_list()
            context.update_database(
                DatabaseNamespace.SANDBOX,
                context.get_database(DatabaseNamespace.SANDBOX).with_columns(
                    pl.lit(None).alias("tool_trace")
                ),
            )
            if responses and responses[0].tool_call_exception is None:
                responses[0] = evolve(responses[0], tool_trace=trace_list)
        self._official.add_messages(responses)

    def teardown(self) -> None:
        self._official.teardown()


def _run(request: dict[str, Any]) -> dict[str, Any]:
    from openai.types.chat.chat_completion_message_tool_call import (
        ChatCompletionMessageToolCall,
        Function,
    )
    from tool_sandbox.cli.utils import resolve_scenarios
    from tool_sandbox.common import message_conversion
    from tool_sandbox.common.execution_context import DatabaseNamespace, RoleType

    api_key = os.environ.get(str(request["agent_model"].get("api_key_env", "MIMO_API_KEY")))
    user_key = os.environ.get(str(request["user_model"].get("api_key_env", "MIMO_API_KEY")))
    if not api_key or not user_key:
        raise RuntimeError("agent or user-model API key is unavailable in the isolated worker")
    if not os.environ.get("RAPID_API_KEY"):
        raise RuntimeError("RAPID_API_KEY is unavailable in the isolated worker")
    name = str(request["scenario_name"])
    scenario = resolve_scenarios([name], _backend(request["tool_backend"]))[name]
    original_parser = message_conversion.python_code_to_openai_tool_call

    def parse_program(code: str, function_name: str | None):
        if function_name != "programmatic_tool_call":
            return original_parser(code, function_name)
        return ChatCompletionMessageToolCall(
            id=_code_call_id(code),
            type="function",
            function=Function(
                name="programmatic_tool_call",
                arguments=json.dumps({"code": code}, ensure_ascii=False),
            ),
        )

    if request.get("agent_mode", "ptc") == "ptc":
        message_conversion.python_code_to_openai_tool_call = parse_program
    agent = ToolSandboxAgent(request, api_key)
    roles = {
        RoleType.USER: MiMoUser(request["user_model"], user_key),
        RoleType.EXECUTION_ENVIRONMENT: ToolSandboxExecutionEnvironment(),
        RoleType.AGENT: agent,
    }
    try:
        result = scenario.play_and_evaluate(
            roles=roles,
            output_directory=Path(request["output_directory"]),
            scenario_name=name,
        )
        context = result.ending_context
        active = bool(
            context.get_database(
                DatabaseNamespace.SANDBOX,
                drop_sandbox_message_index=False,
                get_all_history_snapshots=True,
            )["conversation_active"][-1]
        )
        agent.finish(answered=not active)
        graph = agent.graph_artifact()
        if graph is not None:
            graph_path = Path(request["graph_path"])
            graph_path.parent.mkdir(parents=True, exist_ok=True)
            graph_path.write_text(json.dumps(graph, ensure_ascii=False, indent=2, default=repr), encoding="utf-8")
        evaluation = result.evaluation_result
        telemetry = agent.telemetry()
        return {
            "type": "result",
            "official_commit": request["official_commit"],
            "categories": [str(value) for value in scenario.categories],
            "similarity": evaluation.similarity,
            "milestone_similarity": evaluation.milestone_similarity,
            "minefield_similarity": evaluation.minefield_similarity,
            "turn_count": evaluation.turn_count,
            "milestone_mapping": dict(evaluation.milestone_mapping),
            "minefield_mapping": dict(evaluation.minefield_mapping),
            "conversation_active": active,
            "execution_failures": telemetry["execution_failures"],
            "agent": telemetry,
            "graph_path": request["graph_path"] if graph is not None else None,
        }
    finally:
        message_conversion.python_code_to_openai_tool_call = original_parser
        for role in roles.values():
            role.teardown()


def _state_snapshot(context: Any) -> dict[str, str]:
    from tool_sandbox.common.execution_context import DatabaseNamespace

    result: dict[str, str] = {}
    for namespace in DatabaseNamespace:
        if namespace == DatabaseNamespace.SANDBOX:
            continue
        payload = context.get_database(namespace).write_json(row_oriented=True)
        result[str(namespace)] = hashlib.sha256(payload.encode()).hexdigest()[:16]
    return result


def _code_call_id(code: str) -> str:
    return "ptc_" + hashlib.sha256(code.encode()).hexdigest()[:24]


def _local_types(values: dict[str, Any]) -> dict[str, str]:
    return {
        name: type(value).__name__
        for name, value in values.items()
        if not name.startswith("__") and not callable(value)
    }


def _external_actions(traces: tuple[str, ...] | list[str], changed: bool, success: bool) -> list[dict[str, Any]]:
    actions: list[dict[str, Any]] = []
    for value in traces:
        try:
            item = json.loads(value)
        except (TypeError, json.JSONDecodeError):
            item = {"tool_name": "", "arguments": {}, "result": value}
        actions.append(
            {
                "name": item.get("tool_name", ""),
                "arguments": item.get("arguments", {}),
                "result": item.get("result"),
                "effect": "write" if changed else "read",
                "success": success,
                "effect_basis": "official_state_delta",
            }
        )
    return actions


def _syntax(code: str) -> ast.AST | None:
    try:
        return ast.parse(code)
    except SyntaxError:
        return None


def _loaded_names(code: str) -> list[str]:
    tree = _syntax(code)
    return sorted({node.id for node in ast.walk(tree) if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load)}) if tree else []


def _stored_names(code: str) -> list[str]:
    tree = _syntax(code)
    return sorted({node.id for node in ast.walk(tree) if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store)}) if tree else []


def _program_analysis(code: str) -> dict[str, Any]:
    tree = _syntax(code)
    if tree is None:
        return {"syntax_error": True, "tool_call_count": 0, "transform_count": 0, "control_dependency_count": 0}
    return {
        "syntax_error": False,
        "tool_call_count": sum(isinstance(node, ast.Call) for node in ast.walk(tree)),
        "transform_count": sum(isinstance(node, (ast.ListComp, ast.SetComp, ast.DictComp, ast.GeneratorExp)) for node in ast.walk(tree)),
        "control_dependency_count": sum(isinstance(node, (ast.If, ast.For, ast.While, ast.Try)) for node in ast.walk(tree)),
    }


def main() -> int:
    try:
        request = json.loads(sys.stdin.read())
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            response = _inspect(request) if request.get("type") == "inspect" else _run(request)
    except Exception as exc:
        response = {"type": "error", "error": f"{type(exc).__name__}: {exc}", "traceback": traceback.format_exc()}
    sys.stdout.write(json.dumps(response, ensure_ascii=False, default=repr))
    return 0 if response.get("type") != "error" else 1


if __name__ == "__main__":
    raise SystemExit(main())
