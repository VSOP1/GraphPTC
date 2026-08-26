from __future__ import annotations

import ast
import copy
import hashlib
import json
import os
import subprocess
import sys
from collections.abc import Mapping
from dataclasses import dataclass
from importlib.metadata import version
from pathlib import Path
from types import SimpleNamespace
from typing import Any

SRC_ROOT = Path(__file__).resolve().parents[1]
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from graphptc.config import ModelConfig
from graphptc.goal_adaptation import GoalGraphAdaptation
from graphptc.model import OpenAIChatModel, usage_to_dict
from graphptc.tau3_runtime import BlockComplete, Tau3ProgramRuntime, ToolRequest


def _validated_ptc_calls(calls: list[Any]) -> list[Any]:
    return calls


def _ptc_call_error(call: Any) -> str | None:
    if call.name != "programmatic_tool_call":
        return f"Error: unknown tool: {call.name}"
    code = call.input.get("code")
    if not isinstance(code, str) or not code.strip():
        return "Error: programmatic_tool_call.code must be non-empty"
    return None


def _valid_ptc_call_count(calls: list[Any]) -> int:
    return sum(_ptc_call_error(call) is None for call in calls)


def _stamp_trial(simulation: Any, trial: int) -> Any:
    simulation.trial = trial
    return simulation


def _dump(value: Any) -> Any:
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json")
    if isinstance(value, Mapping):
        return {str(key): _dump(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_dump(item) for item in value]
    return value


def _emit(payload: Mapping[str, Any]) -> None:
    print(json.dumps(payload, ensure_ascii=True, default=repr), flush=True)


@dataclass
class Tau3AgentState:
    messages: list[dict[str, Any]]


class GraphPTCTau3Agent:
    """PTC agent speaking the official tau3 HalfDuplexAgent protocol."""

    def __init__(self, tools: list[Any], domain_policy: str, request: Mapping[str, Any]) -> None:
        from tau2.agent.base_agent import HalfDuplexAgent

        # HalfDuplexAgent has no cooperative mixin state beyond these fields.
        HalfDuplexAgent.__init__(self, tools=tools, domain_policy=domain_policy)
        self.tools = tools
        self.domain_policy = domain_policy
        self._request = dict(request)
        self._runtime_config = dict(request["runtime"])
        self._model = OpenAIChatModel(
            ModelConfig(**request["agent_model"]),
            os.environ[str(request["agent_model"].get("api_key_env", "MIMO_API_KEY"))],
        )
        self._ptc_spec = copy.deepcopy(request["ptc_tool_spec"])
        self._demos = copy.deepcopy(list(request["demonstration_messages"]))
        self._system = self._make_system(str(request["system_prompt"]))
        self._controller = (
            GoalGraphAdaptation(
                {}, {}, task=f"tau3:{request.get('domain')}:{request.get('task_id')}",
                expose_graph_api=False, host_inspection_enabled=False,
            )
            if request["graph_adaptation_mode"] == "generic"
            else None
        )
        self._program: Tau3ProgramRuntime | None = None
        self._pending_ptc_id: str | None = None
        self._pending_payload: dict[str, Any] | None = None
        self._pending_blocks: list[Any] = []
        self._blocks = 0
        self._execution_failures = 0
        self._model_requests = 0
        self._usage = {"input_tokens": 0, "output_tokens": 0, "cached_input_tokens": 0}
        self._closed = False
        self._pending_db_hash: str | None = None
        self._pending_tool_name: str | None = None
        self._block_traces: list[dict[str, Any]] = []
        self._scaffold_failures: list[dict[str, Any]] = []
        self._state_messages: list[dict[str, Any]] = []

    def _make_system(self, overlay: str) -> str:
        schemas = [tool.openai_schema for tool in self.tools]
        return (
            "<policy>\n" + self.domain_policy + "\n</policy>\n\n" + overlay
            + "\n\nOfficial environment tool schemas (authoritative):\n"
            + json.dumps(schemas, ensure_ascii=False, separators=(",", ":"))
        )

    def get_init_state(self, message_history: list[Any] | None = None) -> Tau3AgentState:
        messages = copy.deepcopy(self._demos)
        if self._controller is not None:
            messages.append({"role": "user", "content": self._controller.initial_observation()})
        for message in message_history or ():
            messages.append(_official_to_openai(message))
        self._state_messages = messages
        return Tau3AgentState(messages=messages)

    def generate_next_message(self, message: Any, state: Tau3AgentState):
        from tau2.data_model.message import MultiToolMessage, ToolMessage, UserMessage

        if isinstance(message, UserMessage):
            if self._program is not None:
                raise RuntimeError("received a user message while a PTC program was suspended")
            state.messages.append({"role": "user", "content": message.content or ""})
            result = self._model_step(state)
        elif isinstance(message, MultiToolMessage):
            if len(message.tool_messages) != 1:
                raise ValueError("GraphPTC emits one official tool call at a time")
            result = self._resume(message.tool_messages[0], state)
        elif isinstance(message, ToolMessage):
            result = self._resume(message, state)
        else:
            raise TypeError(f"unsupported tau3 agent input: {type(message).__name__}")
        return result, state

    def _model_step(self, state: Tau3AgentState):
        from tau2.data_model.message import AssistantMessage

        self._model_requests += 1
        turn = self._model.create_turn(
            system=self._system,
            messages=state.messages,
            tools=[self._ptc_spec],
            timeout_seconds=float(self._runtime_config["task_timeout_seconds"]),
        )
        usage = usage_to_dict(turn.usage)
        for key in self._usage:
            self._usage[key] += int(usage.get(key, 0))
        state.messages.append(turn.assistant_message)
        if not turn.tool_calls:
            text = (turn.text or "").strip()
            if not text:
                raise ValueError("agent model returned neither PTC code nor user-facing text")
            return AssistantMessage(role="assistant", content=text)
        calls = _validated_ptc_calls(list(turn.tool_calls))
        if self._blocks + _valid_ptc_call_count(calls) > int(
            self._runtime_config["max_ptc_blocks"]
        ):
            raise RuntimeError("PTC block budget exhausted")
        self._pending_blocks = calls
        return self._start_next_block(state)

    def _start_next_block(self, state: Tau3AgentState):
        call = self._pending_blocks.pop(0)
        call_error = _ptc_call_error(call)
        if call_error is not None:
            self._execution_failures += 1
            self._scaffold_failures.append(
                {
                    "model_request": self._model_requests,
                    "tool_call_id": call.id,
                    "tool_name": call.name,
                    "error": call_error,
                }
            )
            state.messages.append(
                {"role": "tool", "tool_call_id": call.id, "content": call_error}
            )
            if self._pending_blocks:
                return self._start_next_block(state)
            return self._model_step(state)
        payload = dict(call.input)
        code = payload.get("code")
        if self._controller is not None:
            self._controller.prepare_program_action(payload)
        self._pending_ptc_id = call.id
        self._pending_payload = payload
        self._program = Tau3ProgramRuntime(
            tuple(tool.name for tool in self.tools),
            max_stdout_chars=int(self._runtime_config["max_stdout_chars"]),
            timeout_seconds=float(self._runtime_config["code_timeout_seconds"]),
        )
        return self._handle_event(self._program.start(code), state)

    def _resume(self, message: Any, state: Tau3AgentState):
        if self._program is None:
            raise RuntimeError("official tool result has no suspended PTC program")
        return self._handle_event(
            self._program.resume(
                message.content or "",
                error=bool(message.error),
                state_changed=self._state_changed(),
                declared_effect=self._declared_effect(),
            ),
            state,
        )

    def _handle_event(self, event: ToolRequest | BlockComplete, state: Tau3AgentState):
        from tau2.data_model.message import AssistantMessage, ToolCall

        if isinstance(event, ToolRequest):
            self._pending_tool_name = event.name
            self._pending_db_hash = self._database_hash(event.name)
            return AssistantMessage(
                role="assistant",
                content=None,
                tool_calls=[
                    ToolCall(
                        id=event.call_id,
                        name=event.name,
                        arguments=event.arguments,
                        requestor="assistant",
                    )
                ],
            )
        self._blocks += 1
        if not event.success:
            self._execution_failures += 1
        content = event.stdout or ("Execution successful." if event.success else "Execution failed.")
        trace = _block_trace(event, turn=self._model_requests, tool_call_id=self._pending_ptc_id)
        self._block_traces.append(_dump(vars(trace)))
        if self._controller is not None:
            content += "\n\n" + self._controller.observe(trace)
        state.messages.append(
            {"role": "tool", "tool_call_id": self._pending_ptc_id, "content": content}
        )
        self._program.close()
        self._program = None
        self._pending_ptc_id = None
        self._pending_payload = None
        self._pending_db_hash = None
        self._pending_tool_name = None
        if self._pending_blocks:
            return self._start_next_block(state)
        return self._model_step(state)

    def _database_hash(self, tool_name: str) -> str | None:
        tool = next((item for item in self.tools if item.name == tool_name), None)
        owner = getattr(getattr(tool, "_func", None), "__self__", None)
        get_hash = getattr(owner, "get_db_hash", None)
        return str(get_hash()) if callable(get_hash) else None

    def _state_changed(self) -> bool | None:
        if self._pending_db_hash is None:
            return None
        if self._pending_tool_name is None:
            return None
        current = self._database_hash(self._pending_tool_name)
        return None if current is None else current != self._pending_db_hash

    def _declared_effect(self) -> str | None:
        if self._pending_tool_name is None:
            return None
        tool = next(
            (item for item in self.tools if item.name == self._pending_tool_name), None
        )
        function = getattr(tool, "_func", None)
        value = getattr(function, "__tool_type__", None)
        return str(getattr(value, "value", value)) if value is not None else None

    def stop(self, message: Any = None, state: Any = None) -> None:
        self.close(answered=bool(getattr(message, "content", None)))

    def close(self, *, answered: bool) -> None:
        if self._closed:
            return
        self._closed = True
        if self._program is not None:
            self._program.close()
        if self._controller is not None:
            self._controller.finish(answered=answered)
        close = getattr(self._model, "close", None)
        if callable(close):
            close()

    def telemetry(self) -> dict[str, Any]:
        return {
            "model_requests": self._model_requests,
            "ptc_blocks": self._blocks,
            "execution_failures": self._execution_failures,
            "usage": dict(self._usage),
            "graph": self._controller.telemetry() if self._controller is not None else None,
        }

    def graph_artifact(self) -> dict[str, Any] | None:
        return self._controller.graph_artifact() if self._controller is not None else None

    def agent_artifact(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "model": {
                key: value
                for key, value in dict(self._request["agent_model"]).items()
                if key != "api_key"
            },
            "runtime": dict(self._runtime_config),
            "graph_adaptation_mode": self._request["graph_adaptation_mode"],
            "system_prompt": self._system,
            "demonstration_messages": copy.deepcopy(self._demos),
            "messages": copy.deepcopy(self._state_messages),
            "blocks": copy.deepcopy(self._block_traces),
            "scaffold_failures": copy.deepcopy(self._scaffold_failures),
            "telemetry": self.telemetry(),
        }


# The official registry validates factory output at build time, so the concrete class is
# dynamically combined with HalfDuplexAgent after tau2 is importable.
def _agent_class():
    from tau2.agent.base_agent import HalfDuplexAgent

    if issubclass(GraphPTCTau3Agent, HalfDuplexAgent):
        return GraphPTCTau3Agent
    return type("RegisteredGraphPTCTau3Agent", (GraphPTCTau3Agent, HalfDuplexAgent), {})


def _official_to_openai(message: Any) -> dict[str, Any]:
    role = str(message.role)
    payload: dict[str, Any] = {"role": role, "content": getattr(message, "content", None)}
    if role == "tool":
        payload["tool_call_id"] = message.id
    return payload


def _block_trace(event: BlockComplete, *, turn: int, tool_call_id: str | None) -> Any:
    actions = [
        {
            "name": call["name"],
            "arguments": call["arguments"],
            "effect": call.get("effect", "unknown"),
            "success": call["success"],
            "outcome_unknown": False,
            "effect_basis": call.get("effect_basis", "official_tool_result"),
            "state_changed": call.get("state_changed"),
            "output": call.get("output"),
        }
        for call in event.calls
    ]
    return SimpleNamespace(
        turn=turn,
        tool_call_id=tool_call_id,
        code=event.code,
        stdout=event.stdout,
        stdout_chars=event.stdout_chars,
        stdout_truncated=event.stdout_truncated,
        success=event.success,
        duration_ms=event.duration_ms,
        invocation_id=None,
        runtime_calls=len(actions),
        program_analysis=_program_analysis(event.code, {item["name"] for item in actions}),
        runtime_trace={
            "external_actions": actions,
            "state_before": {},
            "state_after": {},
            "loaded_names": [],
            "stored_names": [],
            "error_location": "official_environment" if not event.success else None,
        },
        error_type=event.error_type,
        error_message=event.error_message,
    )


def _program_analysis(code: str, tool_names: set[str]) -> dict[str, Any]:
    try:
        tree = ast.parse(code)
    except SyntaxError as exc:
        return {"tool_call_count": 0, "transform_count": 0, "control_dependency_count": 0, "syntax_error": str(exc)}
    calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call)]
    return {
        "tool_call_count": sum(isinstance(node.func, ast.Name) and node.func.id in tool_names for node in calls),
        "transform_count": sum(isinstance(node, (ast.ListComp, ast.DictComp, ast.SetComp, ast.GeneratorExp)) for node in ast.walk(tree)),
        "control_dependency_count": sum(isinstance(node, (ast.If, ast.For, ast.While, ast.Try)) for node in ast.walk(tree)),
        "syntax_error": None,
    }


def _inspect(request: Mapping[str, Any]) -> dict[str, Any]:
    from tau2.config import (
        DEFAULT_LLM_TEMPERATURE_AGENT,
        DEFAULT_LLM_TEMPERATURE_USER,
        DEFAULT_MAX_CONCURRENCY,
        DEFAULT_MAX_ERRORS,
        DEFAULT_MAX_STEPS,
        DEFAULT_RETRY_ATTEMPTS,
        DEFAULT_RETRY_MIN_WAIT,
        DEFAULT_SEED,
    )
    from tau2.runner import get_tasks

    root = Path(str(request["root"]))
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=root, text=True, capture_output=True, check=True
    ).stdout.strip()
    check = subprocess.run(
        [str(root / ".venv/bin/tau2"), "check-data"],
        cwd=root,
        text=True,
        capture_output=True,
        check=False,
    )
    domains: dict[str, Any] = {}
    for domain in request["domains"]:
        tasks = get_tasks(
            task_set_name=domain,
            task_split_name=request["task_split_name"],
            task_ids=None,
            num_tasks=None,
        )
        task_ids = [str(task.id) for task in tasks]
        domains[domain] = {
            "task_ids": task_ids,
            "count": len(task_ids),
            "task_ids_sha256": hashlib.sha256("\n".join(task_ids).encode()).hexdigest(),
        }
    return {
        "type": "inspection",
        "official_commit": commit,
        "package_version": version("tau2"),
        "python_version": sys.version.split()[0],
        "data_verified": check.returncode == 0,
        "official_defaults": {
            "max_steps": DEFAULT_MAX_STEPS,
            "max_errors": DEFAULT_MAX_ERRORS,
            "seed": DEFAULT_SEED,
            "max_concurrency": DEFAULT_MAX_CONCURRENCY,
            "agent_temperature": DEFAULT_LLM_TEMPERATURE_AGENT,
            "user_temperature": DEFAULT_LLM_TEMPERATURE_USER,
            "enforce_communication_protocol": False,
            "max_retries": DEFAULT_RETRY_ATTEMPTS,
            "retry_delay": DEFAULT_RETRY_MIN_WAIT,
        },
        "domains": domains,
    }


def _run(request: Mapping[str, Any]) -> dict[str, Any]:
    from tau2.data_model.simulation import TextRunConfig
    from tau2.registry import registry
    from tau2.runner import get_tasks, run_single_task

    api_key = os.environ[str(request["agent_model"].get("api_key_env", "MIMO_API_KEY"))]
    os.environ["OPENAI_API_KEY"] = api_key
    os.environ["OPENAI_API_BASE"] = str(request["user_base_url"])
    tasks = get_tasks(
        task_set_name=request["domain"],
        task_split_name=request["task_split_name"],
        task_ids=[str(request["task_id"])],
        num_tasks=None,
    )
    if len(tasks) != 1:
        raise ValueError(f"expected one official task, found {len(tasks)}")
    holder: dict[str, GraphPTCTau3Agent] = {}

    def factory(tools, domain_policy, **kwargs):
        cls = _agent_class()
        agent = cls(tools=tools, domain_policy=domain_policy, request=request)
        holder["agent"] = agent
        return agent

    agent_name = str(request["agent_name"])
    registry.register_agent_factory(factory, agent_name)
    config = TextRunConfig(
        domain=request["domain"],
        task_set_name=request["domain"],
        task_split_name=request["task_split_name"],
        task_ids=[str(request["task_id"])],
        agent=agent_name,
        llm_agent=str(request["agent_model"]["model"]),
        llm_args_agent={},
        user="user_simulator",
        llm_user=str(request["user_model"]),
        llm_args_user={"temperature": 0.0, "api_base": str(request["user_base_url"])},
        num_trials=1,
        max_steps=int(request["max_steps"]),
        max_errors=int(request["max_errors"]),
        max_concurrency=1,
        seed=int(request["seed"]),
        timeout=float(request["timeout"]),
        log_level="ERROR",
        verbose_logs=False,
        max_retries=0,
        hallucination_retries=0,
        enforce_communication_protocol=bool(
            request["enforce_communication_protocol"]
        ),
    )
    simulation = None
    agent = None
    try:
        simulation = _stamp_trial(
            run_single_task(config, tasks[0], seed=int(request["seed"])),
            int(request["trial"]),
        )
        agent = holder.get("agent")
    finally:
        agent = holder.get("agent")
        if agent is not None:
            agent.close(answered=bool(simulation and simulation.reward_info is not None))
    official_path = Path(str(request["official_path"]))
    official_path.parent.mkdir(parents=True, exist_ok=True)
    official_path.write_text(simulation.model_dump_json(indent=2), encoding="utf-8")
    agent_path = Path(str(request["agent_path"]))
    agent_path.parent.mkdir(parents=True, exist_ok=True)
    agent_path.write_text(
        json.dumps(agent.agent_artifact(), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    graph = agent.graph_artifact() if agent is not None else None
    if graph is not None:
        graph_path = Path(str(request["graph_path"]))
        graph_path.parent.mkdir(parents=True, exist_ok=True)
        graph_path.write_text(json.dumps(graph, ensure_ascii=False, indent=2), encoding="utf-8")
    telemetry = agent.telemetry() if agent is not None else {}
    reward = simulation.reward_info.reward if simulation.reward_info is not None else 0.0
    termination = str(simulation.termination_reason)
    return {
        "type": "result",
        "status": "finished",
        "simulation_id": simulation.id,
        "reward": float(reward or 0.0),
        "termination_reason": termination,
        "incomplete": "user_stop" not in termination.lower() and float(reward or 0.0) < 1.0,
        "evaluator_failed": simulation.reward_info is None,
        "execution_failures": int(telemetry.get("execution_failures", 0)),
        "telemetry": telemetry,
        "official_path": str(official_path),
        "agent_path": str(agent_path),
        "graph_path": str(request["graph_path"]) if graph is not None else None,
    }


def _aggregate(request: Mapping[str, Any]) -> dict[str, Any]:
    from tau2.data_model.simulation import Results, SimulationRun, TextRunConfig
    from tau2.runner import get_tasks
    from tau2.runner.helpers import get_info

    task_ids = list(dict.fromkeys(str(value) for value in request["task_ids"]))
    tasks = get_tasks(
        task_set_name=request["domain"],
        task_split_name=request["task_split_name"],
        task_ids=task_ids,
        num_tasks=None,
    )
    simulations = [
        SimulationRun.model_validate_json(Path(path).read_text(encoding="utf-8"))
        for path in request["official_paths"]
    ]
    config = TextRunConfig(
        domain=request["domain"],
        task_set_name=request["domain"],
        task_split_name=request["task_split_name"],
        task_ids=task_ids,
        agent=request["agent_name"],
        llm_agent=request["agent_model"],
        llm_args_agent={},
        user="user_simulator",
        llm_user=request["user_model"],
        llm_args_user={"temperature": 0.0, "api_base": request["user_base_url"]},
        num_trials=int(request["num_trials"]),
        max_steps=int(request["max_steps"]),
        max_errors=int(request["max_errors"]),
        max_concurrency=int(request["max_concurrency"]),
        seed=int(request["seed"]),
        log_level="ERROR",
        verbose_logs=False,
        max_retries=int(request["max_retries"]),
        retry_delay=float(request["retry_delay"]),
        hallucination_retries=0,
        enforce_communication_protocol=bool(
            request["enforce_communication_protocol"]
        ),
    )
    results = Results(info=get_info(config), tasks=tasks, simulations=simulations)
    output_path = Path(str(request["output_path"]))
    results.save(output_path, format="json")
    return {
        "type": "aggregate",
        "domain": request["domain"],
        "output_path": str(output_path),
        "tasks": len(tasks),
        "simulations": len(simulations),
    }


def main() -> int:
    try:
        request = json.loads(sys.stdin.readline().lstrip("\ufeff"))
        if request.get("type") == "inspect":
            response = _inspect(request)
        elif request.get("type") == "aggregate":
            response = _aggregate(request)
        else:
            response = _run(request)
        _emit(response)
        return 0
    except Exception as exc:  # noqa: BLE001 - worker protocol serializes boundary failures
        _emit({"type": "error", "error": f"{type(exc).__name__}: {exc}"})
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
