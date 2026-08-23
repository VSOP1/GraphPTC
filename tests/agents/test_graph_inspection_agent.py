from __future__ import annotations

import json
from typing import Any, Callable

from codecell import BaseRuntime, CodeResult
from codecell.python import PythonValidator

from graphptc.config import RuntimeConfig
from graphptc.goal_adaptation import GoalGraphAdaptation
from graphptc.graph_agent import GraphAgentHooks, extend_ptc_spec_with_graph_control
from graphptc.model import ModelTurn, TokenUsage, ToolCall
from graphptc.ptc import OriginalPTCAgent, PTC_TOOL_SPEC


class _EmptyTools:
    calls: list[dict[str, Any]] = []


class _RecordingRuntime(BaseRuntime):
    def __init__(self) -> None:
        super().__init__(PythonValidator())
        self.codes: list[str] = []
        self.last_execution_trace: dict[str, Any] = {}

    def execute(
        self,
        code: str,
        *,
        namespace: dict[str, Callable[..., Any]] | None = None,
        timeout: float | None = None,
    ) -> CodeResult:
        del namespace, timeout
        self.codes.append(code)
        return CodeResult(stdout=f"executed: {code}\n")


def _tool_turn(call_id: str, payload: dict[str, Any]) -> ModelTurn:
    return ModelTurn(
        assistant_message={
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": call_id,
                    "type": "function",
                    "function": {
                        "name": "programmatic_tool_call",
                        "arguments": "{}",
                    },
                }
            ],
        },
        text="",
        tool_calls=[
            ToolCall(
                id=call_id,
                name="programmatic_tool_call",
                input=payload,
            )
        ],
        usage=TokenUsage(),
        stop_reason="tool_calls",
    )


class _InspectionAwareModel:
    def __init__(self) -> None:
        self.calls = 0
        self.saw_inspection = False

    def create_turn(self, *, messages: list[dict[str, Any]], **_: Any) -> ModelTurn:
        self.calls += 1
        if self.calls == 1:
            return _tool_turn(
                "inspect",
                {
                    "code": "pass",
                    "action": "INSPECT",
                    "target": "task",
                    "expected_change": "read the projected graph frontier",
                    "inspection": {"view": "frontier"},
                },
            )
        if self.calls == 2:
            observation = str(messages[-1]["content"])
            payload = json.loads(observation.split("GRAPH_DELTA ", 1)[1])
            artifacts = (
                (payload.get("inspection_result") or {}).get("result") or {}
            ).get("reusable_artifacts", [])
            self.saw_inspection = bool(artifacts)
            code = (
                f"print('used {artifacts[0]}')"
                if self.saw_inspection
                else "print('placebo')"
            )
            return _tool_turn(
                "act",
                {
                    "code": code,
                    "action": "CONTINUE",
                    "target": "task",
                    "expected_change": "act on the available observation",
                },
            )
        return ModelTurn(
            assistant_message={"role": "assistant", "content": "done"},
            text="done",
            tool_calls=[],
            usage=TokenUsage(),
            stop_reason="stop",
        )


def _run(*, inspection_enabled: bool) -> tuple[_RecordingRuntime, _InspectionAwareModel, dict[str, Any]]:
    runtime = _RecordingRuntime()
    model = _InspectionAwareModel()
    controller = GoalGraphAdaptation(
        {},
        {},
        task="Use graph inspection",
        expose_graph_api=False,
        host_inspection_enabled=inspection_enabled,
    )
    hooks = GraphAgentHooks.from_controller(controller)
    hook_kwargs = hooks.agent_kwargs()
    hook_kwargs["runtime_functions"] = ()
    agent = OriginalPTCAgent(
        model=model,
        search_tools=_EmptyTools(),  # type: ignore[arg-type]
        runtime=RuntimeConfig(max_turns=4, max_ptc_blocks=3),
        ptc_tool_spec=extend_ptc_spec_with_graph_control(
            PTC_TOOL_SPEC,
            include_inspection=inspection_enabled,
        ),
        program_runtime=runtime,
        **hook_kwargs,
    )

    agent.run("Use graph inspection")
    return runtime, model, controller.telemetry()


def test_executed_inspection_changes_the_following_program_vs_metadata_placebo() -> None:
    enabled_runtime, enabled_model, enabled_telemetry = _run(inspection_enabled=True)
    placebo_runtime, placebo_model, placebo_telemetry = _run(inspection_enabled=False)

    assert enabled_runtime.codes[0] == "pass"
    assert enabled_runtime.codes[1].startswith("print('used artifact:block:1:")
    assert placebo_runtime.codes == ["pass", "print('placebo')"]
    assert enabled_model.saw_inspection is True
    assert placebo_model.saw_inspection is False
    assert enabled_telemetry["inspection"]["succeeded"] == 1
    assert placebo_telemetry["inspection"]["succeeded"] == 0
