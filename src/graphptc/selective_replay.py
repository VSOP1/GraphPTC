from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from .failure_attribution import build_failure_contexts
from .invalidation import InvalidationPlan, ToolReplayDecision, analyze_invalidation
from .observability import ExecutionObserver, InMemoryEventSink
from .patch_controller import (
    LocalPatchProposal,
    PatchApplication,
    apply_local_patch,
    build_repair_context,
)
from .persistent_runtime import InterceptedToolResult, PersistentIpcRuntime
from .stage2_graph import DependencyGraph, GraphNode, load_dependency_graph_report
from .stage4_repair import BlockReexecution


@dataclass(frozen=True)
class ReplayToolEvent:
    sequence: int
    source_block_id: str
    source_tool_node_id: str | None
    source_artifact_id: str | None
    action: str
    tool: str
    arguments: dict[str, Any]
    call_site: dict[str, Any] | None
    result: Any


@dataclass(frozen=True)
class SelectiveReplayResult:
    episode_id: str
    target_block_id: str
    original_version_id: str
    patched_version_id: str
    success: bool
    reset_required: bool
    error: str | None
    blocks: tuple[BlockReexecution, ...]
    tool_events: tuple[ReplayToolEvent, ...]
    reused_tool_call_count: int
    executed_tool_call_count: int
    unconsumed_reusable_tool_node_ids: tuple[str, ...]


def selective_replay_patch(
    graph: DependencyGraph,
    application: PatchApplication,
    plan: InvalidationPlan,
    *,
    live_tools: dict[str, Callable[..., Any]],
    timeout_seconds: float = 120.0,
    replay_runtime: PersistentIpcRuntime | None = None,
    close_runtime: bool = True,
) -> SelectiveReplayResult:
    if not close_runtime and replay_runtime is None:
        raise ValueError("retained selective replay requires a supplied runtime")
    _validate_inputs(graph, application, plan, timeout_seconds)
    prefix_blocks = _prefix_blocks(graph, plan.target_block_id)
    prefix_block_ids = {node.block_id for node in prefix_blocks}
    reset_nodes = tuple(
        decision.node_id
        for decision in plan.tool_decisions
        if decision.action == "RESET_REQUIRED"
        and graph.node(decision.node_id).block_id in prefix_block_ids
    )
    if reset_nodes:
        return SelectiveReplayResult(
            episode_id=graph.episode_id,
            target_block_id=plan.target_block_id,
            original_version_id=application.original.id,
            patched_version_id=application.patched.id,
            success=False,
            reset_required=True,
            error="reset required for tool nodes: " + ", ".join(reset_nodes),
            blocks=(),
            tool_events=(),
            reused_tool_call_count=0,
            executed_tool_call_count=0,
            unconsumed_reusable_tool_node_ids=(),
        )

    interceptor = _ReplayInterceptor(graph, plan, live_tools)
    runtime_namespace = dict(live_tools)
    for decision in plan.tool_decisions:
        runtime_namespace.setdefault(decision.tool, _tool_placeholder)
    sink = InMemoryEventSink()
    observer = ExecutionObserver(sink, episode_id=graph.episode_id, task_id=graph.task_id)
    runtime = replay_runtime or PersistentIpcRuntime(observer=observer)
    previous_observer = runtime.observer
    previous_interceptor = runtime.tool_call_interceptor
    runtime.observer = observer
    runtime.tool_call_interceptor = interceptor.intercept
    blocks: list[BlockReexecution] = []
    reached_target = False
    try:
        for block in prefix_blocks:
            assert block.block_id is not None
            runtime.active_block_id = block.block_id
            interceptor.active_block_id = block.block_id
            is_target = block.block_id == plan.target_block_id
            code = (
                application.patched.code
                if is_target
                else str(block.data.get("code", ""))
            )
            execution = runtime.execute(
                code,
                namespace=runtime_namespace,
                timeout=timeout_seconds,
            )
            timed_out = bool(execution.timed_out)
            success = execution.return_code == 0 and not timed_out
            blocks.append(
                BlockReexecution(
                    block_id=block.block_id,
                    program_version_id=(application.patched.id if is_target else None),
                    success=success,
                    stdout=execution.stdout,
                    stderr=execution.stderr,
                    return_code=execution.return_code,
                    timed_out=timed_out,
                    runtime_trace=dict(runtime.last_execution_trace),
                )
            )
            if not success:
                break
            if is_target:
                reached_target = True
                break
    finally:
        runtime.active_block_id = None
        runtime.tool_call_interceptor = previous_interceptor
        runtime.observer = previous_observer
        if close_runtime:
            runtime.close()

    executed_block_ids = {block.block_id for block in blocks}
    unconsumed = interceptor.unconsumed_reusable_nodes(executed_block_ids)
    success = (
        reached_target
        and all(block.success for block in blocks)
        and not unconsumed
    )
    error = None
    if unconsumed:
        error = "reusable tool calls were not observed: " + ", ".join(unconsumed)
    elif not success:
        error = "selective replay execution failed"
    events = tuple(interceptor.events)
    return SelectiveReplayResult(
        episode_id=graph.episode_id,
        target_block_id=plan.target_block_id,
        original_version_id=application.original.id,
        patched_version_id=application.patched.id,
        success=success,
        reset_required=False,
        error=error,
        blocks=tuple(blocks),
        tool_events=events,
        reused_tool_call_count=sum(event.action == "REUSE_RESULT" for event in events),
        executed_tool_call_count=sum(
            event.action in {"REEXECUTE", "EXECUTE_NEW"} for event in events
        ),
        unconsumed_reusable_tool_node_ids=unconsumed,
    )


def write_selective_replay_audit_report(
    graph_path: str | Path,
    expectations_path: str | Path,
    output_path: str | Path,
) -> dict[str, Any]:
    expectation_bytes = Path(expectations_path).read_bytes()
    expectations = json.loads(expectation_bytes)
    if not isinstance(expectations, dict) or expectations.get("schema_version") != 1:
        raise ValueError("Unsupported Stage 5 selective replay expectations")
    cases = expectations.get("cases")
    if not isinstance(cases, list) or not all(isinstance(case, dict) for case in cases):
        raise ValueError("Stage 5 selective replay audit requires a cases list")
    graphs = load_dependency_graph_report(graph_path)
    graphs_by_episode = {graph.episode_id: graph for graph in graphs}

    results = []
    exact_passed = 0
    exact_total = 0
    for case in cases:
        episode_id = str(case.get("episode_id") or "")
        if episode_id not in graphs_by_episode:
            raise ValueError(f"Unknown Stage 5 replay episode: {episode_id}")
        graph = graphs_by_episode[episode_id]
        anchor_node_id = str(case.get("anchor_node_id") or "")
        contexts = [
            context
            for context in build_failure_contexts(graph)
            if context.anchor.node_id == anchor_node_id
        ]
        if len(contexts) != 1:
            raise ValueError(f"Expected one Stage 5 replay anchor: {anchor_node_id}")
        proposal_value = case.get("proposal")
        if not isinstance(proposal_value, dict):
            raise ValueError("Stage 5 selective replay case requires a proposal")
        try:
            proposal = LocalPatchProposal(**proposal_value)
        except TypeError as exc:
            raise ValueError("Invalid Stage 5 selective replay proposal") from exc
        application = apply_local_patch(
            graph,
            build_repair_context(graph, contexts[0]),
            proposal,
        )
        read_only_tools = (
            frozenset()
            if case.get("force_reset") is True
            else frozenset({"search", "fetch"})
        )
        plan = analyze_invalidation(
            graph,
            application,
            read_only_tool_names=read_only_tools,
        )
        live_calls: list[dict[str, Any]] = []
        replay = selective_replay_patch(
            graph,
            application,
            plan,
            live_tools=_audit_live_tools(case, live_calls),
            timeout_seconds=5,
        )
        actual = {
            "success": replay.success,
            "reset_required": replay.reset_required,
            "block_ids": [block.block_id for block in replay.blocks],
            "final_stdout": replay.blocks[-1].stdout.strip() if replay.blocks else None,
            "reused_tool_call_count": replay.reused_tool_call_count,
            "executed_tool_call_count": replay.executed_tool_call_count,
            "tool_events": [
                {
                    "source_tool_node_id": event.source_tool_node_id,
                    "source_artifact_id": event.source_artifact_id,
                    "action": event.action,
                    "tool": event.tool,
                    "arguments": event.arguments,
                }
                for event in replay.tool_events
            ],
            "live_calls": live_calls,
            "unconsumed_reusable_tool_node_ids": list(
                replay.unconsumed_reusable_tool_node_ids
            ),
        }
        expected = case.get("expected")
        if not isinstance(expected, dict):
            raise ValueError("Stage 5 selective replay case requires expected values")
        checks = {name: actual.get(name) == value for name, value in expected.items()}
        exact_passed += sum(checks.values())
        exact_total += len(checks)
        results.append(
            {
                "case_id": case.get("case_id"),
                "episode_id": episode_id,
                "passed": all(checks.values()),
                "checks": checks,
                "actual": actual,
            }
        )
    exact_match_rate = exact_passed / exact_total if exact_total else 1.0
    report = {
        "schema_version": 1,
        "expectations_sha256": hashlib.sha256(expectation_bytes).hexdigest(),
        "case_count": len(results),
        "exact_match_rate": exact_match_rate,
        "passed": all(result["passed"] for result in results)
        and exact_match_rate == 1.0,
        "cases": results,
    }
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return report


class _ReplayInterceptor:
    def __init__(
        self,
        graph: DependencyGraph,
        plan: InvalidationPlan,
        live_tools: dict[str, Callable[..., Any]],
    ) -> None:
        self.graph = graph
        self.live_tools = live_tools
        self.active_block_id: str | None = None
        self.events: list[ReplayToolEvent] = []
        self._consumed: set[str] = set()
        self._decisions = {decision.node_id: decision for decision in plan.tool_decisions}
        self._tool_nodes = [node for node in graph.nodes if node.type == "TOOL"]

    def intercept(
        self,
        tool: str,
        arguments: dict[str, Any],
        call_site: dict[str, Any] | None,
        live_tool: Callable[..., Any] | None,
    ) -> InterceptedToolResult:
        match = self._match(tool, arguments, call_site)
        source_artifact_id = None
        if match is None:
            if self.active_block_id is None:
                raise RuntimeError("replay tool call has no active block")
            actual_tool = self.live_tools.get(tool)
            if actual_tool is None:
                raise RuntimeError(f"live tool is unavailable: {tool}")
            source_block_id = self.active_block_id
            source_tool_node_id = None
            action = "EXECUTE_NEW"
            value = actual_tool(**arguments)
        else:
            node, decision = match
            source_block_id = str(node.block_id)
            source_tool_node_id = node.id
            action = decision.action
        if match is not None and action == "REUSE_RESULT":
            if node.data.get("arguments") != arguments:
                raise RuntimeError("cached tool arguments do not match the replay call")
            if len(node.artifact_ids) != 1:
                raise RuntimeError("reusable tool call requires exactly one artifact")
            source_artifact_id = node.artifact_ids[0]
            value = self.graph.artifact(source_artifact_id).value
        elif match is not None and action == "REEXECUTE":
            actual_tool = self.live_tools.get(tool)
            if actual_tool is None:
                raise RuntimeError(f"live tool is unavailable: {tool}")
            value = actual_tool(**arguments)
        elif match is not None:
            raise RuntimeError(f"unsupported replay action: {action}")
        if match is not None:
            self._consumed.add(node.id)
        event = ReplayToolEvent(
            sequence=len(self.events) + 1,
            source_block_id=source_block_id,
            source_tool_node_id=source_tool_node_id,
            source_artifact_id=source_artifact_id,
            action=action,
            tool=tool,
            arguments=dict(arguments),
            call_site=dict(call_site) if call_site is not None else None,
            result=value,
        )
        self.events.append(event)
        return InterceptedToolResult(
            value=value,
            event_data={
                "replay_action": action,
                "source_block_id": source_block_id,
                "source_tool_node_id": source_tool_node_id,
                "source_artifact_id": source_artifact_id,
            },
        )

    def _match(
        self,
        tool: str,
        arguments: dict[str, Any],
        call_site: dict[str, Any] | None,
    ) -> tuple[GraphNode, ToolReplayDecision] | None:
        candidates = [
            node
            for node in self._tool_nodes
            if node.id not in self._consumed
            and node.block_id == self.active_block_id
            and node.data.get("tool") == tool
            and node.id in self._decisions
        ]
        exact_site = [
            node for node in candidates if node.data.get("call_site") == call_site
        ]
        for node in exact_site:
            decision = self._decisions[node.id]
            if (
                decision.action == "REUSE_RESULT"
                and node.data.get("arguments") == arguments
            ):
                return node, decision
        for node in exact_site:
            decision = self._decisions[node.id]
            if decision.action == "REEXECUTE":
                return node, decision
        for node in candidates:
            decision = self._decisions[node.id]
            if decision.action == "REEXECUTE":
                return node, decision
        return None

    def unconsumed_reusable_nodes(
        self,
        executed_block_ids: set[str],
    ) -> tuple[str, ...]:
        return tuple(
            node.id
            for node in self._tool_nodes
            if node.block_id in executed_block_ids
            and node.id not in self._consumed
            and self._decisions.get(node.id) is not None
            and self._decisions[node.id].action == "REUSE_RESULT"
        )


def _validate_inputs(
    graph: DependencyGraph,
    application: PatchApplication,
    plan: InvalidationPlan,
    timeout_seconds: float,
) -> None:
    if timeout_seconds <= 0:
        raise ValueError("timeout_seconds must be positive")
    if plan.episode_id != graph.episode_id:
        raise ValueError("invalidation plan belongs to a different episode")
    if plan.original_version_id != application.original.id:
        raise ValueError("invalidation plan original version does not match")
    if plan.patched_version_id != application.patched.id:
        raise ValueError("invalidation plan patched version does not match")
    if plan.target_block_id != application.patched.block_id:
        raise ValueError("invalidation plan target block does not match")


def _prefix_blocks(graph: DependencyGraph, target_block_id: str) -> tuple[GraphNode, ...]:
    blocks = []
    for node in graph.nodes:
        if node.type != "BLOCK":
            continue
        blocks.append(node)
        if node.block_id == target_block_id:
            return tuple(blocks)
    raise ValueError(f"unknown selective replay target block: {target_block_id}")


def _tool_placeholder(**kwargs: Any) -> Any:
    raise RuntimeError(f"tool call bypassed selective replay interceptor: {kwargs}")


def _audit_live_tools(
    case: dict[str, Any],
    live_calls: list[dict[str, Any]],
) -> dict[str, Callable[..., Any]]:
    fixture = case.get("live_tool")
    if fixture is None:
        return {}
    if not isinstance(fixture, dict):
        raise ValueError("Stage 5 selective replay live_tool must be an object")
    name = str(fixture.get("name") or "")
    if not name or "result" not in fixture:
        raise ValueError("Stage 5 selective replay live_tool requires name and result")
    result = fixture["result"]

    def tool(**kwargs: Any) -> Any:
        live_calls.append({"tool": name, "arguments": dict(kwargs)})
        return result

    return {name: tool}
