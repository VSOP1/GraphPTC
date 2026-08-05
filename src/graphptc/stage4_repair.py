from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Callable, Protocol

from .model import ModelAttempt, ModelTurn, TokenUsage, ToolCall
from .observability import ExecutionObserver, InMemoryEventSink
from .patch_controller import (
    LocalPatchProposal,
    PatchApplication,
    ProgramVersion,
    RepairContext,
)
from .persistent_runtime import PersistentIpcRuntime
from .stage2_graph import DependencyGraph


REPAIR_PROMPT_VERSION = "graphptc-repair-v2"


REPAIR_SYSTEM_PROMPT = """You are the local patch component for GraphPTC fewshot-ptc-v1.
You receive only a bounded failure context, not the full execution history. Submit exactly one
minimal patch inside an exposed code region. Preserve unrelated code and do not regenerate the
whole program. Start with preferred_patch_region and expand only when changing that exact range
cannot repair the failure. Do not include unchanged neighboring lines in a patch. Runtime tools
accept only the arguments declared in runtime_tool_manifest. expected_code must exactly match the
specified source lines. Use only the submit_local_patch tool and do not provide a prose answer."""


LOCAL_PATCH_TOOL_SPEC: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "submit_local_patch",
        "description": "Submit one localized replacement for an exposed source range.",
        "parameters": {
            "type": "object",
            "properties": {
                "block_id": {"type": "string", "minLength": 1},
                "start_line": {"type": "integer", "minimum": 1},
                "end_line": {"type": "integer", "minimum": 1},
                "expected_code": {"type": "string"},
                "replacement_code": {"type": "string"},
            },
            "required": [
                "block_id",
                "start_line",
                "end_line",
                "expected_code",
                "replacement_code",
            ],
            "additionalProperties": False,
        },
        "strict": True,
    },
}


class RepairModel(Protocol):
    def create_turn(
        self,
        *,
        system: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        timeout_seconds: float | None = None,
        max_completion_tokens: int | None = None,
        thinking: str | None = None,
    ) -> ModelTurn: ...


@dataclass(frozen=True)
class GeneratedPatch:
    proposal: LocalPatchProposal
    usage: TokenUsage
    stop_reason: str | None
    tool_call_id: str
    attempts: tuple[ModelAttempt, ...]


@dataclass(frozen=True)
class ReexecutionResult:
    program_version_id: str
    success: bool
    stdout: str
    stderr: str
    return_code: int
    timed_out: bool
    runtime_trace: dict[str, Any]


@dataclass(frozen=True)
class BlockReexecution:
    block_id: str
    program_version_id: str | None
    success: bool
    stdout: str
    stderr: str
    return_code: int
    timed_out: bool
    runtime_trace: dict[str, Any]


@dataclass(frozen=True)
class PatchPrefixReexecution:
    target_block_id: str
    success: bool
    blocks: tuple[BlockReexecution, ...]
    reused_block_ids: tuple[str, ...]


def request_local_patch(
    model: RepairModel,
    repair: RepairContext,
    *,
    timeout_seconds: float | None = None,
    max_completion_tokens: int = 1024,
) -> GeneratedPatch:
    payload = {
        "repair_prompt_version": REPAIR_PROMPT_VERSION,
        "prompt_variant": repair.prompt_variant,
        "task": repair.task,
        "failure": repair.failure.to_dict(),
        "preferred_patch_region": (
            None
            if repair.preferred_patch_region is None
            else {
                "block_id": repair.preferred_patch_region.block_id,
                "start_line": repair.preferred_patch_region.start_line,
                "end_line": repair.preferred_patch_region.end_line,
                "code": repair.preferred_patch_region.code,
            }
        ),
        "runtime_tool_manifest": list(repair.runtime_tool_manifest),
        "patchable_regions": [
            {
                "block_id": region.block_id,
                "start_line": region.start_line,
                "end_line": region.end_line,
                "focus_lines": list(region.focus_lines),
                "code": region.code,
            }
            for region in repair.patchable_regions
        ],
    }
    turn = model.create_turn(
        system=REPAIR_SYSTEM_PROMPT,
        messages=[
            {
                "role": "user",
                "content": json.dumps(
                    payload,
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                ),
            }
        ],
        tools=[LOCAL_PATCH_TOOL_SPEC],
        timeout_seconds=timeout_seconds,
        max_completion_tokens=max_completion_tokens,
        thinking="disabled",
    )
    if len(turn.tool_calls) != 1:
        raise ValueError("repair model must submit exactly one local patch")
    call = turn.tool_calls[0]
    if call.name != "submit_local_patch":
        raise ValueError("repair model must call submit_local_patch")
    proposal = _proposal_from_tool_call(call)
    return GeneratedPatch(
        proposal=proposal,
        usage=turn.usage,
        stop_reason=turn.stop_reason,
        tool_call_id=call.id,
        attempts=turn.attempts,
    )


def reexecute_program_version(
    version: ProgramVersion,
    *,
    namespace: dict[str, Callable[..., Any]] | None = None,
    timeout_seconds: float = 120.0,
) -> ReexecutionResult:
    if timeout_seconds <= 0:
        raise ValueError("timeout_seconds must be positive")
    observer = ExecutionObserver(
        InMemoryEventSink(),
        episode_id=version.episode_id,
        task_id="stage4-reexecution",
    )
    runtime = PersistentIpcRuntime(observer=observer)
    runtime.active_block_id = version.block_id
    try:
        result = runtime.execute(
            version.code,
            namespace=namespace,
            timeout=timeout_seconds,
        )
        runtime_trace = dict(runtime.last_execution_trace)
    finally:
        runtime.close()
    timed_out = bool(result.timed_out)
    return ReexecutionResult(
        program_version_id=version.id,
        success=result.return_code == 0 and not timed_out,
        stdout=result.stdout,
        stderr=result.stderr,
        return_code=result.return_code,
        timed_out=timed_out,
        runtime_trace=runtime_trace,
    )


def reexecute_patch_prefix(
    graph: DependencyGraph,
    application: PatchApplication,
    *,
    namespace: dict[str, Callable[..., Any]] | None = None,
    timeout_seconds: float = 120.0,
) -> PatchPrefixReexecution:
    if timeout_seconds <= 0:
        raise ValueError("timeout_seconds must be positive")
    if application.patched.episode_id != graph.episode_id:
        raise ValueError("patch application belongs to a different episode")
    target_block_id = application.patched.block_id
    blocks = [node for node in graph.nodes if node.type == "BLOCK"]
    if not any(node.block_id == target_block_id for node in blocks):
        raise ValueError(f"unknown patch target block: {target_block_id}")

    observer = ExecutionObserver(
        InMemoryEventSink(),
        episode_id=graph.episode_id,
        task_id=graph.task_id,
    )
    runtime = PersistentIpcRuntime(observer=observer)
    results: list[BlockReexecution] = []
    reached_target = False
    try:
        for block in blocks:
            assert block.block_id is not None
            runtime.active_block_id = block.block_id
            is_target = block.block_id == target_block_id
            code = (
                application.patched.code
                if is_target
                else str(block.data.get("code", ""))
            )
            execution = runtime.execute(
                code,
                namespace=namespace,
                timeout=timeout_seconds,
            )
            timed_out = bool(execution.timed_out)
            success = execution.return_code == 0 and not timed_out
            results.append(
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
        runtime.close()
    return PatchPrefixReexecution(
        target_block_id=target_block_id,
        success=reached_target and all(result.success for result in results),
        blocks=tuple(results),
        reused_block_ids=(),
    )


def _proposal_from_tool_call(call: ToolCall) -> LocalPatchProposal:
    expected_keys = {
        "block_id",
        "start_line",
        "end_line",
        "expected_code",
        "replacement_code",
    }
    if set(call.input) != expected_keys:
        raise ValueError("local patch fields do not match the required schema")
    if not isinstance(call.input["block_id"], str) or not call.input["block_id"]:
        raise ValueError("local patch block_id must be non-empty")
    for name in ("start_line", "end_line"):
        value = call.input[name]
        if not isinstance(value, int) or isinstance(value, bool) or value < 1:
            raise ValueError(f"local patch {name} must be a positive integer")
    for name in ("expected_code", "replacement_code"):
        if not isinstance(call.input[name], str):
            raise ValueError(f"local patch {name} must be a string")
    return LocalPatchProposal(
        block_id=call.input["block_id"],
        start_line=call.input["start_line"],
        end_line=call.input["end_line"],
        expected_code=call.input["expected_code"],
        replacement_code=call.input["replacement_code"],
    )
