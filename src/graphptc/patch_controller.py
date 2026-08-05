from __future__ import annotations

import ast
import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .failure_attribution import CodeRegion, FailureContext, build_failure_contexts
from .stage2_graph import (
    DependencyGraph,
    GraphNode,
    load_dependency_graph_report,
)


GRAPHPTC_REPAIR_PROMPT_VARIANT = "fewshot-ptc-v1"


@dataclass(frozen=True)
class RepairContext:
    episode_id: str
    task_id: str
    task: str
    prompt_variant: str
    failure: FailureContext
    patchable_regions: tuple[CodeRegion, ...]
    preferred_patch_region: CodeRegion | None
    runtime_tool_manifest: tuple[dict[str, Any], ...]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class LocalPatchProposal:
    block_id: str
    start_line: int
    end_line: int
    expected_code: str
    replacement_code: str


@dataclass(frozen=True)
class ProgramVersion:
    id: str
    episode_id: str
    block_id: str
    parent_version_id: str | None
    code_sha256: str
    code: str


@dataclass(frozen=True)
class CodeVersionMapping:
    old_start_line: int
    old_end_line: int
    new_start_line: int
    new_end_line: int


@dataclass(frozen=True)
class PatchApplication:
    original: ProgramVersion
    patched: ProgramVersion
    proposal: LocalPatchProposal
    mapping: CodeVersionMapping

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def build_repair_context(
    graph: DependencyGraph,
    failure: FailureContext,
    *,
    runtime_tool_manifest: tuple[dict[str, Any], ...] = (),
) -> RepairContext:
    if failure.anchor.episode_id != graph.episode_id:
        raise ValueError("failure context belongs to a different episode")
    graph.node(failure.anchor.node_id)
    episode = next(node for node in graph.nodes if node.type == "EPISODE")
    return RepairContext(
        episode_id=graph.episode_id,
        task_id=graph.task_id,
        task=str(episode.data.get("task") or ""),
        prompt_variant=GRAPHPTC_REPAIR_PROMPT_VARIANT,
        failure=failure,
        patchable_regions=failure.code_regions,
        preferred_patch_region=_preferred_patch_region(graph, failure),
        runtime_tool_manifest=runtime_tool_manifest,
    )


def apply_local_patch(
    graph: DependencyGraph,
    repair: RepairContext,
    proposal: LocalPatchProposal,
) -> PatchApplication:
    if repair.episode_id != graph.episode_id or repair.task_id != graph.task_id:
        raise ValueError("repair context belongs to a different graph")
    if repair.prompt_variant != GRAPHPTC_REPAIR_PROMPT_VARIANT:
        raise ValueError("repair context has an unsupported prompt variant")
    matching_regions = [
        region
        for region in repair.patchable_regions
        if region.block_id == proposal.block_id
        and region.start_line <= proposal.start_line
        and proposal.end_line <= region.end_line
    ]
    if not matching_regions:
        raise ValueError("patch target is outside the repair context")
    if proposal.start_line < 1 or proposal.end_line < proposal.start_line:
        raise ValueError("patch line range is invalid")

    block = _block_node(graph, proposal.block_id)
    original_code = str(block.data.get("code", ""))
    original_lines = original_code.splitlines()
    if proposal.end_line > len(original_lines):
        raise ValueError("patch line range exceeds the source program")
    selected_code = "\n".join(
        original_lines[proposal.start_line - 1 : proposal.end_line]
    )
    if selected_code != proposal.expected_code:
        raise ValueError("expected_code does not match the source program")

    replacement_lines = proposal.replacement_code.splitlines()
    patched_lines = (
        original_lines[: proposal.start_line - 1]
        + replacement_lines
        + original_lines[proposal.end_line :]
    )
    patched_code = "\n".join(patched_lines)
    try:
        ast.parse(patched_code)
    except SyntaxError as exc:
        raise ValueError("patched program is not valid Python") from exc

    original = _program_version(
        graph.episode_id,
        proposal.block_id,
        original_code,
        parent_version_id=None,
    )
    patched = _program_version(
        graph.episode_id,
        proposal.block_id,
        patched_code,
        parent_version_id=original.id,
    )
    new_end_line = proposal.start_line + len(replacement_lines) - 1
    return PatchApplication(
        original=original,
        patched=patched,
        proposal=proposal,
        mapping=CodeVersionMapping(
            old_start_line=proposal.start_line,
            old_end_line=proposal.end_line,
            new_start_line=proposal.start_line,
            new_end_line=new_end_line,
        ),
    )


def write_stage4_patch_report(
    graph_path: str | Path,
    proposal_path: str | Path,
    output_path: str | Path,
) -> dict[str, Any]:
    proposal_bytes = Path(proposal_path).read_bytes()
    specification = json.loads(proposal_bytes)
    if not isinstance(specification, dict) or specification.get("schema_version") != 1:
        raise ValueError("Unsupported Stage 4 patch specification")
    if specification.get("prompt_variant") != GRAPHPTC_REPAIR_PROMPT_VARIANT:
        raise ValueError("Stage 4 patch requires prompt_variant='fewshot-ptc-v1'")
    episode_id = str(specification.get("episode_id") or "")
    graphs = [
        graph
        for graph in load_dependency_graph_report(graph_path)
        if graph.episode_id == episode_id
    ]
    if len(graphs) != 1:
        raise ValueError(f"Expected exactly one graph for episode: {episode_id}")
    graph = graphs[0]
    anchor_node_id = str(specification.get("anchor_node_id") or "")
    contexts = [
        context
        for context in build_failure_contexts(graph)
        if context.anchor.node_id == anchor_node_id
    ]
    if len(contexts) != 1:
        raise ValueError(f"Expected exactly one failure anchor: {anchor_node_id}")
    proposal_value = specification.get("proposal")
    if not isinstance(proposal_value, dict):
        raise ValueError("Stage 4 patch specification requires a proposal")
    try:
        proposal = LocalPatchProposal(**proposal_value)
    except TypeError as exc:
        raise ValueError("Invalid Stage 4 local patch proposal") from exc
    repair = build_repair_context(graph, contexts[0])
    application = apply_local_patch(graph, repair, proposal)
    report = {
        "schema_version": 1,
        "specification_sha256": hashlib.sha256(proposal_bytes).hexdigest(),
        "source_events_sha256": graph.source_events_sha256,
        "episode_id": graph.episode_id,
        "task_id": graph.task_id,
        "prompt_variant": repair.prompt_variant,
        "repair_context": repair.to_dict(),
        "application": application.to_dict(),
    }
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return report


def _block_node(graph: DependencyGraph, block_id: str) -> GraphNode:
    for node in graph.nodes:
        if node.type == "BLOCK" and node.block_id == block_id:
            return node
    raise ValueError(f"unknown block_id: {block_id}")


def _preferred_patch_region(
    graph: DependencyGraph,
    failure: FailureContext,
) -> CodeRegion | None:
    block_id = failure.anchor.block_id
    location = failure.anchor.location
    if block_id is None or location is None:
        return None
    block = _block_node(graph, block_id)
    lines = str(block.data.get("code", "")).splitlines()
    line = location["line"]
    if line < 1 or line > len(lines):
        return None
    return CodeRegion(
        block_id=block_id,
        start_line=line,
        end_line=line,
        focus_lines=(line,),
        code=lines[line - 1],
    )


def _program_version(
    episode_id: str,
    block_id: str,
    code: str,
    *,
    parent_version_id: str | None,
) -> ProgramVersion:
    code_sha256 = hashlib.sha256(code.encode("utf-8")).hexdigest()
    identity = json.dumps(
        {
            "episode_id": episode_id,
            "block_id": block_id,
            "parent_version_id": parent_version_id,
            "code_sha256": code_sha256,
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return ProgramVersion(
        id=f"program-version:{hashlib.sha256(identity).hexdigest()}",
        episode_id=episode_id,
        block_id=block_id,
        parent_version_id=parent_version_id,
        code_sha256=code_sha256,
        code=code,
    )
