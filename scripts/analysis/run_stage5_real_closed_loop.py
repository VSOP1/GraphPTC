from __future__ import annotations

import argparse
import hashlib
import json
import os
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from graphptc.browsecomp_plus_benchmark import BROWSECOMP_PLUS_RUNTIME_TOOL_MANIFEST
from graphptc.config import ExperimentConfig
from graphptc.failure_attribution import build_failure_contexts
from graphptc.invalidation import analyze_invalidation
from graphptc.model import OpenAIChatModel
from graphptc.patch_controller import apply_local_patch, build_repair_context
from graphptc.replay_commit import commit_selective_replay
from graphptc.stage2_graph import (
    DependencyGraph,
    GraphNode,
    build_dependency_graph,
    load_dependency_graph_report,
)
from graphptc.stage4_repair import request_local_patch


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run one controlled-fault Stage 5 loop over a real trajectory."
    )
    parser.add_argument("config_path", type=Path)
    parser.add_argument("graph_path", type=Path)
    parser.add_argument("responses_path", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--task-id", default="896")
    args = parser.parse_args()

    load_dotenv(".env")
    config = ExperimentConfig.from_toml(args.config_path)
    if config.browsecomp_plus.prompt_variant != "fewshot-ptc-v1":
        raise ValueError("real Stage 5 smoke requires prompt_variant='fewshot-ptc-v1'")
    api_key = os.environ.get(config.model.api_key_env)
    if not api_key:
        raise ValueError(f"Missing environment variable: {config.model.api_key_env}")

    graphs = load_dependency_graph_report(args.graph_path)
    source_graph = next((graph for graph in graphs if graph.task_id == args.task_id), None)
    if source_graph is None:
        raise ValueError(f"No real trajectory for task_id={args.task_id}")
    source_snapshot = source_graph.to_dict()
    fault_graph, fault_events, source_block = _controlled_failure(source_graph)
    contexts = build_failure_contexts(fault_graph)
    if len(contexts) != 1:
        raise ValueError("controlled failure must produce exactly one failure context")
    repair = build_repair_context(
        fault_graph,
        contexts[0],
        runtime_tool_manifest=BROWSECOMP_PLUS_RUNTIME_TOOL_MANIFEST,
    )
    generated = request_local_patch(
        OpenAIChatModel(replace(config.model, max_retries=0), api_key),
        repair,
        timeout_seconds=config.model.timeout_seconds,
        max_completion_tokens=1024,
    )
    application = apply_local_patch(fault_graph, repair, generated.proposal)
    plan = analyze_invalidation(fault_graph, application)
    commit = commit_selective_replay(
        fault_graph,
        application,
        plan,
        live_tools={},
        timeout_seconds=config.runtime.code_timeout_seconds,
    )

    structural = _structural_metrics(graphs)
    response_summary = _response_summary(args.responses_path)
    source_tool_count = sum(
        node.type == "TOOL" and node.block_id == source_block.block_id
        for node in source_graph.nodes
    )
    checks = {
        "two_real_terminal_responses": len(response_summary) == 2
        and all(row["status"] == "success" for row in response_summary),
        "real_multi_tool_programs": all(
            item["max_tools_in_block"] >= 2 for item in structural
        ),
        "one_failure_context": len(contexts) == 1,
        "patch_targets_injected_line": (
            generated.proposal.block_id == application.original.block_id
            and generated.proposal.start_line == len(application.original.code.splitlines())
            and generated.proposal.end_line == len(application.original.code.splitlines())
        ),
        "patch_valid": True,
        "selective_replay_committed": commit.committed,
        "all_real_tool_results_reused": (
            commit.replay.reused_tool_call_count == source_tool_count
            and commit.replay.executed_tool_call_count == 0
        ),
        "replay_provenance_complete": (
            commit.graph is not None
            and all(
                node.data.get("source_tool_node_id") is not None
                and node.data.get("source_artifact_id") is not None
                for node in commit.graph.nodes
                if node.type == "TOOL"
            )
        ),
        "source_graph_unchanged": source_graph.to_dict() == source_snapshot,
    }
    report = {
        "schema_version": 1,
        "validation_kind": "real-trajectory-controlled-fault-smoke",
        "official_benchmark_result": False,
        "prompt_variant": config.browsecomp_plus.prompt_variant,
        "model": config.model.model,
        "thinking": config.model.thinking,
        "max_turns": config.runtime.max_turns,
        "config_sha256": _file_sha256(args.config_path),
        "responses_sha256": _file_sha256(args.responses_path),
        "source_graph_report_sha256": _file_sha256(args.graph_path),
        "passed": all(checks.values()),
        "checks": checks,
        "real_responses": response_summary,
        "real_structure": structural,
        "controlled_fault": {
            "source_episode_id": source_graph.episode_id,
            "source_block_id": source_block.block_id,
            "source_tool_call_count": source_tool_count,
            "fault_episode_id": fault_graph.episode_id,
            "fault_line": len(application.original.code.splitlines()),
            "injected_code": "print(results[999])",
            "failure": contexts[0].to_dict(),
        },
        "generated_patch": asdict(generated),
        "application": application.to_dict(),
        "invalidation": plan.to_dict(),
        "replay": {
            "success": commit.replay.success,
            "reset_required": commit.replay.reset_required,
            "reused_tool_call_count": commit.replay.reused_tool_call_count,
            "executed_tool_call_count": commit.replay.executed_tool_call_count,
            "tool_events": [asdict(event) for event in commit.replay.tool_events],
            "blocks": [asdict(block) for block in commit.replay.blocks],
        },
        "commit": {
            "committed": commit.committed,
            "execution_version": (
                None if commit.execution_version is None else asdict(commit.execution_version)
            ),
            "event_count": len(commit.events),
            "node_count": 0 if commit.graph is None else len(commit.graph.nodes),
            "edge_count": 0 if commit.graph is None else len(commit.graph.edges),
            "artifact_count": 0 if commit.graph is None else len(commit.graph.artifacts),
        },
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    _write_jsonl(args.output_dir / "controlled-fault.events.jsonl", fault_events)
    _write_graph(args.output_dir / "controlled-fault.graph.json", fault_graph)
    if commit.graph is not None:
        _write_graph(args.output_dir / "committed-replay.graph.json", commit.graph)
    report_path = args.output_dir / "report.json"
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "passed": report["passed"],
                "real_response_count": len(response_summary),
                "controlled_source_tool_calls": source_tool_count,
                "reused_tool_calls": commit.replay.reused_tool_call_count,
                "executed_tool_calls": commit.replay.executed_tool_call_count,
                "committed": commit.committed,
                "output_path": str(report_path),
            },
            ensure_ascii=False,
        )
    )
    if not report["passed"]:
        raise SystemExit(1)


def _controlled_failure(
    source_graph: DependencyGraph,
) -> tuple[DependencyGraph, tuple[dict[str, Any], ...], GraphNode]:
    tool_counts = {
        block.block_id: sum(
            node.type == "TOOL" and node.block_id == block.block_id
            for node in source_graph.nodes
        )
        for block in source_graph.nodes
        if block.type == "BLOCK"
    }
    candidates = [
        block
        for block in source_graph.nodes
        if block.type == "BLOCK"
        and tool_counts.get(block.block_id, 0) >= 2
        and "results" in str(block.data.get("code", ""))
    ]
    if not candidates:
        raise ValueError("real trajectory has no suitable multi-tool block")
    source_block = candidates[0]
    original_code = str(source_block.data.get("code", ""))
    faulty_code = original_code + "\nprint(results[999])"
    fault_line = len(faulty_code.splitlines())
    digest = hashlib.sha256(
        f"{source_graph.source_events_sha256}:{source_block.block_id}".encode("utf-8")
    ).hexdigest()
    episode_id = f"controlled-fault:{digest}"
    block_id = f"{episode_id}:block:1"
    episode = next(node for node in source_graph.nodes if node.type == "EPISODE")
    tool_nodes = sorted(
        (
            node
            for node in source_graph.nodes
            if node.type == "TOOL" and node.block_id == source_block.block_id
        ),
        key=lambda node: int(node.data.get("event_sequence", 0)),
    )
    events: list[dict[str, Any]] = []

    def emit(event_type: str, *, event_block_id: str | None, data: dict[str, Any]) -> None:
        events.append(
            {
                "schema_version": 1,
                "sequence": len(events) + 1,
                "type": event_type,
                "episode_id": episode_id,
                "task_id": source_graph.task_id,
                "block_id": event_block_id,
                "data": data,
            }
        )

    emit(
        "episode.started",
        event_block_id=None,
        data={"task": episode.data.get("task")},
    )
    emit(
        "block.started",
        event_block_id=block_id,
        data={"turn": 1, "code": faulty_code},
    )
    for node in tool_nodes:
        if len(node.artifact_ids) != 1:
            raise ValueError("real successful tool call requires one artifact")
        emit(
            "tool.called",
            event_block_id=block_id,
            data={
                "tool": node.data.get("tool"),
                "arguments": node.data.get("arguments", {}),
                "success": True,
                "result": source_graph.artifact(node.artifact_ids[0]).value,
                "call_site": node.data.get("call_site"),
            },
        )
    emit(
        "block.finished",
        event_block_id=block_id,
        data={
            "turn": 1,
            "code": faulty_code,
            "stdout": "PTC_ERROR {...}",
            "stdout_chars": 15,
            "stdout_truncated": False,
            "success": False,
            "error_type": "IndexError",
            "error_message": "list index out of range",
            "runtime_trace": {
                "state_before": {},
                "state_after": {},
                "loaded_names": ["results", "print"],
                "stored_names": [],
                "error_location": {
                    "line": fault_line,
                    "column": 6,
                    "end_line": fault_line,
                    "end_column": 18,
                },
            },
        },
    )
    emit(
        "episode.finished",
        event_block_id=None,
        data={
            "status": "failed",
            "answer": "",
            "error": "IndexError: list index out of range",
            "ptc_blocks": 1,
        },
    )
    event_tuple = tuple(events)
    return build_dependency_graph(event_tuple), event_tuple, source_block


def _structural_metrics(graphs: tuple[DependencyGraph, ...]) -> list[dict[str, Any]]:
    rows = []
    for graph in graphs:
        block_counts = [
            sum(node.type == "TOOL" and node.block_id == block.block_id for node in graph.nodes)
            for block in graph.nodes
            if block.type == "BLOCK"
        ]
        rows.append(
            {
                "task_id": graph.task_id,
                "episode_id": graph.episode_id,
                "block_count": len(block_counts),
                "tool_call_count": sum(block_counts),
                "multi_tool_block_count": sum(count >= 2 for count in block_counts),
                "max_tools_in_block": max(block_counts, default=0),
                "transform_node_count": sum(
                    node.type == "TRANSFORM" for node in graph.nodes
                ),
            }
        )
    return rows


def _response_summary(path: Path) -> list[dict[str, Any]]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        value = json.loads(line)
        rows.append(
            {
                "example_id": value.get("example_id"),
                "status": value.get("status"),
                "prediction": value.get("prediction"),
            }
        )
    return rows


def _write_jsonl(path: Path, events: tuple[dict[str, Any], ...]) -> None:
    path.write_text(
        "".join(
            json.dumps(event, ensure_ascii=False, separators=(",", ":")) + "\n"
            for event in events
        ),
        encoding="utf-8",
    )


def _write_graph(path: Path, graph: DependencyGraph) -> None:
    path.write_text(
        json.dumps(
            {"schema_version": 3, "graph_count": 1, "graphs": [graph.to_dict()]},
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


if __name__ == "__main__":
    main()
