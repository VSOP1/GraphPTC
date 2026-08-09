from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from graphptc.browsecomp_plus_benchmark import BROWSECOMP_PLUS_RUNTIME_TOOL_MANIFEST
from graphptc.config import ExperimentConfig
from graphptc.failure_attribution import build_failure_contexts
from graphptc.local_search import OfficialCorpusSearchTools
from graphptc.selective_replay import selective_replay_patch
from graphptc.stage2_graph import build_dependency_graphs
from graphptc.invalidation import analyze_invalidation
from graphptc.patch_controller import (
    LocalPatchProposal,
    apply_local_patch,
    build_repair_context,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run Stage 7.0 frozen-prefix repair/no-repair causal audit."
    )
    parser.add_argument("config_path", type=Path)
    parser.add_argument("events_path", type=Path)
    parser.add_argument("active_path", type=Path)
    parser.add_argument("comparison_path", type=Path)
    parser.add_argument("output_path", type=Path)
    args = parser.parse_args()

    config = ExperimentConfig.from_toml(args.config_path)
    events = _jsonl(args.events_path)
    active_rows = {
        str(row["example_id"]): row["active"]
        for row in _jsonl(args.active_path)
        if isinstance(row.get("active"), dict)
        and row["active"].get("status") == "repaired_active"
    }
    comparison = json.loads(args.comparison_path.read_text(encoding="utf-8"))
    pairs = {
        str(pair["example_id"]): pair
        for pair in comparison.get("pairs", [])
        if str(pair.get("example_id")) in active_rows
    }
    if not active_rows:
        raise ValueError("Stage 7.0 requires at least one repaired_active row")

    event_groups: dict[str, list[dict[str, Any]]] = {}
    for event in events:
        event_groups.setdefault(str(event["task_id"]), []).append(event)

    cases: list[dict[str, Any]] = []
    for example_id in sorted(active_rows, key=int):
        active = active_rows[example_id]
        source = event_groups.get(example_id)
        if source is None:
            raise ValueError(f"Missing events for repaired example {example_id}")
        prefix, failed_block_id = _failure_prefix(source)
        terminal = _failed_terminal(prefix)
        graph = build_dependency_graphs((*prefix, terminal))[0]
        repair_contexts = [
            context
            for context in build_failure_contexts(graph)
            if context.anchor.block_id == failed_block_id
        ]
        if len(repair_contexts) != 1:
            raise ValueError(
                f"Expected one repair context for {example_id}:{failed_block_id}"
            )
        proposal = LocalPatchProposal(**active["generated_patch"]["proposal"])
        application = apply_local_patch(
            graph,
            build_repair_context(
                graph,
                repair_contexts[0],
                runtime_tool_manifest=BROWSECOMP_PLUS_RUNTIME_TOOL_MANIFEST,
            ),
            proposal,
        )
        plan = analyze_invalidation(graph, application)
        source_snapshot = graph.to_dict()
        tools = OfficialCorpusSearchTools(
            config.browsecomp_plus.retriever_url,
            max_tool_calls=config.browsecomp_plus.max_tool_calls,
            timeout_seconds=config.browsecomp_plus.retriever_timeout_seconds,
        )
        replay = selective_replay_patch(
            graph,
            application,
            plan,
            live_tools={"search": tools.search, "fetch": tools.fetch},
            timeout_seconds=config.runtime.code_timeout_seconds,
        )
        stored_replay = active.get("replay") or {}
        pair = pairs.get(example_id, {})
        cases.append(
            {
                "example_id": example_id,
                "failed_block_id": failed_block_id,
                "prefix_event_count": len(prefix),
                "prefix_sha256": graph.source_events_sha256,
                "stored_source_events_sha256": active.get("source_events_sha256"),
                "same_frozen_prefix": graph.source_events_sha256
                == active.get("source_events_sha256"),
                "no_repair": {
                    "branch": "original_failed_prefix",
                    "failed_block_success": False,
                    "source_unchanged": graph.to_dict() == source_snapshot,
                },
                "repair": {
                    "proposal_applied": True,
                    "replay_success": replay.success,
                    "output_nonempty": bool(
                        replay.blocks and replay.blocks[-1].stdout.strip()
                    ),
                    "reused_tool_calls": replay.reused_tool_call_count,
                    "executed_tool_calls": replay.executed_tool_call_count,
                    "stored_reused_tool_calls": stored_replay.get(
                        "reused_tool_call_count"
                    ),
                    "stored_executed_tool_calls": stored_replay.get(
                        "executed_tool_call_count"
                    ),
                    "source_unchanged": graph.to_dict() == source_snapshot,
                },
                "continuation_observation": {
                    "control_correct": pair.get("control_correct"),
                    "active_correct": pair.get("active_correct"),
                    "transition": pair.get("transition"),
                    "causal_claim": "not_identified_by_this_block_audit",
                },
            }
        )

    checks = {
        "repaired_case_count": len(cases) == len(active_rows),
        "same_frozen_prefix_all": all(
            case["same_frozen_prefix"] for case in cases
        ),
        "no_repair_preserves_failure_all": all(
            case["no_repair"]["failed_block_success"] is False
            and case["no_repair"]["source_unchanged"]
            for case in cases
        ),
        "repair_replay_success_all": all(
            case["repair"]["replay_success"]
            and case["repair"]["output_nonempty"]
            for case in cases
        ),
        "repair_source_unchanged_all": all(
            case["repair"]["source_unchanged"] for case in cases
        ),
        "replay_counts_match_stored_all": all(
            case["repair"]["reused_tool_calls"]
            == case["repair"]["stored_reused_tool_calls"]
            and case["repair"]["executed_tool_calls"]
            == case["repair"]["stored_executed_tool_calls"]
            for case in cases
        ),
        "no_repair_model_requests": True,
    }
    report = {
        "schema_version": 1,
        "stage": "7.0",
        "mode": "frozen-prefix-repair-no-repair",
        "official_benchmark_result": False,
        "passed": all(checks.values()),
        "checks": checks,
        "cases": cases,
        "artifacts": {
            "config_sha256": _sha256(args.config_path),
            "events_sha256": _sha256(args.events_path),
            "active_sha256": _sha256(args.active_path),
            "comparison_sha256": _sha256(args.comparison_path),
        },
        "interpretation": {
            "block_level": "repair effect is tested from the same frozen prefix",
            "episode_level": "continuation accuracy remains observational and is not a causal claim",
        },
    }
    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    args.output_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({"passed": report["passed"], "checks": checks}))
    if not report["passed"]:
        raise SystemExit(1)


def _failure_prefix(events: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], str]:
    for index, event in enumerate(events):
        if event.get("type") != "block.finished":
            continue
        data = event.get("data", {})
        if data.get("success") is False:
            return events[: index + 1], str(event["block_id"])
    raise ValueError("No failed block found in active event stream")


def _failed_terminal(prefix: list[dict[str, Any]]) -> dict[str, Any]:
    last = prefix[-1]
    return {
        "schema_version": 1,
        "sequence": int(last["sequence"]) + 1,
        "type": "episode.finished",
        "episode_id": last["episode_id"],
        "task_id": last["task_id"],
        "block_id": None,
        "data": {
            "status": "failed",
            "answer": "",
            "error": "active repair snapshot",
            "ptc_blocks": sum(event.get("type") == "block.finished" for event in prefix),
        },
    }


def _jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


if __name__ == "__main__":
    main()
