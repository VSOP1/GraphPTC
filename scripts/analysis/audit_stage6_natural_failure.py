from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from graphptc.config import ExperimentConfig
from graphptc.invalidation import analyze_invalidation
from graphptc.local_search import OfficialCorpusSearchTools
from graphptc.patch_controller import LocalPatchProposal, apply_local_patch, build_repair_context
from graphptc.replay_commit import commit_selective_replay
from graphptc.failure_attribution import build_failure_contexts
from graphptc.stage2_graph import build_dependency_graphs, load_execution_events


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Revalidate a frozen Stage 6.1 natural-failure patch."
    )
    parser.add_argument("config_path", type=Path)
    parser.add_argument("responses_path", type=Path)
    parser.add_argument("events_path", type=Path)
    parser.add_argument("shadow_path", type=Path)
    parser.add_argument("output_path", type=Path)
    args = parser.parse_args()

    source_hashes_before = _hashes(
        args.config_path, args.responses_path, args.events_path, args.shadow_path
    )
    config = ExperimentConfig.from_toml(args.config_path)
    responses = _jsonl(args.responses_path)
    shadow_rows = _jsonl(args.shadow_path)
    graphs = build_dependency_graphs(load_execution_events(args.events_path))
    response = responses[0] if len(responses) == 1 else {}
    shadow = shadow_rows[0].get("shadow", {}) if len(shadow_rows) == 1 else {}
    graph = graphs[0] if len(graphs) == 1 else None

    commit = None
    contexts = []
    if graph is not None:
        contexts = build_failure_contexts(graph)
        proposal_data = shadow.get("generated_patch", {}).get("proposal")
        if len(contexts) == 1 and isinstance(proposal_data, dict):
            proposal = LocalPatchProposal(**proposal_data)
            repair = build_repair_context(graph, contexts[0])
            application = apply_local_patch(graph, repair, proposal)
            plan = analyze_invalidation(graph, application)
            tools = OfficialCorpusSearchTools(
                config.browsecomp_plus.retriever_url,
                max_tool_calls=config.browsecomp_plus.max_tool_calls,
                timeout_seconds=config.browsecomp_plus.retriever_timeout_seconds,
            )
            commit = commit_selective_replay(
                graph,
                application,
                plan,
                live_tools={"search": tools.search, "fetch": tools.fetch},
                timeout_seconds=config.runtime.code_timeout_seconds,
            )

    source_hashes_after = _hashes(
        args.config_path, args.responses_path, args.events_path, args.shadow_path
    )
    replay = None if commit is None else commit.replay
    new_events = (
        []
        if replay is None
        else [event for event in replay.tool_events if event.action == "EXECUTE_NEW"]
    )
    checks = {
        "fewshot_prompt": config.browsecomp_plus.prompt_variant == "fewshot-ptc-v1",
        "one_primary_response": len(responses) == 1,
        "one_complete_event_graph": len(graphs) == 1,
        "one_natural_runtime_failure": len(contexts) == 1
        and contexts[0].anchor.kind == "RUNTIME_ERROR",
        "exact_block_and_source_anchor": len(contexts) == 1
        and contexts[0].anchor.block_id is not None
        and contexts[0].anchor.location is not None,
        "one_real_repair_model_request": shadow.get("model_request_count") == 1,
        "frozen_patch_was_generated": isinstance(
            shadow.get("generated_patch", {}).get("proposal"), dict
        ),
        "revalidation_used_zero_model_requests": True,
        "selective_replay_succeeded": replay is not None and replay.success,
        "prior_tool_results_reused": replay is not None
        and replay.reused_tool_call_count > 0,
        "new_patch_calls_executed": replay is not None
        and replay.executed_tool_call_count == len(new_events)
        and len(new_events) > 0,
        "new_call_provenance_is_explicit": bool(new_events)
        and all(
            event.source_tool_node_id is None
            and event.source_block_id == contexts[0].anchor.block_id
            for event in new_events
        ),
        "replay_committed": commit is not None and commit.committed,
        "source_artifacts_unchanged": source_hashes_before == source_hashes_after,
        "primary_result_preserved": shadow_rows[0].get("primary_status")
        == response.get("status")
        and shadow_rows[0].get("primary_prediction") == response.get("prediction"),
    }
    report = {
        "schema_version": 1,
        "stage": "6.1",
        "mode": "natural-failure-frozen-patch-revalidation",
        "official_benchmark_result": False,
        "passed": all(checks.values()),
        "checks": checks,
        "artifacts": source_hashes_after,
        "example_id": response.get("example_id"),
        "primary_status": response.get("status"),
        "primary_prediction": response.get("prediction"),
        "failure": None if not contexts else contexts[0].anchor.to_dict(),
        "repair_model_request_count": shadow.get("model_request_count"),
        "revalidation_model_request_count": 0,
        "replay": None
        if replay is None
        else {
            "success": replay.success,
            "reused_tool_call_count": replay.reused_tool_call_count,
            "executed_tool_call_count": replay.executed_tool_call_count,
            "actions": [event.action for event in replay.tool_events],
            "new_call_count": len(new_events),
        },
        "commit": None
        if commit is None
        else {
            "committed": commit.committed,
            "execution_version_id": (
                None if commit.execution_version is None else commit.execution_version.id
            ),
            "event_count": len(commit.events),
            "node_count": 0 if commit.graph is None else len(commit.graph.nodes),
        },
    }
    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    args.output_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "passed": report["passed"],
                "check_count": len(checks),
                "output_path": str(args.output_path),
            },
            ensure_ascii=False,
        )
    )
    if not report["passed"]:
        raise SystemExit(1)


def _jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _hashes(*paths: Path) -> dict[str, str]:
    return {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}


if __name__ == "__main__":
    main()
