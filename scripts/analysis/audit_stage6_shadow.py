from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from graphptc.config import ExperimentConfig
from graphptc.stage2_graph import build_dependency_graphs, load_execution_events


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Audit the Stage 6.1 live shadow integration smoke."
    )
    parser.add_argument("config_path", type=Path)
    parser.add_argument("responses_path", type=Path)
    parser.add_argument("events_path", type=Path)
    parser.add_argument("shadow_path", type=Path)
    parser.add_argument("controlled_loop_report_path", type=Path)
    parser.add_argument("output_path", type=Path)
    args = parser.parse_args()

    config = ExperimentConfig.from_toml(args.config_path)
    responses = _jsonl(args.responses_path)
    shadow_rows = _jsonl(args.shadow_path)
    events = load_execution_events(args.events_path)
    graphs = build_dependency_graphs(events)
    controlled = json.loads(
        args.controlled_loop_report_path.read_text(encoding="utf-8")
    )
    graph = graphs[0] if len(graphs) == 1 else None
    block_tool_counts = []
    if graph is not None:
        block_tool_counts = [
            sum(
                node.type == "TOOL" and node.block_id == block.block_id
                for node in graph.nodes
            )
            for block in graph.nodes
            if block.type == "BLOCK"
        ]
    response = responses[0] if len(responses) == 1 else {}
    shadow_row = shadow_rows[0] if len(shadow_rows) == 1 else {}
    shadow = shadow_row.get("shadow", {})
    controlled_replay = controlled.get("replay", {})
    controlled_commit = controlled.get("commit", {})
    checks = {
        "fewshot_prompt": config.browsecomp_plus.prompt_variant == "fewshot-ptc-v1",
        "one_terminal_response": len(responses) == 1
        and response.get("status") == "success",
        "one_complete_event_graph": len(graphs) == 1,
        "real_multi_tool_block": max(block_tool_counts, default=0) >= 2,
        "one_shadow_record": len(shadow_rows) == 1,
        "primary_status_preserved": shadow_row.get("primary_status")
        == response.get("status"),
        "primary_prediction_preserved": shadow_row.get("primary_prediction")
        == response.get("prediction"),
        "primary_record_unchanged": shadow_row.get("primary_record_unchanged") is True,
        "successful_episode_is_noop": shadow.get("status")
        == "no_repairable_failure",
        "successful_episode_zero_repair_requests": shadow.get("model_request_count") == 0,
        "successful_episode_zero_commit": shadow.get("commit") is None,
        "controlled_real_loop_passed": controlled.get("passed") is True,
        "controlled_real_tools_reused": controlled_replay.get(
            "reused_tool_call_count"
        )
        == controlled.get("controlled_fault", {}).get("source_tool_call_count"),
        "controlled_real_tools_not_reexecuted": controlled_replay.get(
            "executed_tool_call_count"
        )
        == 0,
        "controlled_real_loop_committed": controlled_commit.get("committed") is True,
    }
    report = {
        "schema_version": 1,
        "stage": "6.1",
        "mode": "shadow",
        "official_benchmark_result": False,
        "passed": all(checks.values()),
        "checks": checks,
        "artifacts": {
            "config_sha256": _sha256(args.config_path),
            "responses_sha256": _sha256(args.responses_path),
            "events_sha256": _sha256(args.events_path),
            "shadow_sha256": _sha256(args.shadow_path),
            "controlled_loop_report_sha256": _sha256(
                args.controlled_loop_report_path
            ),
        },
        "live_noop": {
            "example_id": response.get("example_id"),
            "prediction": response.get("prediction"),
            "block_count": len(block_tool_counts),
            "tool_call_count": sum(block_tool_counts),
            "multi_tool_block_count": sum(count >= 2 for count in block_tool_counts),
            "max_tools_in_block": max(block_tool_counts, default=0),
            "shadow_status": shadow.get("status"),
            "shadow_model_request_count": shadow.get("model_request_count"),
        },
        "controlled_repair": {
            "validation_kind": controlled.get("validation_kind"),
            "source_tool_call_count": controlled.get("controlled_fault", {}).get(
                "source_tool_call_count"
            ),
            "reused_tool_call_count": controlled_replay.get(
                "reused_tool_call_count"
            ),
            "executed_tool_call_count": controlled_replay.get(
                "executed_tool_call_count"
            ),
            "committed": controlled_commit.get("committed"),
        },
    }
    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    args.output_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
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


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


if __name__ == "__main__":
    main()
