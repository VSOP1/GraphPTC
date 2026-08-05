from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from graphptc.config import ExperimentConfig
from graphptc.local_search import OfficialCorpusSearchTools
from graphptc.model import ModelAttempt, ModelTurn, TokenUsage, ToolCall
from graphptc.persistent_runtime import PersistentIpcRuntime
from graphptc.stage2_graph import load_execution_events
from graphptc.stage6_active import repair_active_block


class FrozenPatchModel:
    def __init__(self, proposal: dict[str, Any]) -> None:
        self.proposal = proposal
        self.calls = 0

    def create_turn(self, **kwargs: Any) -> ModelTurn:
        self.calls += 1
        return ModelTurn(
            assistant_message={"role": "assistant", "content": None},
            text="",
            tool_calls=[
                ToolCall(
                    id="frozen-real-patch",
                    name="submit_local_patch",
                    input=self.proposal,
                )
            ],
            usage=TokenUsage(),
            stop_reason="tool_calls",
            attempts=(ModelAttempt(attempt=1, duration_ms=0, status="success"),),
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Audit Stage 6.2 against a frozen natural failure and real patch."
    )
    parser.add_argument("config_path", type=Path)
    parser.add_argument("events_path", type=Path)
    parser.add_argument("shadow_path", type=Path)
    parser.add_argument("output_path", type=Path)
    args = parser.parse_args()

    hashes_before = _hashes(args.config_path, args.events_path, args.shadow_path)
    config = ExperimentConfig.from_toml(args.config_path)
    events = load_execution_events(args.events_path)
    active_events = tuple(
        event for event in events if event.get("type") != "episode.finished"
    )
    shadow_row = _jsonl(args.shadow_path)[0]
    shadow = shadow_row["shadow"]
    proposal = shadow["generated_patch"]["proposal"]
    model = FrozenPatchModel(proposal)
    tools = OfficialCorpusSearchTools(
        config.browsecomp_plus.retriever_url,
        max_tool_calls=config.browsecomp_plus.max_tool_calls,
        timeout_seconds=config.browsecomp_plus.retriever_timeout_seconds,
    )
    runtime = PersistentIpcRuntime()
    try:
        active = repair_active_block(
            active_events,
            block_id=str(proposal["block_id"]),
            repair_model=model,
            live_tools={"search": tools.search, "fetch": tools.fetch},
            runtime=runtime,
            timeout_seconds=config.runtime.code_timeout_seconds,
        )
        followup = runtime.execute(
            "print(len(results2), len(results3))",
            timeout=config.runtime.code_timeout_seconds,
        )
    finally:
        runtime.close()
    hashes_after = _hashes(args.config_path, args.events_path, args.shadow_path)

    replay = active.get("replay", {})
    checks = {
        "fewshot_source_prompt": config.browsecomp_plus.prompt_variant
        == "fewshot-ptc-v1",
        "source_was_natural_failure": shadow.get("failure_count") == 1
        and shadow.get("repairable_failure_count") == 1,
        "source_used_one_real_repair_request": shadow.get("model_request_count") == 1,
        "frozen_real_patch_used_once": model.calls == 1,
        "active_repair_succeeded": active.get("status") == "repaired_active",
        "prior_results_reused": replay.get("reused_tool_call_count") == 4,
        "new_calls_executed": replay.get("executed_tool_call_count") == 2,
        "repaired_block_produced_output": bool(str(active.get("output", "")).strip()),
        "repaired_state_remained_live": followup.return_code == 0
        and followup.stdout.strip() == "5 5",
        "source_artifacts_unchanged": hashes_before == hashes_after,
    }
    report = {
        "schema_version": 1,
        "stage": "6.2",
        "mode": "frozen-natural-failure-active-replay",
        "official_benchmark_result": False,
        "passed": all(checks.values()),
        "checks": checks,
        "artifacts": hashes_after,
        "source_example_id": shadow_row.get("example_id"),
        "source_repair_model_request_count": shadow.get("model_request_count"),
        "new_repair_api_request_count": 0,
        "replay": replay,
        "followup_stdout": followup.stdout.strip(),
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
