from __future__ import annotations

from pathlib import Path
from typing import Any

from graphptc.model import ModelAttempt, ModelTurn, TokenUsage, ToolCall
from graphptc.persistent_runtime import PersistentIpcRuntime
from graphptc.stage2_graph import load_execution_events
from graphptc.stage6_active import repair_active_block


class PatchModel:
    def __init__(self, arguments: dict[str, Any]) -> None:
        self.arguments = arguments
        self.calls = 0

    def create_turn(self, **kwargs: Any) -> ModelTurn:
        self.calls += 1
        return ModelTurn(
            assistant_message={"role": "assistant", "content": None},
            text="",
            tool_calls=[
                ToolCall(
                    id="active-patch-1",
                    name="submit_local_patch",
                    input=self.arguments,
                )
            ],
            usage=TokenUsage(input_tokens=100, output_tokens=20),
            stop_reason="tool_calls",
            attempts=(ModelAttempt(attempt=1, duration_ms=1.0, status="success"),),
        )


def test_active_repair_retains_replayed_state_in_supplied_runtime() -> None:
    root = Path(__file__).parents[2]
    events = tuple(
        event
        for event in load_execution_events(
            root / "data" / "stage3" / "failure-audit.events.jsonl"
        )
        if event["episode_id"] == "audit-runtime"
        and event["type"] != "episode.finished"
    )
    model = PatchModel(
        {
            "block_id": "audit-runtime:block:2",
            "start_line": 1,
            "end_line": 1,
            "expected_code": "print(hits[2]['title'])",
            "replacement_code": "print(hits[0]['title'])",
        }
    )
    runtime = PersistentIpcRuntime()
    try:
        result = repair_active_block(
            events,
            block_id="audit-runtime:block:2",
            repair_model=model,
            live_tools={},
            runtime=runtime,
            timeout_seconds=5,
        )
        followup = runtime.execute("print(hits[0]['title'])", timeout=5)
    finally:
        runtime.close()

    assert result["status"] == "repaired_active"
    assert result["output"].strip() == "Alpha"
    assert result["replay"]["reused_tool_call_count"] == 1
    assert followup.stdout.strip() == "Alpha"
    assert model.calls == 1
