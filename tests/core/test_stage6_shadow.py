from __future__ import annotations

from pathlib import Path
from typing import Any

from graphptc.model import ModelAttempt, ModelTurn, TokenUsage, ToolCall
from graphptc.stage2_graph import load_execution_events
from graphptc.stage6_shadow import analyze_shadow_episode


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
                    id="shadow-patch-1",
                    name="submit_local_patch",
                    input=self.arguments,
                )
            ],
            usage=TokenUsage(input_tokens=100, output_tokens=20),
            stop_reason="tool_calls",
            attempts=(
                ModelAttempt(
                    attempt=1,
                    duration_ms=1.0,
                    status="success",
                ),
            ),
        )


def _episode_events(episode_id: str):  # type: ignore[no-untyped-def]
    root = Path(__file__).parents[2]
    return tuple(
        event
        for event in load_execution_events(
            root / "data" / "stage3" / "failure-audit.events.jsonl"
        )
        if event["episode_id"] == episode_id
    )


def test_shadow_skips_successful_episode_without_model_request() -> None:
    events = list(_episode_events("audit-runtime"))
    events[-2]["data"]["success"] = True
    events[-2]["data"]["error_type"] = None
    events[-2]["data"]["error_message"] = None
    events[-1]["data"]["status"] = "success"
    events[-1]["data"]["error"] = None
    events[-1]["data"]["answer"] = "Alpha"

    result = analyze_shadow_episode(
        tuple(events),
        repair_model=None,
        live_tools={},
        timeout_seconds=5,
    )

    assert result["status"] == "no_repairable_failure"
    assert result["model_request_count"] == 0
    assert result["commit"] is None


def test_shadow_repairs_at_most_one_failure_and_commits_without_mutating_events() -> None:
    events = _episode_events("audit-runtime")
    snapshot = tuple(event.copy() for event in events)
    model = PatchModel(
        {
            "block_id": "audit-runtime:block:2",
            "start_line": 1,
            "end_line": 1,
            "expected_code": "print(hits[2]['title'])",
            "replacement_code": "print(hits[0]['title'])",
        }
    )

    result = analyze_shadow_episode(
        events,
        repair_model=model,
        live_tools={},
        timeout_seconds=5,
    )

    assert result["status"] == "committed_shadow"
    assert result["failure_count"] == 1
    assert result["selected_failure_index"] == 0
    assert result["model_request_count"] == 1
    assert result["replay"]["reused_tool_call_count"] == 1
    assert result["replay"]["executed_tool_call_count"] == 0
    assert result["commit"]["committed"] is True
    assert result["commit"]["execution_version_id"].startswith(
        "execution-version:"
    )
    assert model.calls == 1
    assert events == snapshot
