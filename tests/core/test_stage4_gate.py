from __future__ import annotations

from pathlib import Path
import json
from typing import Any

from graphptc.model import ModelTurn, TokenUsage, ToolCall
from graphptc.stage2_graph import write_dependency_graph_report
from graphptc.stage4_gate import (
    write_stage4_gate_report,
    write_stage4_model_gate_report,
)


class GateModel:
    def __init__(self, turns: list[ModelTurn]) -> None:
        self._turns = iter(turns)
        self.requests: list[dict[str, Any]] = []

    def create_turn(self, **kwargs: Any) -> ModelTurn:
        self.requests.append(kwargs)
        return next(self._turns)


def test_stage4_gate_passes_and_is_byte_deterministic(tmp_path: Path) -> None:
    root = Path(__file__).parents[2]
    graph_path = tmp_path / "graphs.json"
    output_path = tmp_path / "stage4-gate.json"
    write_dependency_graph_report(
        root / "data" / "stage3" / "failure-audit.events.jsonl",
        graph_path,
    )

    first = write_stage4_gate_report(
        graph_path,
        root / "configs" / "stage4.promotion-gate.json",
        output_path,
    )
    first_bytes = output_path.read_bytes()
    second = write_stage4_gate_report(
        graph_path,
        root / "configs" / "stage4.promotion-gate.json",
        output_path,
    )

    assert first == second
    assert output_path.read_bytes() == first_bytes
    assert first["passed"] is True
    assert first["positive_case_count"] == 5
    assert first["negative_case_count"] == 3
    assert first["patch_valid_rate"] == 1.0
    assert first["reexecution_success_rate"] == 1.0
    assert first["negative_rejection_rate"] == 1.0
    assert first["out_of_bounds_acceptance_count"] == 0
    assert all(case["passed"] for case in first["positive_cases"])
    assert all(case["passed"] for case in first["negative_cases"])


def test_stage4_model_gate_uses_one_request_per_case_and_passes(tmp_path: Path) -> None:
    root = Path(__file__).parents[2]
    expectations_path = root / "configs" / "stage4.promotion-gate.json"
    expectations = json.loads(expectations_path.read_text(encoding="utf-8"))
    turns = [
        ModelTurn(
            assistant_message={"role": "assistant", "content": None},
            text="",
            tool_calls=[
                ToolCall(
                    id=f"patch-{index}",
                    name="submit_local_patch",
                    input=case["proposal"],
                )
            ],
            usage=TokenUsage(input_tokens=100, output_tokens=20),
            stop_reason="tool_calls",
        )
        for index, case in enumerate(expectations["positive_cases"], 1)
    ]
    model = GateModel(turns)
    graph_path = tmp_path / "graphs.json"
    output_path = tmp_path / "model-gate.json"
    write_dependency_graph_report(
        root / "data" / "stage3" / "failure-audit.events.jsonl",
        graph_path,
    )

    report = write_stage4_model_gate_report(
        model,
        graph_path,
        expectations_path,
        output_path,
    )

    assert report["passed"] is True
    assert report["case_count"] == 5
    assert report["model_request_count"] == 5
    assert report["location_match_rate"] == 1.0
    assert report["patch_valid_rate"] == 1.0
    assert report["reexecution_success_rate"] == 1.0
    assert len(model.requests) == 5
    assert all(request["thinking"] == "disabled" for request in model.requests)
