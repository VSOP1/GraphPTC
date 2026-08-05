from __future__ import annotations

from pathlib import Path

from graphptc.stage2_graph import write_dependency_graph_report
from graphptc.stage5_commit_gate import write_stage5_commit_gate_report


def test_stage5_commit_gate_is_exact_and_deterministic(tmp_path: Path) -> None:
    root = Path(__file__).parents[2]
    graph_path = tmp_path / "graphs.json"
    output_path = tmp_path / "commit-gate.json"
    write_dependency_graph_report(
        root / "data" / "stage3" / "failure-audit.events.jsonl",
        graph_path,
    )

    first = write_stage5_commit_gate_report(
        graph_path,
        root / "configs" / "stage5.commit-gate.json",
        output_path,
    )
    first_bytes = output_path.read_bytes()
    second = write_stage5_commit_gate_report(
        graph_path,
        root / "configs" / "stage5.commit-gate.json",
        output_path,
    )

    assert first == second
    assert output_path.read_bytes() == first_bytes
    assert first["passed"] is True
    assert first["case_count"] == 4
    assert first["exact_match_rate"] == 1.0
    assert all(case["passed"] for case in first["cases"])
