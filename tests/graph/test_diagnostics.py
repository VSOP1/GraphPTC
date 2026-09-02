from __future__ import annotations

from graphptc.graph.diagnostics import graph_delta_sequence


def test_graph_delta_sequence_reports_temporal_order_without_claiming_causality() -> None:
    messages = [
        {"role": "user", "content": "task"},
        {"role": "tool", "content": "result\n\nGRAPH_DELTA {}"},
        {"role": "assistant", "tool_calls": [{"id": "call-2"}]},
    ]

    sequence = graph_delta_sequence(messages)

    assert sequence["graph_deltas"] == 1
    assert sequence["temporal_exposure_verified"] is True
    assert sequence["causal_influence_established"] is False
