from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any


def graph_delta_sequence(messages: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    delta_positions = [
        index
        for index, message in enumerate(messages)
        if message.get("role") == "tool"
        and "GRAPH_DELTA " in str(message.get("content", ""))
    ]
    later_actions = 0
    for position in delta_positions:
        if any(
            message.get("role") == "assistant" and message.get("tool_calls")
            for message in messages[position + 1 :]
        ):
            later_actions += 1
    return {
        "graph_deltas": len(delta_positions),
        "deltas_preceding_later_action": later_actions,
        "temporal_exposure_verified": bool(delta_positions) and later_actions > 0,
        "causal_influence_established": False,
        "causal_note": "Temporal order and action alignment do not identify counterfactual influence.",
    }
