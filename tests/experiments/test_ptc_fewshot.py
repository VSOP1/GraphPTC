from __future__ import annotations

import json

from graphptc.experiments.ptc_fewshot import PTC_FEW_SHOT_MESSAGES


def test_fewshot_messages_form_complete_tool_protocol() -> None:
    call_ids: set[str] = set()
    result_ids: set[str] = set()
    for message in PTC_FEW_SHOT_MESSAGES:
        for call in message.get("tool_calls", []):
            assert call["function"]["name"] == "programmatic_tool_call"
            arguments = json.loads(call["function"]["arguments"])
            assert "search(" in arguments["code"]
            assert "print(" in arguments["code"]
            call_ids.add(call["id"])
        if message["role"] == "tool":
            result_ids.add(message["tool_call_id"])
    assert call_ids == result_ids
    assert len(call_ids) == 2
