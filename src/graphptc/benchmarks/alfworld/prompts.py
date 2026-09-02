from __future__ import annotations

import json
from typing import Any


def _tool_call(call_id: str, *, code: str, expected_change: str) -> dict[str, Any]:
    return {
        "role": "assistant",
        "content": "I will execute one coherent phase and inspect its compact result.",
        "tool_calls": [
            {
                "id": call_id,
                "type": "function",
                "function": {
                    "name": "programmatic_tool_call",
                    "arguments": json.dumps(
                        {
                            "code": code,
                            "action": "CONTINUE",
                            "target": "task",
                            "expected_change": expected_change,
                        }
                    ),
                },
            }
        ],
    }


ALFWORLD_PTC_FEW_SHOT_MESSAGES: tuple[dict[str, Any], ...] = (
    {
        "role": "user",
        "content": (
            "Demonstration-only ALFWorld task: put sample 1 in box 1.\n\n"
            "All object names and observations below are synthetic; read the real task's values "
            "from its own initial observation and action results."
        ),
    },
    _tool_call(
        "alfworld_demo_locate",
        code=(
            'seen = act("look")\n'
            'at_table = act("go to worktable 1")\n'
            'print(at_table["observation"])'
        ),
        expected_change="locate the synthetic sample and reach its receptacle",
    ),
    {
        "role": "tool",
        "tool_call_id": "alfworld_demo_locate",
        "content": (
            "On the worktable 1, you see a sample 1.\n\n"
            'GRAPH_DELTA {"declared_action":{"action":"CONTINUE","target":"task"},'
            '"action_verification":{"realized":true}}'
        ),
    },
    _tool_call(
        "alfworld_demo_move",
        code=(
            "commands = [\n"
            '    "take sample 1 from worktable 1",\n'
            '    "go to box 1",\n'
            '    "open box 1",\n'
            '    "move sample 1 to box 1",\n'
            "]\n"
            "for command in commands:\n"
            "    outcome = act(command)\n"
            '    if outcome["done"]:\n'
            "        break\n"
            'print(outcome["observation"])'
        ),
        expected_change="move the synthetic sample into the target box",
    ),
    {
        "role": "tool",
        "tool_call_id": "alfworld_demo_move",
        "content": "You move the sample 1 to the box 1.\nExecution successful.",
    },
)
