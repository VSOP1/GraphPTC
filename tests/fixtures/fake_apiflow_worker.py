from __future__ import annotations

import json
import sys


counter = 0
abstained = False

for line in sys.stdin:
    request = json.loads(line)
    kind = request["type"]
    if kind == "initialize":
        response = {
            "type": "ready",
            "task_id": request["task_id"],
            "instruction": "complete the fake API task",
            "system_prompt": "official APIFlow prompt",
            "axis": "statefulness",
            "protocol": "rest",
        }
    elif kind == "call_tool":
        counter += 1
        name = request["name"]
        abstained = name in {"clarify", "report_blocked"}
        response = {
            "type": "tool_result",
            "success": True,
            "result": {"name": name, "counter": counter, "arguments": request["arguments"]},
            "effect": "read" if name in {"read", "search"} else "write",
            "task_completed": abstained,
        }
    elif kind == "evaluate":
        response = {
            "type": "evaluation",
            "passed": counter > 0,
            "reason": "fake verdict",
            "score": 1.0 if counter > 0 else 0.0,
            "predicates": [],
            "sub_predicates": {},
        }
    elif kind == "close":
        print(json.dumps({"type": "closed", "success": True}), flush=True)
        break
    else:
        response = {"type": "error", "error": f"unknown request: {kind}"}
    print(json.dumps(response), flush=True)
