from __future__ import annotations

import json
import sys


def send(value: dict) -> None:
    print(json.dumps(value), flush=True)


request = json.loads(sys.stdin.readline())
assert request["type"] == "initialize"
send({"type": "ready", "domain": request["domain"], "sample_id": request["sample_id"], "tools": [], "tool_names": ["fake_tool"], "official_prompt": "official"})
for line in sys.stdin:
    request = json.loads(line)
    if request["type"] == "execute":
        if request["code"] == "fail()":
            send({"type": "execution", "stdout": "", "stderr": "ValueError: failed", "rc": 1, "external_actions": [], "state_effects": [], "artifacts": []})
        else:
            send({"type": "execution", "stdout": "ok\n", "stderr": "", "rc": 0, "external_actions": [{"tool": "fake_tool", "success": True}], "state_effects": [{"artifact": "cart.json"}], "artifacts": [{"kind": "tool_result"}]})
    elif request["type"] == "close":
        send({"type": "closed"})
        break
