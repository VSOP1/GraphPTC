from __future__ import annotations

import json
import sys


service_state = 0
closed = False


def emit(payload: dict) -> None:
    print(json.dumps(payload), flush=True)


for raw in sys.stdin:
    request = json.loads(raw)
    kind = request["type"]
    if kind == "inspect":
        emit(
            {
                "type": "inspection",
                "sdk_version": "fake-agent-diff",
                "official_commit": request["official_commit"],
                "base_url": "http://fake",
            }
        )
    elif kind == "initialize":
        emit(
            {
                "type": "ready",
                "environment_id": f"env-{request['task']['test_id']}-{request['trial']}",
                "run_id": f"run-{request['task']['test_id']}-{request['trial']}",
                "sdk_version": "fake-agent-diff",
                "official_commit": request["official_commit"],
                "python_state_persistent": False,
            }
        )
    elif kind == "execute":
        code = request["code"]
        if code == "write_service_state()":
            service_state += 1
            emit(
                {
                    "type": "execution",
                    "status": "success",
                    "stdout": "written\n",
                    "stderr": "",
                    "exit_code": 0,
                    "external_actions": [
                        {
                            "name": "POST /items",
                            "arguments": {"method": "POST", "url": "/items"},
                            "effect": "write",
                            "success": True,
                            "outcome_unknown": False,
                            "effect_basis": "official_state_diff",
                        }
                    ],
                    "state_effects": [{"diff_type": "added", "entity": "items", "count": 1}],
                }
            )
        elif code == "print_service_state()":
            emit(
                {
                    "type": "execution",
                    "status": "success",
                    "stdout": f"{service_state}\n",
                    "stderr": "",
                    "exit_code": 0,
                    "external_actions": [],
                    "state_effects": [],
                }
            )
        elif code == "long_output()":
            emit(
                {
                    "type": "execution",
                    "status": "success",
                    "stdout": "x" * 9000,
                    "stderr": "",
                    "exit_code": 0,
                    "external_actions": [],
                    "state_effects": [],
                }
            )
        elif code == "fail()":
            emit(
                {
                    "type": "execution",
                    "status": "error",
                    "stdout": "",
                    "stderr": "ValueError: failed request",
                    "exit_code": 1,
                    "external_actions": [],
                    "state_effects": [],
                }
            )
        else:
            emit(
                {
                    "type": "execution",
                    "status": "success",
                    "stdout": "ok\n",
                    "stderr": "",
                    "exit_code": 0,
                    "external_actions": [],
                    "state_effects": [],
                }
            )
    elif kind == "evaluate":
        emit(
            {
                "type": "evaluation",
                "evaluation": {
                    "passed": service_state == 1,
                    "score": 1.0 if service_state == 1 else 0.0,
                    "satisfied_assertions": 1 if service_state == 1 else 0,
                    "total_assertions": 1,
                    "clean": True,
                },
                "official_diff": {"inserts": [{"entity": "items"}], "updates": [], "deletes": []},
            }
        )
    elif kind == "close":
        closed = True
        emit({"type": "closed", "environment_deleted": True})
        break
    else:
        emit({"type": "error", "error": f"unknown request: {kind}"})


sys.exit(0 if closed else 1)
