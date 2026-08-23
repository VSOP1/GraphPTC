from __future__ import annotations

import json
import sys


def send(payload: dict[str, object]) -> None:
    sys.stdout.write(json.dumps(payload) + "\n")
    sys.stdout.flush()


request = json.loads(sys.stdin.readline())
assert request["type"] == "initialize"
task_id = str(request["task_id"])
state: dict[str, object] = {"counter": 0}
send(
    {
        "type": "ready",
        "task_id": task_id,
        "instruction": f"instruction for {task_id}",
        "db_version": "fake-db",
        "appworld_version": "fake-appworld",
        "output_directory": f"/outputs/{task_id}",
    }
)

for line in sys.stdin:
    request = json.loads(line)
    request_type = request["type"]
    if request_type == "execute":
        code = str(request["code"])
        if code == "counter += 1":
            state["counter"] = int(state["counter"]) + 1
            output = "Execution successful."
            success = True
            completed = False
        elif code == "print(counter)":
            output = f"{state['counter']}\n"
            success = True
            completed = False
        elif code == "fail()":
            output = "Execution failed. Traceback:\nValueError: fake failure"
            success = False
            completed = False
        elif code == "apis.supervisor.complete_task()":
            output = "Execution successful."
            success = True
            completed = True
        else:
            output = code
            success = True
            completed = False
        send(
            {
                "type": "execution",
                "stdout": output,
                "success": success,
                "completed": completed,
                "api_calls": [
                    {"method": "post", "url": "/fake/action", "data": {"code": code}}
                ],
            }
        )
    elif request_type == "evaluate":
        send({"type": "evaluation", "evaluation": {"success": bool(state["counter"])}})
    elif request_type == "close":
        send({"type": "closed"})
        break
    else:
        raise AssertionError(request_type)
