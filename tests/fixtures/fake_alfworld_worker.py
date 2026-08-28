from __future__ import annotations

import contextlib
import io
import json
import sys
import traceback
from typing import Any


def send(payload: dict[str, Any]) -> None:
    print(json.dumps(payload), flush=True)


def main() -> int:
    request = json.loads(sys.stdin.readline())
    if request["type"] != "initialize":
        return 2
    state = {
        "observation": "A fake room.",
        "done": False,
        "step": 0,
        "steps_remaining": 3,
    }
    actions: list[dict[str, Any]] = []
    won = False

    def act(command: str) -> dict[str, Any]:
        nonlocal won
        state["step"] += 1
        state["steps_remaining"] -= 1
        state["observation"] = f"observed {command}"
        if command == "finish":
            state["done"] = True
            won = True
        actions.append(
            {
                "command": command,
                "observation": state["observation"],
                "effect": "read" if command == "look" else "write",
                "accepted": True,
            }
        )
        return dict(state)

    namespace = {"act": act, "state": state, "counter": 0}
    send(
        {
            "type": "ready",
            "task_id": request["task_id"],
            "task": "finish the fake task",
            "initial_state": dict(state),
            "alfworld_version": "fake",
        }
    )
    for line in sys.stdin:
        message = json.loads(line)
        if message["type"] == "execute":
            actions.clear()
            output = io.StringIO()
            try:
                with contextlib.redirect_stdout(output):
                    exec(message["code"], namespace, namespace)  # noqa: S102
                success = True
            except Exception:  # noqa: BLE001 - fake runtime returns program failures.
                success = False
                traceback.print_exc(file=output)
            send(
                {
                    "type": "execution",
                    "stdout": output.getvalue(),
                    "success": success,
                    "completed": state["done"],
                    "won": won,
                    "goal_condition_success_rate": 1.0 if won else 0.0,
                    "steps": state["step"],
                    "environment_actions": list(actions),
                }
            )
        elif message["type"] == "evaluate":
            send(
                {
                    "type": "evaluation",
                    "evaluation": {
                        "success": won,
                        "won": won,
                        "goal_condition_success_rate": 1.0 if won else 0.0,
                        "steps": state["step"],
                        "done": state["done"],
                    },
                }
            )
        elif message["type"] == "close":
            send({"type": "closed"})
            return 0
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
