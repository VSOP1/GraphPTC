from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from ..appworld.runtime import AppWorldProgramRuntime


class InterCodeProgramRuntime(AppWorldProgramRuntime):
    """Task-scoped client for one official InterCode episode."""

    def __init__(
        self,
        *,
        worker_command: Sequence[str],
        root: str,
        task_id: str,
        environment: str,
        data_path: str,
        data_index: int,
        image_name: str,
        container_prefix: str,
        max_actions: int,
        timeout_seconds: float,
    ) -> None:
        super().__init__(
            worker_command=worker_command,
            root=root,
            task_id=task_id,
            experiment_name="graphptc-intercode",
            timeout_seconds=timeout_seconds,
            initialization={
                "environment": environment,
                "data_path": data_path,
                "data_index": data_index,
                "image_name": image_name,
                "container_prefix": container_prefix,
                "max_actions": max_actions,
            },
            runtime_name="intercode",
            runtime_label="InterCode",
        )

    def _execution_trace(
        self, message: Mapping[str, Any], *, success: bool
    ) -> dict[str, Any]:
        values = message.get("environment_actions", [])
        actions = values if isinstance(values, list) else []
        external_actions = []
        for item in actions:
            if not isinstance(item, dict):
                continue
            tool = str(item.get("tool", "act"))
            command = str(item.get("command", ""))
            external_actions.append(
                {
                    "name": tool,
                    "arguments": {
                        "query" if tool == "sql" else "command": command,
                    },
                    "effect": str(item.get("effect", "read")),
                    "success": bool(item.get("action_executed", False))
                    if success
                    else None,
                    "outcome_unknown": not success,
                    "effect_basis": "intercode_action",
                    "observation": item.get("observation"),
                    "reward": item.get("reward"),
                }
            )
        return {
            "environment_actions": actions,
            "external_actions": external_actions,
            "completed": self.task_completed,
            "official_success": bool(message.get("official_success", False)),
            "max_reward": float(message.get("max_reward", 0.0) or 0.0),
            "actions": int(message.get("actions", 0) or 0),
        }
