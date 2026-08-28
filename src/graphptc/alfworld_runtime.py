from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from .appworld_runtime import AppWorldProgramRuntime


class AlfWorldProgramRuntime(AppWorldProgramRuntime):
    """Task-scoped client for one persistent official ALFWorld text episode."""

    def __init__(
        self,
        *,
        worker_command: Sequence[str],
        data_root: str,
        official_config_path: str,
        split: str,
        task_id: str,
        seed: int,
        max_steps: int,
        timeout_seconds: float = 100,
    ) -> None:
        super().__init__(
            worker_command=worker_command,
            root=data_root,
            task_id=task_id,
            experiment_name="graphptc-alfworld",
            timeout_seconds=timeout_seconds,
            initialization={
                "data_root": data_root,
                "config_path": official_config_path,
                "split": split,
                "seed": seed,
                "max_steps": max_steps,
            },
            runtime_name="alfworld",
            runtime_label="ALFWorld",
        )

    def _execution_trace(
        self, message: Mapping[str, Any], *, success: bool
    ) -> dict[str, Any]:
        actions = message.get("environment_actions", [])
        if not isinstance(actions, list):
            actions = []
        return {
            "environment_actions": actions,
            "external_actions": [
                {
                    "name": str(item.get("command", "")),
                    "arguments": {"command": str(item.get("command", ""))},
                    "effect": str(item.get("effect", "write")),
                    "success": bool(item.get("accepted", True)) if success else None,
                    "outcome_unknown": not success,
                    "effect_basis": "alfworld_command_family",
                    "observation": item.get("observation"),
                }
                for item in actions
                if isinstance(item, dict)
            ],
            "completed": self.task_completed,
            "won": bool(message.get("won", False)),
            "goal_condition_success_rate": float(
                message.get("goal_condition_success_rate", 0.0) or 0.0
            ),
            "steps": int(message.get("steps", 0) or 0),
        }
