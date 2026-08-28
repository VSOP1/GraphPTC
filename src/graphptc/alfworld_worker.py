from __future__ import annotations

import contextlib
import hashlib
import io
import json
import os
import sys
import traceback
from importlib.metadata import version
from pathlib import Path
from typing import Any


def _send(payload: dict[str, Any]) -> None:
    print(json.dumps(payload, ensure_ascii=True, default=repr), flush=True)


def _load_config(path: str, data_root: str) -> dict[str, Any]:
    import yaml

    os.environ["ALFWORLD_DATA"] = data_root
    with Path(path).open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError("official ALFWorld config must contain a mapping")  # noqa: TRY004
    return config


def _split_path(config: dict[str, Any], split: str) -> Path:
    keys = {
        "train": "data_path",
        "eval_in_distribution": "eval_id_data_path",
        "eval_out_of_distribution": "eval_ood_data_path",
    }
    if split not in keys:
        raise ValueError(f"unsupported ALFWorld split: {split!r}")
    return Path(os.path.expandvars(str(config["dataset"][keys[split]]))).resolve()


def _task_id(game_file: str | Path, split_root: Path) -> str:
    path = Path(game_file).resolve()
    try:
        return path.parent.relative_to(split_root).as_posix()
    except ValueError as exc:
        raise ValueError(f"ALFWorld game is outside configured split: {path}") from exc


def _extract_task(initial_observation: str) -> tuple[str, str]:
    marker = "Your task is to: "
    if marker not in initial_observation:
        return initial_observation.strip(), initial_observation.strip()
    before, _, task = initial_observation.partition(marker)
    return task.strip(), before.strip()


def _effect(command: str) -> str:
    family = command.strip().lower().split(" ", 1)[0]
    return "read" if family in {"look", "inventory", "examine"} else "write"


def _official_defaults(config: dict[str, Any]) -> dict[str, Any]:
    return {
        "env_type": config["env"]["type"],
        "domain_randomization": config["env"]["domain_randomization"],
        "task_types": list(config["env"]["task_types"]),
        "random_seed": config["general"]["random_seed"],
        "training_method": config["general"]["training_method"],
        "eval_batch_size": config["general"]["evaluate"]["batch_size"],
        "dagger_action_space": config["dagger"]["action_space"],
        "max_steps": config["dagger"]["training"]["max_nb_steps_per_episode"],
        "num_eval_games": config["dataset"]["num_eval_games"],
    }


def _environment(config: dict[str, Any], split: str):
    from alfworld.agents.environment import get_environment

    with contextlib.redirect_stdout(sys.stderr):
        return get_environment(str(config["env"]["type"]))(config, train_eval=split)


def _inspection(request: dict[str, Any]) -> dict[str, Any]:
    config_path = str(request["config_path"])
    data_root = str(request["data_root"])
    split = str(request["split"])
    config = _load_config(config_path, data_root)
    split_root = _split_path(config, split)
    manager = _environment(config, split)
    tasks = []
    for game_file in manager.game_files:
        task_id = _task_id(game_file, split_root)
        tasks.append(
            {
                "task_id": task_id,
                "game_sha256": hashlib.sha256(Path(game_file).read_bytes()).hexdigest(),
            }
        )
    dataset_payload = json.dumps(tasks, sort_keys=True, separators=(",", ":"))
    return {
        "type": "inspection",
        "alfworld_version": version("alfworld"),
        "textworld_version": version("textworld"),
        "placement_command": "move OBJECT to RECEPTACLE",
        "config_sha256": hashlib.sha256(Path(config_path).read_bytes()).hexdigest(),
        "dataset_sha256": hashlib.sha256(dataset_payload.encode()).hexdigest(),
        "split": split,
        "split_root": str(split_root),
        "task_ids": [item["task_id"] for item in tasks],
        "task_count": len(tasks),
        "official_defaults": _official_defaults(config),
        "adapter_batch_size": 1,
    }


class _Episode:
    def __init__(self, request: dict[str, Any]) -> None:
        self.config = _load_config(
            str(request["config_path"]), str(request["data_root"])
        )
        self.split = str(request["split"])
        self.split_root = _split_path(self.config, self.split)
        self.task_id = str(request["task_id"])
        self.seed = int(request["seed"])
        self.max_steps = int(request["max_steps"])
        configured_steps = int(
            self.config["dagger"]["training"]["max_nb_steps_per_episode"]
        )
        if self.max_steps != configured_steps:
            raise ValueError(
                f"adapter max_steps={self.max_steps} does not match official config {configured_steps}"
            )
        manager = _environment(self.config, self.split)
        indexed = {
            _task_id(path, self.split_root): str(path) for path in manager.game_files
        }
        if self.task_id not in indexed:
            raise ValueError(f"unknown ALFWorld task ID: {self.task_id}")
        self.game_file = indexed[self.task_id]
        manager.game_files = [self.game_file]
        manager.num_games = 1
        with contextlib.redirect_stdout(sys.stderr):
            self.env = manager.init_env(batch_size=1)
            self.env.seed(self.seed)
            observations, infos = self.env.reset()
        self.task, initial_observation = _extract_task(str(observations[0]))
        self.state: dict[str, Any] = {
            "observation": initial_observation,
            "done": False,
            "step": 0,
            "steps_remaining": self.max_steps,
        }
        self.namespace: dict[str, Any] = {
            "act": self.act,
            "state": self.state,
            "__builtins__": __builtins__,
        }
        self.done = False
        self.won = False
        self.steps = 0
        self.goal_condition_success_rate = 0.0
        self._block_actions: list[dict[str, Any]] = []
        game_from_env = str(infos["extra.gamefile"][0])
        if Path(game_from_env).resolve() != Path(self.game_file).resolve():
            raise RuntimeError("official ALFWorld environment reset another game")

    def ready(self) -> dict[str, Any]:
        return {
            "type": "ready",
            "task_id": self.task_id,
            "task": self.task,
            "initial_state": dict(self.state),
            "game_sha256": hashlib.sha256(
                Path(self.game_file).read_bytes()
            ).hexdigest(),
            "alfworld_version": version("alfworld"),
            "textworld_version": version("textworld"),
            "placement_command": "move OBJECT to RECEPTACLE",
            "split": self.split,
            "seed": self.seed,
            "max_steps": self.max_steps,
            "action_space": self.config["dagger"]["action_space"],
            "adapter_batch_size": 1,
        }

    def act(self, command: str) -> dict[str, Any]:
        if self.done:
            raise RuntimeError("ALFWorld episode is already done")
        if not isinstance(command, str) or not command.strip():
            raise ValueError("act(command) requires a non-empty string")
        observations, _, dones, infos = self.env.step([command])
        self.steps += 1
        self.won = self.won or bool(infos["won"][0])
        if "goal_condition_success_rate" in infos:
            self.goal_condition_success_rate = max(
                self.goal_condition_success_rate,
                float(infos["goal_condition_success_rate"][0]),
            )
        self.done = bool(dones[0]) or self.steps >= self.max_steps
        result = {
            "observation": str(observations[0]),
            "done": self.done,
            "step": self.steps,
            "steps_remaining": max(0, self.max_steps - self.steps),
        }
        self.state.update(result)
        self._block_actions.append(
            {
                "command": command,
                "observation": result["observation"],
                "effect": _effect(command),
                "accepted": "Nothing happens" not in result["observation"],
                "done": self.done,
            }
        )
        return result

    def execute(self, code: str) -> dict[str, Any]:
        self._block_actions = []
        output = io.StringIO()
        try:
            with contextlib.redirect_stdout(output):
                exec(  # noqa: S102 - exact model code execution is the PTC contract.
                    compile(code, "<alfworld-ptc>", "exec"),
                    self.namespace,
                    self.namespace,
                )
            success = True
        except Exception:  # noqa: BLE001 - program errors become model observations.
            success = False
            traceback.print_exc(file=output)
        return {
            "type": "execution",
            "stdout": output.getvalue(),
            "success": success,
            "completed": self.done,
            "won": self.won,
            "goal_condition_success_rate": self.goal_condition_success_rate,
            "steps": self.steps,
            "environment_actions": list(self._block_actions),
        }

    def evaluate(self) -> dict[str, Any]:
        return {
            "success": self.won,
            "won": self.won,
            "goal_condition_success_rate": self.goal_condition_success_rate,
            "steps": self.steps,
            "done": self.done,
        }

    def close(self) -> None:
        self.env.close()


def main() -> int:
    first = sys.stdin.readline()
    if not first:
        return 1
    request = json.loads(first)
    try:
        if request.get("type") == "inspect":
            _send(_inspection(request))
            return 0
        if request.get("type") != "initialize":
            raise ValueError(
                f"unsupported ALFWorld worker request: {request.get('type')!r}"
            )
        episode = _Episode(request)
        _send(episode.ready())
        for line in sys.stdin:
            command = json.loads(line)
            command_type = command.get("type")
            if command_type == "execute":
                _send(episode.execute(str(command["code"])))
            elif command_type == "evaluate":
                _send({"type": "evaluation", "evaluation": episode.evaluate()})
            elif command_type == "close":
                episode.close()
                _send({"type": "closed"})
                return 0
            else:
                raise ValueError(f"unsupported ALFWorld command: {command_type!r}")
        episode.close()
        return 0
    except Exception as exc:  # noqa: BLE001 - worker serializes all boundary failures.
        _send({"type": "error", "error": f"{type(exc).__name__}: {exc}"})
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
