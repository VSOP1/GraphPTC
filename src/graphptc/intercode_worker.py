import contextlib
import builtins
import io
import json
import math
import os
import re
import subprocess
import sys
import traceback
from pathlib import Path
from typing import Any, Dict, List


OFFICIAL_COMMIT = "c3e46d827cfc9d4c704ec078f7abf9f41e3191d8"
BASH_DATA = tuple(f"data/nl2bash/nl2bash_fs_{number}.json" for number in range(1, 5))
SQL_DATA = "data/sql/spider/ic_spider_dev.json"

_SAFE_BUILTINS = {
    name: getattr(builtins, name)
    for name in (
        "Exception",
        "KeyError",
        "RuntimeError",
        "TypeError",
        "ValueError",
        "abs",
        "all",
        "any",
        "bool",
        "dict",
        "enumerate",
        "filter",
        "float",
        "format",
        "int",
        "isinstance",
        "len",
        "list",
        "map",
        "max",
        "min",
        "next",
        "print",
        "range",
        "repr",
        "reversed",
        "round",
        "set",
        "slice",
        "sorted",
        "str",
        "sum",
        "tuple",
        "type",
        "zip",
    )
}


def _send(payload: dict[str, Any]) -> None:
    print(json.dumps(payload, ensure_ascii=True, default=repr), flush=True)


def _records(root: Path, relative_path: str) -> list[dict[str, Any]]:
    value = json.loads((root / relative_path).read_text(encoding="utf-8"))
    if not isinstance(value, list) or any(not isinstance(item, dict) for item in value):
        raise ValueError(f"InterCode data is not a list of objects: {relative_path}")
    return value


def _task_specs(root: Path) -> list[dict[str, Any]]:
    tasks: list[dict[str, Any]] = []
    for filesystem, relative_path in enumerate(BASH_DATA, start=1):
        for index, record in enumerate(_records(root, relative_path)):
            tasks.append(
                {
                    "task_id": f"bash:fs{filesystem}:{index}",
                    "environment": "bash",
                    "data_path": relative_path,
                    "data_index": index,
                    "image_name": f"intercode-nl2bash-fs{filesystem}",
                    "filesystem": filesystem,
                    "hardness": None,
                    "database": None,
                }
            )
    for index, record in enumerate(_records(root, SQL_DATA)):
        tasks.append(
            {
                "task_id": f"sql:{index}",
                "environment": "sql",
                "data_path": SQL_DATA,
                "data_index": index,
                "image_name": "docker-env-sql",
                "filesystem": None,
                "hardness": record.get("hardness"),
                "database": record.get("db"),
            }
        )
    return tasks


def _inspection(request: dict[str, Any]) -> dict[str, Any]:
    root = Path(str(request["root"])).resolve()
    commit = subprocess.run(
        ["git", "-C", str(root), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    import docker

    client = docker.from_env()
    images = {
        name: bool(client.images.list(name=name))
        for name in (
            "intercode-nl2bash-fs1",
            "intercode-nl2bash-fs2",
            "intercode-nl2bash-fs3",
            "intercode-nl2bash-fs4",
            "docker-env-sql",
        )
    }
    tasks = _task_specs(root)
    return {
        "type": "inspection",
        "official_commit": commit,
        "expected_official_commit": OFFICIAL_COMMIT,
        "docker_version": client.version().get("Version"),
        "images": images,
        "bash_tasks": sum(item["environment"] == "bash" for item in tasks),
        "sql_tasks": sum(item["environment"] == "sql" for item in tasks),
        "tasks": tasks,
        "official_protocol": {
            "max_turns": 10,
            "reward_after_each_action": True,
            "bash_template": "v2",
            "bash_dialogue_limit": 7,
            "sql_template": "game_sql",
            "sql_dialogue_limit": 5,
        },
    }


def _preprocess_sql(record: Dict) -> List:
    return [f"use {record['db']}"]


class _Episode:
    def __init__(self, request: dict[str, Any]) -> None:
        self.root = Path(str(request["root"])).resolve()
        self.task_id = str(request["task_id"])
        self.environment = str(request["environment"])
        self.data_path = str(request["data_path"])
        self.data_index = int(request["data_index"])
        self.image_name = str(request["image_name"])
        self.container_prefix = re.sub(
            r"[^a-zA-Z0-9_.-]+", "-", str(request["container_prefix"])
        )
        self.max_actions = int(request["max_actions"])
        sys.path.insert(0, str(self.root))
        os.chdir(self.root)
        records = _records(self.root, self.data_path)
        self.record = records[self.data_index]
        self.env = self._create_env()
        self.env.reset(self.data_index)
        self.query = str(self.env.query)
        self.actions = 0
        self.invalid_actions = 0
        self.max_reward = 0.0
        self.official_success = False
        self.completed = False
        self._block_actions: list[dict[str, Any]] = []
        self.state: dict[str, Any] = {
            "observation": self.query,
            "reward": None,
            "done": False,
            "action": 0,
            "actions_remaining": self.max_actions,
            "action_executed": None,
        }
        tool = self.bash if self.environment == "bash" else self.sql
        self.namespace: dict[str, Any] = {
            tool.__name__: tool,
            "state": self.state,
            "json": json,
            "math": math,
            "re": re,
            "__builtins__": _SAFE_BUILTINS,
        }

    def _create_env(self) -> Any:
        from intercode.envs import BashEnv, SqlEnv

        if self.environment == "sql":
            return SqlEnv(
                image_name=self.image_name,
                data_path=str(self.root / self.data_path),
                preprocess=_preprocess_sql,
                verbose=False,
            )
        if self.environment != "bash":
            raise ValueError(f"unsupported InterCode environment: {self.environment}")

        import intercode.envs.ic_env as ic_env_module
        import intercode.envs.bash.bash_env as bash_module
        from intercode.utils import get_container as official_get_container

        def isolated_get_container(
            container_name: str, image_name: str, **kwargs: Any
        ) -> Any:
            role = "eval" if container_name.endswith("_eval") else "agent"
            return official_get_container(
                f"{self.container_prefix}-{role}", image_name, **kwargs
            )

        ic_env_module.get_container = isolated_get_container
        bash_module.get_container = isolated_get_container
        bash_module.IMAGE_TO_SETTINGS[self.image_name] = "/bin/bash"
        return BashEnv(
            image_name=self.image_name,
            data_path=str(self.root / self.data_path),
            verbose=False,
        )

    def ready(self) -> dict[str, Any]:
        return {
            "type": "ready",
            "task_id": self.task_id,
            "query": self.query,
            "environment": self.environment,
            "data_path": self.data_path,
            "data_index": self.data_index,
            "image_name": self.image_name,
            "filesystem": self._filesystem(),
            "hardness": self.record.get("hardness"),
            "database": self.record.get("db"),
            "max_actions": self.max_actions,
            "official_commit": OFFICIAL_COMMIT,
            "prompt_template": "v2" if self.environment == "bash" else "game_sql",
            "dialogue_limit": 7 if self.environment == "bash" else 5,
            "initial_state": dict(self.state),
        }

    def _filesystem(self) -> int | None:
        match = re.search(r"nl2bash_fs_(\d+)\.json$", self.data_path)
        return int(match.group(1)) if match else None

    def bash(self, command: str) -> dict[str, Any]:
        if self.environment != "bash":
            raise RuntimeError("bash() is unavailable in InterCode-SQL")
        return self._act(command, tool="bash")

    def sql(self, query: str) -> dict[str, Any]:
        if self.environment != "sql":
            raise RuntimeError("sql() is unavailable in InterCode-Bash")
        return self._act(query, tool="sql")

    def _act(self, command: str, *, tool: str) -> dict[str, Any]:
        if not isinstance(command, str) or not command.strip():
            raise ValueError(f"{tool} requires a non-empty string")
        if self.completed:
            return dict(self.state)
        observation, _, _, action_info = self.env.step(command)
        action_executed = bool(action_info.get("action_executed", False))
        _, reward, _, _ = self.env.step("submit")
        reward_value = float(reward or 0.0)
        self.actions += 1
        self.invalid_actions += int(not action_executed)
        self.max_reward = max(self.max_reward, reward_value)
        self.official_success = self.max_reward == 1.0
        self.completed = self.official_success or self.actions >= self.max_actions
        result = {
            "observation": observation,
            "reward": reward_value,
            "done": self.completed,
            "action": self.actions,
            "actions_remaining": max(0, self.max_actions - self.actions),
            "action_executed": action_executed,
        }
        self.state.update(result)
        self._block_actions.append(
            {
                "tool": tool,
                "command": command,
                "observation": observation,
                "reward": reward_value,
                "action_executed": action_executed,
                "effect": _effect(command, environment=self.environment),
                "done": self.completed,
            }
        )
        return result

    def execute(self, code: str) -> dict[str, Any]:
        self._block_actions = []
        output = io.StringIO()
        try:
            with contextlib.redirect_stdout(output):
                exec(
                    compile(code, "<intercode-ptc>", "exec"),
                    self.namespace,
                    self.namespace,
                )
            success = True
        except Exception:
            success = False
            traceback.print_exc(file=output)
        return {
            "type": "execution",
            "stdout": output.getvalue(),
            "success": success,
            "completed": self.completed,
            "official_success": self.official_success,
            "max_reward": self.max_reward,
            "actions": self.actions,
            "environment_actions": list(self._block_actions),
        }

    def evaluate(self) -> dict[str, Any]:
        return {
            "success": self.official_success,
            "max_reward": self.max_reward,
            "actions": self.actions,
            "max_actions": self.max_actions,
            "invalid_actions": self.invalid_actions,
            "error_percentage": (
                100.0 * self.invalid_actions / self.actions if self.actions else 0.0
            ),
            "completed": self.completed,
        }

    def close(self) -> None:
        if self.environment == "bash":
            self.env.close()
            return
        self.env.cur.close()
        self.env.cnx.close()


def _effect(command: str, *, environment: str) -> str:
    if environment == "sql":
        return "read"
    lowered = command.strip().lower()
    write_markers = (
        ">",
        " rm ",
        " mv ",
        " cp ",
        "touch ",
        "mkdir ",
        "chmod ",
        "chown ",
        "sed -i",
        "ln ",
    )
    padded = f" {lowered} "
    return "write" if any(marker in padded for marker in write_markers) else "read"


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
                f"unsupported InterCode worker request: {request.get('type')!r}"
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
                raise ValueError(f"unsupported InterCode command: {command_type!r}")
        episode.close()
        return 0
    except Exception as exc:
        _send({"type": "error", "error": f"{type(exc).__name__}: {exc}"})
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
