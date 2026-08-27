from __future__ import annotations

import dataclasses
import hashlib
import importlib.metadata
import io
import json
import logging
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping

from dotenv import load_dotenv


_PROTOCOL_STDOUT = sys.stdout


def _emit(value: Mapping[str, Any]) -> None:
    print(
        json.dumps(dict(value), ensure_ascii=True, default=str),
        file=_PROTOCOL_STDOUT,
        flush=True,
    )


class OfficialSession:
    def __init__(self) -> None:
        self.root: Path | None = None
        self.task_manager: Any = None
        self.state_manager: Any = None
        self.task: Any = None
        self.setup_attempted = False
        self.cleaned = False
        self.cleanup_attempted = False
        self.cleanup_result: dict[str, Any] | None = None

    def inspect(self, request: Mapping[str, Any]) -> dict[str, Any]:
        self._prepare(request)
        from src.factory import MCPServiceFactory

        services = request.get("services") or [
            "filesystem",
            "notion",
            "github",
            "postgres",
            "playwright",
            "playwright_webarena",
        ]
        suites: dict[str, Any] = {}
        for service in services:
            manager = MCPServiceFactory.create_task_manager(
                str(service), task_suite=str(request.get("task_suite", "standard"))
            )
            tasks = manager.discover_all_tasks()
            suites[str(service)] = {
                "count": len(tasks),
                "tasks": [_task_record(task) for task in tasks],
            }
        return {
            "type": "inspection",
            "official_commit": _git(self.root, "rev-parse", "HEAD"),
            "official_tree": _git(self.root, "rev-parse", "HEAD^{tree}"),
            "official_dirty": (
                _git_returncode(
                    self.root, "diff", "--quiet", "--ignore-space-at-eol"
                )
                != 0
                or bool(
                    _git(
                        self.root,
                        "ls-files",
                        "--others",
                        "--exclude-standard",
                    )
                )
            ),
            "worktree_line_endings_differ": bool(
                _git(self.root, "status", "--porcelain")
            ),
            "python": sys.version,
            "pixi_lock_sha256": _file_hash(self.root / "pixi.lock"),
            "packages": _installed_packages(),
            "task_suite": str(request.get("task_suite", "standard")),
            "services": suites,
        }

    def initialize(self, request: Mapping[str, Any]) -> dict[str, Any]:
        self._prepare(request)
        from src.factory import MCPServiceFactory

        service = str(request["service"])
        suite = str(request.get("task_suite", "standard"))
        self.task_manager = MCPServiceFactory.create_task_manager(
            service, task_suite=suite
        )
        self.state_manager = MCPServiceFactory.create_state_manager(service)
        task_key = str(request["task_key"])
        tasks = self.task_manager.filter_tasks(task_key)
        exact = [task for task in tasks if _task_key(task) == task_key]
        if len(exact) != 1:
            raise ValueError(f"expected one exact task for {service}:{task_key}, found {len(exact)}")
        self.task = exact[0]
        self.setup_attempted = True
        capture, handler = _start_log_capture()
        try:
            setup_success = bool(self.state_manager.set_up(self.task))
        finally:
            _stop_log_capture(handler)
        response = {
            "type": "initialized",
            "setup_success": setup_success,
            "task": _task_record(self.task),
            "setup_log": capture.getvalue(),
            "state": _state_snapshot(self.state_manager, self.task),
        }
        if setup_success:
            response["instruction"] = self.task_manager.get_task_instruction(self.task)
            response["service_config"] = self.state_manager.get_service_config_for_agent()
        return response

    def verify(self, request: Mapping[str, Any]) -> dict[str, Any]:
        if self.task is None:
            raise RuntimeError("official session is not initialized")
        messages_path = str(request["messages_path"])
        self.state_manager.set_verification_environment(messages_path)
        capture, handler = _start_log_capture()
        try:
            try:
                result = self.task_manager.execute_task(
                    self.task,
                    {
                        "success": bool(request.get("agent_success", False)),
                        "error": request.get("agent_error"),
                        "output": [],
                        "token_usage": dict(request.get("token_usage") or {}),
                        "turn_count": int(request.get("turn_count", 0)),
                    },
                )
            finally:
                os.environ.pop("MCP_MESSAGES", None)
                os.environ.pop("MCP_GITHUB_TOKEN", None)
                os.environ.pop("PLAYWRIGHT_WORK_DIR", None)
                os.environ.pop("PLAYWRIGHT_BASE_URL", None)
        finally:
            _stop_log_capture(handler)
        payload = dataclasses.asdict(result)
        payload["model_output"] = None
        return {
            "type": "verification",
            "result": payload,
            "verification_log": capture.getvalue(),
            "state": _state_snapshot(self.state_manager, self.task),
        }

    def cleanup(self) -> dict[str, Any]:
        if self.cleanup_attempted:
            return {**dict(self.cleanup_result or {}), "already_attempted": True}
        self.cleanup_attempted = True
        success = True
        error: str | None = None
        capture, handler = _start_log_capture()
        try:
            if self.state_manager is not None and self.setup_attempted:
                try:
                    success = bool(self.state_manager.clean_up(self.task))
                except BaseException as exc:
                    success = False
                    error = f"{type(exc).__name__}: {exc}"
        finally:
            _stop_log_capture(handler)
        self.cleaned = success
        self.cleanup_result = {
            "type": "cleanup",
            "success": success,
            "cleanup_log": capture.getvalue(),
            "state": _state_snapshot(self.state_manager, self.task),
        }
        if error is not None:
            self.cleanup_result["error"] = error
        return dict(self.cleanup_result)

    def _prepare(self, request: Mapping[str, Any]) -> None:
        root = Path(str(request["root"])).resolve()
        if not (root / "pipeline.py").exists():
            raise ValueError(f"invalid MCPMark root: {root}")
        self.root = root
        os.chdir(root)
        if str(root) not in sys.path:
            sys.path.insert(0, str(root))
        load_dotenv(str(request["env_path"]), override=False)


def main() -> int:
    # MCPMark's logger writes to stdout. Keep the JSON-lines protocol clean.
    sys.stdout = sys.stderr
    session = OfficialSession()
    for line in sys.stdin:
        if not line.strip():
            continue
        try:
            request = json.loads(line)
            kind = request.get("type")
            if kind == "inspect":
                response = session.inspect(request)
            elif kind == "initialize":
                response = session.initialize(request)
            elif kind == "verify":
                response = session.verify(request)
            elif kind == "cleanup":
                response = session.cleanup()
            elif kind == "close":
                if not session.cleanup_attempted:
                    session.cleanup()
                _emit({"type": "closed"})
                return 0
            else:
                raise ValueError(f"unknown request type: {kind}")
            _emit(response)
        except BaseException as exc:
            _emit(
                {
                    "type": "error",
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                }
            )
    if not session.cleanup_attempted:
        session.cleanup()
    return 0


def _task_key(task: Any) -> str:
    return f"{task.category_id}/{task.task_id}"


def _task_record(task: Any) -> dict[str, Any]:
    description = Path(task.task_instruction_path)
    verifier = Path(task.task_verification_path)
    return {
        "service": str(task.service),
        "category_id": str(task.category_id),
        "task_id": str(task.task_id),
        "task_key": _task_key(task),
        "description_path": str(description),
        "description_sha256": _file_hash(description),
        "verifier_path": str(verifier),
        "verifier_sha256": _file_hash(verifier),
    }


def _state_snapshot(manager: Any, task: Any) -> dict[str, Any]:
    if manager is None:
        return {}
    task_vars = vars(task) if task is not None and hasattr(task, "__dict__") else {}
    task_values = {
        key: value
        for key, value in task_vars.items()
        if key not in {"task_instruction_path", "task_verification_path"}
    }
    return _redact(
        {
            "manager": type(manager).__name__,
            "task": task_values,
            "tracked_resources": list(getattr(manager, "tracked_resources", ())),
        }
    )


def _redact(value: Any) -> Any:
    if isinstance(value, Mapping):
        result: dict[str, Any] = {}
        for key, item in value.items():
            normalized = str(key).lower().replace("-", "_")
            if normalized in {"authorization", "password", "secret", "token", "api_key"} or normalized.endswith(
                ("_token", "_password", "_secret", "_api_key", "_key")
            ):
                result[str(key)] = "<redacted>"
            else:
                result[str(key)] = _redact(item)
        return result
    if isinstance(value, (list, tuple)):
        return [_redact(item) for item in value]
    return value


def _file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _start_log_capture() -> tuple[io.StringIO, logging.Handler]:
    capture = io.StringIO()
    handler = logging.StreamHandler(capture)
    handler.setFormatter(logging.Formatter("%(message)s"))
    logging.getLogger().addHandler(handler)
    return capture, handler


def _stop_log_capture(handler: logging.Handler) -> None:
    logging.getLogger().removeHandler(handler)


def _git(root: Path | None, *args: str) -> str:
    if root is None:
        return ""
    completed = subprocess.run(
        ("git", "-C", str(root), *args),
        capture_output=True,
        text=True,
        check=True,
    )
    return completed.stdout.strip()


def _git_returncode(root: Path, *args: str) -> int:
    return subprocess.run(
        ("git", "-C", str(root), *args),
        capture_output=True,
        text=True,
        check=False,
    ).returncode


def _installed_packages() -> dict[str, str]:
    packages: dict[str, str] = {}
    for distribution in importlib.metadata.distributions():
        name = distribution.metadata.get("Name")
        if name:
            packages[str(name).lower().replace("_", "-")] = distribution.version
    return dict(sorted(packages.items()))


if __name__ == "__main__":
    raise SystemExit(main())
