from __future__ import annotations

import asyncio
import json
import os
import platform
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


class OfficialSession:
    def __init__(self) -> None:
        self.root: Path | None = None
        self.task_dir: Path | None = None
        self.spec: Any = None
        self.workspace: Any = None
        self.adapter: Any = None
        self.sandbox: Any = None
        self.handles: Any = None
        self.tools: dict[str, Any] = {}
        self.abstention: dict[str, Any] | None = None
        self.transcript: list[dict[str, Any]] = []
        self.closed = False

    def inspect(self, root: str) -> dict[str, Any]:
        import importlib.metadata

        from apiflow_bench.harness.task_loader import discover_tasks

        checkout = Path(root).resolve()
        specs = discover_tasks([checkout / "tasks" / "v1.0"])
        tasks = []
        for spec in specs:
            evaluation = spec.metadata.get("metadata") or {}
            tasks.append(
                {
                    "task_id": spec.task_id,
                    "world": spec.task_dir.parent.name,
                    "kind": "chain" if "-chain-" in spec.task_id else "solo",
                    "axis": spec.axis,
                    "protocol": spec.protocol,
                    "difficulty": spec.difficulty,
                    "primary_bucket": evaluation.get("primary_bucket"),
                    "horizon": evaluation.get("horizon"),
                    "capability_category": evaluation.get("capability_category"),
                    "message_limit": spec.message_limit,
                    "time_limit": spec.time_limit,
                }
            )
        return {
            "type": "inspection",
            "python": platform.python_version(),
            "apiflow_bench": importlib.metadata.version("apiflow-bench"),
            "inspect_ai": importlib.metadata.version("inspect-ai"),
            "tasks": tasks,
        }

    def score(self, outcomes: list[dict[str, Any]]) -> dict[str, Any]:
        import numpy as np

        from apiflow_bench.scoring.bootstrap import cluster_bootstrap_ci

        by_world: dict[str, list[bool]] = {}
        for item in outcomes:
            by_world.setdefault(str(item["world"]), []).append(bool(item["passed"]))
        ci = cluster_bootstrap_ci(
            list(by_world.values()), rng=np.random.default_rng(0)
        )
        return {
            "type": "score",
            "passed": sum(bool(item["passed"]) for item in outcomes),
            "total": len(outcomes),
            "pass_rate": ci.point,
            "ci90_low": ci.lo,
            "ci90_high": ci.hi,
            "effective_worlds": ci.n,
            "bootstrap_iterations": ci.iters,
        }

    async def initialize(self, root: str, task_id: str) -> dict[str, Any]:
        from apiflow_bench.action_space import tools as official_tools
        from apiflow_bench.harness.neutral_solver import (
            _make_adapter,
            _seed_workspace,
            _system_prompt,
        )
        from apiflow_bench.harness.sandbox import make_sandbox
        from apiflow_bench.harness.task_loader import load_spec

        self.root = Path(root).resolve()
        matches = list((self.root / "tasks" / "v1.0").glob(f"*/{task_id}"))
        if len(matches) != 1:
            raise ValueError(f"expected one APIFlow task {task_id!r}, found {len(matches)}")
        self.task_dir = matches[0].resolve()
        self.spec = load_spec(self.task_dir)
        if self.spec.protocol != "rest":
            raise ValueError("APIFlow-Bench 1.0 adapter supports the frozen REST bank only")
        self.workspace = _seed_workspace(self.task_dir)
        self.sandbox = make_sandbox(task_dir=self.task_dir, protocol=self.spec.protocol)
        self.handles = await self.sandbox.__aenter__()
        self.adapter = _make_adapter(self.spec.protocol, self.handles.base_url)
        official_tools._ws = lambda: self.workspace
        official_tools._adapter = lambda: self.adapter
        self.tools = {
            name: getattr(official_tools, name)()
            for name in ("read", "write", "edit", "search", "execute")
        }
        self.transcript = [{"role": "user", "content": self.spec.instruction}]
        return {
            "type": "ready",
            "task_id": self.spec.task_id,
            "instruction": self.spec.instruction,
            "system_prompt": _system_prompt(self.task_dir),
            "axis": self.spec.axis,
            "protocol": self.spec.protocol,
            "difficulty": self.spec.difficulty,
            "metadata": self.spec.metadata,
            "message_limit": self.spec.message_limit,
            "time_limit": self.spec.time_limit,
            "base_url": self.handles.base_url,
        }

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        from apiflow_bench.action_space import tools as official_tools

        if self.abstention is not None:
            return {
                "type": "tool_result",
                "success": False,
                "error": "trial already terminated by abstention",
                "effect": "write",
                "task_completed": True,
            }
        self.transcript.append(
            {
                "role": "assistant",
                "tool_calls": [{"tool": name, "arguments": arguments}],
            }
        )
        try:
            if name in self.tools:
                raw = await self.tools[name](**arguments)
            elif name == "clarify":
                reason = str(arguments.get("question", ""))
                self.abstention = {"kind": "clarify", "reason": reason}
                official_tools._record(
                    "clarify", {"question": reason}, result="abstained:clarify"
                )
                raw = f"trial terminated (clarify): {reason}"
            elif name == "report_blocked":
                reason = str(arguments.get("reason", ""))
                self.abstention = {"kind": "report_blocked", "reason": reason}
                official_tools._record(
                    "report_blocked", {"reason": reason}, result="abstained:report_blocked"
                )
                raw = f"trial terminated (report_blocked): {reason}"
            else:
                raise ValueError(f"unknown APIFlow tool: {name}")
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
            self.transcript.append({"role": "tool", "content": error})
            return {
                "type": "tool_result",
                "success": False,
                "error": error,
                "effect": _effect(name, arguments),
                "task_completed": self.abstention is not None,
            }
        self.transcript.append({"role": "tool", "content": str(raw)})
        return {
            "type": "tool_result",
            "success": True,
            "result": _json_result(raw),
            "effect": _effect(name, arguments),
            "task_completed": self.abstention is not None,
        }

    async def evaluate(self, final_answer: str) -> dict[str, Any]:
        import httpx

        from apiflow_bench.scoring.validator_runner import _load_validator

        if self.task_dir is None or self.handles is None:
            raise RuntimeError("APIFlow session is not initialized")
        self.transcript.append({"role": "assistant", "content": final_answer})
        async with httpx.AsyncClient(timeout=5.0) as client:
            response = await client.get(f"{self.handles.admin_url}/state")
            response.raise_for_status()
            mock_state = response.json()
        validate, _ = _load_validator(self.task_dir)
        verdict = validate(
            state=self.workspace,
            transcript=self.transcript,
            abstention=self.abstention,
            mock_state=mock_state,
        )
        return {
            "type": "evaluation",
            "passed": bool(verdict.passed),
            "reason": str(verdict.reason or ""),
            "score": float(verdict.score),
            "predicates": verdict.predicates or [],
            "sub_predicates": verdict.sub_predicates or {},
            "workspace": self.workspace.model_dump(mode="json"),
            "mock_state": mock_state,
            "transcript": self.transcript,
        }

    async def close(self) -> dict[str, Any]:
        if self.closed:
            return {"type": "closed", "success": True, "already_closed": True}
        errors: list[str] = []
        if self.adapter is not None:
            try:
                await self.adapter.close()
            except Exception as exc:
                errors.append(f"adapter: {type(exc).__name__}: {exc}")
        if self.sandbox is not None:
            try:
                if os.name == "nt":
                    process = getattr(self.sandbox, "_proc", None)
                    if process is not None and process.returncode is None:
                        process.kill()
                        await process.communicate()
                else:
                    await self.sandbox.__aexit__(None, None, None)
            except Exception as exc:
                errors.append(f"sandbox: {type(exc).__name__}: {exc}")
        self.closed = True
        return {
            "type": "closed",
            "success": not errors,
            "errors": errors,
            "closed_at": datetime.now(UTC).isoformat(),
        }


def _effect(name: str, arguments: dict[str, Any]) -> str:
    if name in {"read", "search"}:
        return "read"
    if name == "execute":
        request = arguments.get("request")
        if isinstance(request, dict) and str(request.get("method", "")).upper() in {
            "GET",
            "HEAD",
            "OPTIONS",
        }:
            return "read"
    return "write"


def _json_result(value: Any) -> Any:
    if not isinstance(value, str):
        return value
    try:
        return json.loads(value)
    except (TypeError, ValueError):
        return value


async def main() -> None:
    session = OfficialSession()
    while True:
        line = await asyncio.to_thread(sys.stdin.readline)
        if not line:
            await session.close()
            return
        try:
            request = json.loads(line)
            kind = request.get("type")
            if kind == "inspect":
                response = session.inspect(request["root"])
            elif kind == "score":
                response = session.score(list(request.get("outcomes") or []))
            elif kind == "initialize":
                response = await session.initialize(request["root"], request["task_id"])
            elif kind == "call_tool":
                response = await session.call_tool(
                    str(request["name"]), dict(request.get("arguments") or {})
                )
            elif kind == "evaluate":
                response = await session.evaluate(str(request.get("final_answer", "")))
            elif kind == "close":
                response = await session.close()
                print(json.dumps(response, ensure_ascii=True, default=str), flush=True)
                return
            else:
                raise ValueError(f"unknown request type: {kind}")
        except BaseException as exc:
            response = {"type": "error", "error": f"{type(exc).__name__}: {exc}"}
        print(json.dumps(response, ensure_ascii=True, default=str), flush=True)


if __name__ == "__main__":
    asyncio.run(main())
