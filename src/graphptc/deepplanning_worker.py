from __future__ import annotations

import contextlib
import hashlib
import io
import json
import os
import sys
import time
from pathlib import Path
from typing import Any


def _send(payload: dict[str, Any]) -> None:
    sys.__stdout__.write(json.dumps(payload, ensure_ascii=True, default=repr) + "\n")
    sys.__stdout__.flush()


def _sha256(value: Any) -> str:
    raw = json.dumps(value, ensure_ascii=False, sort_keys=True, default=repr).encode()
    return hashlib.sha256(raw).hexdigest()


def _file_sha256(path: Path) -> str | None:
    return hashlib.sha256(path.read_bytes()).hexdigest() if path.exists() else None


def _json_value(value: str) -> Any:
    try:
        return json.loads(value)
    except (json.JSONDecodeError, TypeError):
        return value


class OfficialTools:
    def __init__(self, request: dict[str, Any]) -> None:
        self.domain = str(request["domain"])
        self.sample_id = str(request["sample_id"])
        self.database_dir = Path(request["database_dir"]).resolve()
        self.calls: list[dict[str, Any]] = []
        deepplanning = Path(request["official_root"]).resolve() / "benchmark" / "deepplanning"
        domain_root = deepplanning / ("travelplanning" if self.domain == "travel" else "shoppingplanning")
        sys.path.insert(0, str(domain_root))
        if self.domain == "travel":
            from agent.tools_fn_agent import ToolsFnAgent
            from agent.prompts import get_system_prompt

            language = str(request["language"])
            schema = domain_root / "tools" / f"tool_schema_{language}.json"
            self.agent = ToolsFnAgent(
                model="qwen-plus",
                sample_id=self.sample_id,
                database_base_path=str(self.database_dir),
                tool_schema_path=str(schema),
                language=language,
            )
            self.cart_path = None
            self.official_prompt = get_system_prompt(language)
        elif self.domain == "shopping":
            from agent.shopping_agent import ShoppingFnAgent
            from agent.prompts import prompt_lib

            schema = domain_root / "tools" / "shopping_tool_schema.json"
            self.agent = ShoppingFnAgent(
                model="qwen-plus",
                sample_id=self.sample_id,
                database_base_path=str(self.database_dir),
                tool_schema_path=str(schema),
            )
            self.cart_path = self.database_dir / f"case_{self.sample_id}" / "cart.json"
            self.official_prompt = getattr(prompt_lib, f"SYSTEM_PROMPT_level{int(request['level'])}")
        else:
            raise ValueError(f"unsupported DeepPlanning domain: {self.domain}")
        if not self.agent.tool_instances:
            raise RuntimeError("official DeepPlanning tools failed to load")
        self.schemas = list(self.agent.openai_tools)

    def namespace(self) -> dict[str, Any]:
        return {
            name: self._wrapper(name)
            for name in sorted(self.agent.tool_instances)
        }

    def _wrapper(self, name: str):
        def invoke(**arguments: Any) -> Any:
            started = time.perf_counter()
            cart_before = _file_sha256(self.cart_path) if self.cart_path else None
            raw = self.agent._exec_tool(name, json.dumps(arguments, ensure_ascii=False))
            value = _json_value(raw)
            cart_after = _file_sha256(self.cart_path) if self.cart_path else None
            success = not (isinstance(value, dict) and "error" in value)
            record = {
                "index": len(self.calls) + 1,
                "tool": name,
                "arguments": arguments,
                "success": success,
                "result_sha256": _sha256(value),
                "result_chars": len(json.dumps(value, ensure_ascii=False, default=repr)),
                "duration_ms": (time.perf_counter() - started) * 1000,
                "effect": "write" if cart_before != cart_after else "read",
            }
            if cart_before != cart_after:
                record["state_effect"] = {
                    "artifact": "cart.json",
                    "before_sha256": cart_before,
                    "after_sha256": cart_after,
                }
            self.calls.append(record)
            if not success:
                raise RuntimeError(str(value["error"]))
            return value

        invoke.__name__ = name
        invoke.__qualname__ = name
        schema = next(
            (item["function"] for item in self.schemas if item.get("function", {}).get("name") == name),
            {},
        )
        invoke.__doc__ = str(schema.get("description", ""))
        return invoke


def main() -> int:
    os.environ.setdefault("PYTHONUTF8", "1")
    request = json.loads(sys.stdin.readline())
    if request.get("type") != "initialize":
        _send({"type": "error", "error": "expected initialize"})
        return 2
    with contextlib.redirect_stdout(sys.stderr), contextlib.redirect_stderr(sys.stderr):
        tools = OfficialTools(request)
    globals_: dict[str, Any] = {"__builtins__": __builtins__, **tools.namespace()}
    _send(
        {
            "type": "ready",
            "domain": tools.domain,
            "sample_id": tools.sample_id,
            "tools": tools.schemas,
            "tool_names": sorted(tools.agent.tool_instances),
            "official_prompt": tools.official_prompt,
        }
    )
    for line in sys.stdin:
        request = json.loads(line)
        kind = request.get("type")
        if kind == "execute":
            calls_before = len(tools.calls)
            stdout = io.StringIO()
            stderr = io.StringIO()
            rc = 0
            try:
                with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
                    exec(compile(str(request["code"]), "<deepplanning-ptc>", "exec"), globals_, globals_)
            except BaseException as exc:
                rc = 1
                stderr.write(f"{type(exc).__name__}: {exc}")
            recent = tools.calls[calls_before:]
            _send(
                {
                    "type": "execution",
                    "stdout": stdout.getvalue(),
                    "stderr": stderr.getvalue(),
                    "rc": rc,
                    "external_actions": recent,
                    "state_effects": [c["state_effect"] for c in recent if "state_effect" in c],
                    "artifacts": [
                        {"kind": "tool_result", "tool": c["tool"], "sha256": c["result_sha256"]}
                        for c in recent
                    ],
                }
            )
        elif kind == "telemetry":
            _send({"type": "telemetry", "tool_calls": tools.calls})
        elif kind == "close":
            _send({"type": "closed"})
            return 0
        else:
            _send({"type": "error", "error": f"unknown request: {kind}"})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
