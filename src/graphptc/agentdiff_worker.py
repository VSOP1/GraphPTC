from __future__ import annotations

import ast
import json
import os
import sys
from importlib.metadata import version
from typing import Any, Mapping


def _dump(value: Any) -> Any:
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json")
    if isinstance(value, Mapping):
        return {str(key): _dump(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_dump(item) for item in value]
    return value


def _emit(payload: dict[str, Any]) -> None:
    print(json.dumps(payload, ensure_ascii=True, default=repr), flush=True)


def _literal(node: ast.AST | None) -> str:
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    return "<dynamic>"


def _http_actions(code: str, *, block_success: bool) -> list[dict[str, Any]]:
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return []
    actions: list[dict[str, Any]] = []
    methods = {"get", "head", "options", "post", "put", "patch", "delete", "request"}
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
            continue
        name = node.func.attr.lower()
        if name not in methods:
            continue
        method = name.upper()
        url_node: ast.AST | None = node.args[0] if node.args else None
        if name == "request":
            method = _literal(node.args[0] if node.args else None).upper()
            url_node = node.args[1] if len(node.args) > 1 else None
            for keyword in node.keywords:
                if keyword.arg == "method":
                    method = _literal(keyword.value).upper()
                elif keyword.arg == "url":
                    url_node = keyword.value
        else:
            for keyword in node.keywords:
                if keyword.arg == "url":
                    url_node = keyword.value
        effect = "read" if method.lower() in {"get", "head", "options"} else "write"
        success: bool | None = False if not block_success else None
        actions.append(
            {
                "name": f"{method} {_literal(url_node)}",
                "arguments": {"method": method, "url": _literal(url_node)},
                "effect": effect,
                "success": success,
                "outcome_unknown": success is None,
                "effect_basis": "program_execution_only",
            }
        )
    return actions


class Session:
    def __init__(self, request: Mapping[str, Any]) -> None:
        from agent_diff import AgentDiff, PythonExecutorProxy

        task = dict(request["task"])
        info = task["info"]
        if isinstance(info, str):
            info = json.loads(info)
        answer = task["answer"]
        if isinstance(answer, str):
            answer = json.loads(answer)
        self.task = task
        self.expected = answer
        self.client = AgentDiff()
        self.env = self.client.init_env(
            templateService=info["service"],
            templateName=info["seed_template"],
            impersonateUserId=info["impersonate_user_id"],
            ttlSeconds=max(900, int(float(request.get("timeout_seconds", 480))) + 300),
        )
        self.run = self.client.start_run(envId=self.env.environmentId)
        self.executor = PythonExecutorProxy(
            self.env.environmentId,
            base_url=self.client.base_url,
            api_key=self.client.api_key,
        )
        self.deleted = False

    def execute(self, code: str) -> dict[str, Any]:
        result = dict(self.executor.execute(code))
        success = result.get("status") == "success" and int(result.get("exit_code", 1)) == 0
        return {
            "type": "execution",
            **result,
            "external_actions": _http_actions(code, block_success=success),
            "state_effects": [],
        }

    def evaluate(self) -> dict[str, Any]:
        evaluated = self.client.evaluate_run(runId=self.run.runId, expectedOutput=self.expected)
        result = self.client.get_results_for_run(runId=self.run.runId)
        data = _dump(result)
        total = len(self.expected.get("assertions", []))
        normalized = _normalize_evaluation(data, _dump(evaluated), total)
        return {
            "type": "evaluation",
            "evaluation": normalized,
            "official_result": data,
            "official_end_run": _dump(evaluated),
            "official_diff": data.get("diff", {}) if isinstance(data, Mapping) else {},
        }

    def close(self) -> bool:
        if not self.deleted:
            self.client.delete_env(envId=self.env.environmentId)
            self.deleted = True
        workspace = getattr(self.executor, "workspace", None)
        destroy = getattr(workspace, "destroy", None)
        if callable(destroy):
            destroy()
        return self.deleted


def _normalize_evaluation(result: Any, evaluated: Any, total: int) -> dict[str, Any]:
    source = result if isinstance(result, Mapping) else {}
    fallback = evaluated if isinstance(evaluated, Mapping) else {}
    passed = bool(source.get("passed", fallback.get("passed", False)))
    raw_score = source.get("score", fallback.get("score", 0))
    score_details = dict(raw_score) if isinstance(raw_score, Mapping) else {}
    official_total = int(score_details.get("total", total) or total)
    satisfied = int(score_details.get("passed", 0) or 0)
    if not score_details:
        score = _numeric_score(raw_score)
        satisfied = total if passed else round(score * total) if 0 <= score <= 1 else int(score)
        official_total = total
    else:
        score = satisfied / official_total if official_total else float(passed)
    failures = source.get("failures", [])
    if not isinstance(failures, list):
        failures = [str(failures)]
    return {
        "passed": passed,
        "score": score,
        "score_details": score_details,
        "satisfied_assertions": max(0, min(official_total, satisfied)),
        "total_assertions": official_total,
        "failures": failures,
        "status": str(source.get("status", fallback.get("status", ""))),
    }


def _numeric_score(value: Any) -> float:
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, Mapping):
        for key in ("score", "value", "similarity"):
            if key in value:
                return _numeric_score(value[key])
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _inspect(request: Mapping[str, Any]) -> dict[str, Any]:
    from agent_diff import AgentDiff

    client = AgentDiff()
    return {
        "type": "inspection",
        "sdk_version": version("agent-diff"),
        "official_commit": request["official_commit"],
        "base_url": client.base_url,
        "python_version": sys.version.split()[0],
        "api_key_configured": bool(os.getenv("AGENT_DIFF_API_KEY")),
    }


def main() -> int:
    session: Session | None = None
    try:
        for raw in sys.stdin:
            request = json.loads(raw)
            kind = request.get("type")
            if kind == "inspect":
                _emit(_inspect(request))
            elif kind == "initialize":
                session = Session(request)
                _emit(
                    {
                        "type": "ready",
                        "environment_id": session.env.environmentId,
                        "run_id": session.run.runId,
                        "sdk_version": version("agent-diff"),
                        "official_commit": request["official_commit"],
                        "python_state_persistent": False,
                    }
                )
            elif kind == "execute" and session is not None:
                _emit(session.execute(str(request.get("code", ""))))
            elif kind == "evaluate" and session is not None:
                _emit(session.evaluate())
            elif kind == "close" and session is not None:
                _emit({"type": "closed", "environment_deleted": session.close()})
                return 0
            else:
                raise RuntimeError(f"invalid Agent-Diff worker request: {kind!r}")
    except Exception as exc:
        _emit({"type": "error", "error": f"{type(exc).__name__}: {exc}"})
        return 1
    finally:
        if session is not None:
            try:
                session.close()
            except Exception:
                pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
