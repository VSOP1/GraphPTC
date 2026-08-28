from __future__ import annotations

import dataclasses
import hashlib
import json
import os
import re
import subprocess
import sys
import time
from collections.abc import Mapping, Sequence
from importlib.metadata import version
from pathlib import Path
from typing import Any

SRC_ROOT = Path(__file__).resolve().parents[1]
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from graphptc.tau3_worker import _agent_class, _dump, _emit, _stamp_trial


def _sha256(value: Any) -> str:
    payload = json.dumps(
        value, sort_keys=True, ensure_ascii=False, default=str, separators=(",", ":")
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _git_manifest(root: Path, pattern: str, pathspec: str) -> dict[str, Any]:
    expression = re.compile(pattern)
    entries: list[dict[str, Any]] = []
    if (root / ".git").exists():
        completed = subprocess.run(
            ["git", "ls-tree", "-r", "-l", "HEAD", "--", pathspec],
            cwd=root,
            text=True,
            encoding="utf-8",
            capture_output=True,
            check=True,
        )
        for line in completed.stdout.splitlines():
            metadata, path = line.split("\t", 1)
            _, kind, sha, size = metadata.split()
            if kind == "blob" and expression.fullmatch(path):
                entries.append({"path": path, "sha": sha, "size": int(size)})
    else:
        for candidate in root.rglob("*"):
            if not candidate.is_file():
                continue
            path = candidate.relative_to(root).as_posix()
            if not expression.fullmatch(path):
                continue
            content = candidate.read_bytes()
            git_blob = b"blob " + str(len(content)).encode("ascii") + b"\0" + content
            entries.append(
                {
                    "path": path,
                    "sha": hashlib.sha1(git_blob).hexdigest(),
                    "size": len(content),
                }
            )
    entries.sort(key=lambda item: item["path"])
    manifest_text = "\n".join(
        f"{item['path']}\t{item['sha']}\t{item['size']}" for item in entries
    )
    return {
        "count": len(entries),
        "git_manifest_sha256": hashlib.sha256(
            manifest_text.encode("utf-8")
        ).hexdigest(),
    }


def _retrieval_probe(
    *,
    retrieval_config: str,
    retrieval_kwargs: Mapping[str, Any],
    queries: Sequence[str],
) -> dict[str, Any]:
    from tau2.domains.banking_knowledge.environment import (
        get_environment,
        get_knowledge_base,
    )
    from tau2.domains.banking_knowledge.retrieval import resolve_variant

    variant = resolve_variant(retrieval_config, **dict(retrieval_kwargs))
    knowledge_base = get_knowledge_base()
    document_ids = list(knowledge_base.documents)

    def build_arm() -> dict[str, Any]:
        environment = get_environment(
            retrieval_variant=retrieval_config,
            retrieval_kwargs=dict(retrieval_kwargs),
        )
        visible_tools = environment.get_tools()
        visible_names = [tool.name for tool in visible_tools]
        hidden_names = sorted(environment.tools.get_discoverable_tools())
        outputs = [
            re.sub(
                r"\n\n\[Timing: retrieval=.*?\]\s*$",
                "",
                str(environment.use_tool("KB_search", query=query)),
            )
            for query in queries
        ]
        return {
            "visible_tool_names": visible_names,
            "visible_schema_sha256": _sha256(
                [tool.openai_schema for tool in visible_tools]
            ),
            "hidden_tool_count": len(hidden_names),
            "hidden_tool_names_sha256": _sha256(hidden_names),
            "hidden_names_exposed": bool(set(hidden_names) & set(visible_names)),
            "policy_sha256": hashlib.sha256(
                environment.get_policy().encode("utf-8")
            ).hexdigest(),
            "query_output_sha256": [
                hashlib.sha256(value.encode("utf-8")).hexdigest() for value in outputs
            ],
        }

    left = build_arm()
    right = build_arm()
    stable_variant = {
        "name": variant.name,
        "prompt_template": Path(variant.prompt_template).name,
        "build_prompt": (
            f"{variant.build_prompt.__module__}.{variant.build_prompt.__qualname__}"
        ),
        "kb_search": dataclasses.asdict(variant.kb_search)
        if variant.kb_search is not None
        else None,
        "kb_search_bm25": dataclasses.asdict(variant.kb_search_bm25)
        if variant.kb_search_bm25 is not None
        else None,
        "kb_search_dense": dataclasses.asdict(variant.kb_search_dense)
        if variant.kb_search_dense is not None
        else None,
        "grep": dataclasses.asdict(variant.grep) if variant.grep is not None else None,
        "shell": dataclasses.asdict(variant.shell)
        if variant.shell is not None
        else None,
        "supports_top_k": variant.supports_top_k,
    }
    return {
        "config": retrieval_config,
        "config_kwargs": dict(retrieval_kwargs),
        "variant": stable_variant,
        "document_count": len(document_ids),
        "document_order_sha256": _sha256(document_ids),
        "queries": list(queries),
        "graphptc_probe": left,
        "fewshot_ptc_probe": right,
        "arms_identical": left == right,
        "offline_bm25_only": (
            retrieval_config == "bm25"
            and variant.kb_search is not None
            and variant.kb_search.type == "bm25"
            and not variant.kb_search.reranker
            and variant.grep is None
            and variant.shell is None
            and variant.kb_search_bm25 is None
            and variant.kb_search_dense is None
        ),
    }


def _inspect(request: Mapping[str, Any]) -> dict[str, Any]:
    from tau2.config import (
        DEFAULT_LLM_TEMPERATURE_AGENT,
        DEFAULT_LLM_TEMPERATURE_USER,
        DEFAULT_MAX_CONCURRENCY,
        DEFAULT_MAX_ERRORS,
        DEFAULT_MAX_STEPS,
        DEFAULT_SEED,
    )
    from tau2.runner import get_tasks

    root = Path(str(request["root"])).resolve()
    if not (root / ".git").exists():
        raise RuntimeError("tau2 root must be an exact git checkout")
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=root,
        text=True,
        encoding="utf-8",
        errors="replace",
        capture_output=True,
        check=True,
    ).stdout.strip()
    repository = dict(request["source_repository"])
    source_provenance = {
        "transport": "git",
        "url": subprocess.run(
            ["git", "remote", "get-url", "origin"],
            cwd=root,
            text=True,
            encoding="utf-8",
            errors="replace",
            capture_output=True,
            check=True,
        ).stdout.strip(),
        "tag": subprocess.run(
            ["git", "describe", "--tags", "--exact-match"],
            cwd=root,
            text=True,
            encoding="utf-8",
            errors="replace",
            capture_output=True,
            check=True,
        ).stdout.strip(),
        "commit": commit,
    }
    if source_provenance["url"] != repository["url"]:
        raise RuntimeError("tau2 git remote differs from the frozen protocol")
    required_runtime_files = {}
    for relative_path, expected_sha in dict(request["required_runtime_files"]).items():
        content = (root / relative_path).read_bytes()
        blob = b"blob " + str(len(content)).encode("ascii") + b"\0" + content
        actual_sha = hashlib.sha1(blob).hexdigest()
        if actual_sha != expected_sha:
            raise RuntimeError(f"official runtime file changed: {relative_path}")
        required_runtime_files[relative_path] = actual_sha
    check = subprocess.run(
        [sys.executable, "-X", "utf8", "-m", "tau2.cli", "check-data"],
        cwd=root,
        text=True,
        encoding="utf-8",
        errors="replace",
        capture_output=True,
        check=False,
    )
    tasks = get_tasks(
        task_set_name="banking_knowledge",
        task_split_name=str(request["task_split_name"]),
        task_ids=None,
        num_tasks=None,
    )
    task_ids = [str(task.id) for task in tasks]
    retrieval = _retrieval_probe(
        retrieval_config=str(request["retrieval_config"]),
        retrieval_kwargs=dict(request["retrieval_config_kwargs"]),
        queries=[str(value) for value in request["retrieval_probe_queries"]],
    )
    return {
        "type": "inspection",
        "official_commit": commit,
        "source_provenance": source_provenance,
        "required_runtime_files": required_runtime_files,
        "package_version": version("tau2"),
        "python_version": sys.version.split()[0],
        "data_verified": check.returncode == 0,
        "official_defaults": {
            "max_steps": DEFAULT_MAX_STEPS,
            "max_errors": DEFAULT_MAX_ERRORS,
            "seed": DEFAULT_SEED,
            "max_concurrency": DEFAULT_MAX_CONCURRENCY,
            "agent_temperature": DEFAULT_LLM_TEMPERATURE_AGENT,
            "user_temperature": DEFAULT_LLM_TEMPERATURE_USER,
        },
        "domain": "banking_knowledge",
        "task_split_name": str(request["task_split_name"]),
        "task_ids": task_ids,
        "task_count": len(task_ids),
        "task_ids_sha256": _sha256(task_ids),
        "task_files": _git_manifest(
            root,
            r"data/tau2/domains/banking_knowledge/tasks/task_[0-9]+\.json",
            "data/tau2/domains/banking_knowledge/tasks",
        ),
        "knowledge_documents": _git_manifest(
            root,
            r"data/tau2/domains/banking_knowledge/documents/.+\.json",
            "data/tau2/domains/banking_knowledge/documents",
        ),
        "knowledge_prompts": _git_manifest(
            root,
            r"data/tau2/domains/banking_knowledge/prompts/.+",
            "data/tau2/domains/banking_knowledge/prompts",
        ),
        "retrieval": retrieval,
    }


def _text_config(request: Mapping[str, Any], *, aggregate: bool = False):
    from tau2.data_model.simulation import TextRunConfig

    return TextRunConfig(
        domain="banking_knowledge",
        task_set_name="banking_knowledge",
        task_split_name=str(request["task_split_name"]),
        task_ids=[str(value) for value in request["task_ids"]],
        agent=str(request["agent_name"]),
        llm_agent=str(
            request["agent_model"] if aggregate else request["agent_model"]["model"]
        ),
        llm_args_agent={},
        user="user_simulator",
        llm_user=str(request["user_model"]),
        llm_args_user={"temperature": 0.0, "api_base": str(request["user_base_url"])},
        retrieval_config=str(request["retrieval_config"]),
        retrieval_config_kwargs=dict(request["retrieval_config_kwargs"]),
        num_trials=1,
        max_steps=int(request["max_steps"]),
        max_errors=int(request["max_errors"]),
        max_concurrency=int(request["max_concurrency"] if aggregate else 1),
        seed=int(request["seed"]),
        timeout=None if aggregate else float(request["timeout"]),
        log_level="ERROR",
        verbose_logs=False,
        max_retries=0,
        hallucination_retries=0,
        enforce_communication_protocol=bool(request["enforce_communication_protocol"]),
    )


def _runtime_metrics(agent: Any, duration_seconds: float) -> dict[str, Any]:
    artifact = agent.agent_artifact()
    blocks = artifact.get("blocks") or []
    calls = [
        call
        for block in blocks
        for call in ((block.get("runtime_trace") or {}).get("external_actions") or [])
    ]
    usage = (artifact.get("telemetry") or {}).get("usage") or {}
    return {
        "model_turns": int((artifact.get("telemetry") or {}).get("model_requests", 0)),
        "ptc_blocks": len(blocks),
        "tool_calls": len(calls),
        "retrieval_calls": sum(call.get("name") == "KB_search" for call in calls),
        "unlock_calls": sum(
            call.get("name") == "unlock_discoverable_agent_tool" for call in calls
        ),
        "dynamic_tool_calls": sum(
            call.get("name") == "call_discoverable_agent_tool" for call in calls
        ),
        "state_change_calls": sum(call.get("state_changed") is True for call in calls),
        "input_tokens": int(usage.get("input_tokens", 0)),
        "output_tokens": int(usage.get("output_tokens", 0)),
        "cached_input_tokens": int(usage.get("cached_input_tokens", 0)),
        "duration_seconds": duration_seconds,
    }


def _save_agent_artifacts(agent: Any, request: Mapping[str, Any]) -> None:
    agent_path = Path(str(request["agent_path"]))
    agent_path.parent.mkdir(parents=True, exist_ok=True)
    agent_path.write_text(
        json.dumps(agent.agent_artifact(), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    graph = agent.graph_artifact()
    if graph is not None:
        graph_path = Path(str(request["graph_path"]))
        graph_path.parent.mkdir(parents=True, exist_ok=True)
        graph_path.write_text(
            json.dumps(graph, ensure_ascii=False, indent=2), encoding="utf-8"
        )


def _run(request: Mapping[str, Any]) -> dict[str, Any]:
    from tau2.registry import registry
    from tau2.runner import get_tasks, run_single_task

    api_key = os.environ[str(request["agent_model"].get("api_key_env", "MIMO_API_KEY"))]
    os.environ["OPENAI_API_KEY"] = api_key
    os.environ["OPENAI_API_BASE"] = str(request["user_base_url"])
    request = {**request, "task_ids": [str(request["task_id"])], "max_concurrency": 1}
    tasks = get_tasks(
        task_set_name="banking_knowledge",
        task_split_name=str(request["task_split_name"]),
        task_ids=request["task_ids"],
        num_tasks=None,
    )
    if len(tasks) != 1:
        raise ValueError(f"expected one official task, found {len(tasks)}")
    holder: dict[str, Any] = {}

    def factory(tools, domain_policy, **kwargs):
        visible_names = {tool.name for tool in tools}
        owners = {
            getattr(getattr(tool, "_func", None), "__self__", None) for tool in tools
        }
        hidden_names = {
            name
            for owner in owners
            if owner is not None and hasattr(owner, "get_discoverable_tools")
            for name in owner.get_discoverable_tools()
        }
        if visible_names & hidden_names:
            raise ValueError(
                "discoverable tool names leaked into the initial agent tool surface"
            )
        cls = _agent_class()
        agent = cls(tools=tools, domain_policy=domain_policy, request=request)
        holder["agent"] = agent
        holder["tool_surface"] = {
            "visible_tool_names": sorted(visible_names),
            "visible_tool_schema_sha256": _sha256(
                [tool.openai_schema for tool in tools]
            ),
            "hidden_tool_count": len(hidden_names),
            "hidden_tool_names_sha256": _sha256(sorted(hidden_names)),
            "hidden_names_exposed": False,
        }
        return agent

    registry.register_agent_factory(factory, str(request["agent_name"]))
    config = _text_config(request)
    simulation = None
    started = time.perf_counter()
    try:
        simulation = _stamp_trial(
            run_single_task(config, tasks[0], seed=int(request["seed"])),
            int(request["trial"]),
        )
    finally:
        agent = holder.get("agent")
        if agent is not None:
            agent.close(
                answered=bool(simulation and simulation.reward_info is not None)
            )
            _save_agent_artifacts(agent, request)
    duration_seconds = time.perf_counter() - started
    if simulation is None or agent is None:
        raise RuntimeError("official simulation or agent was not created")

    official_path = Path(str(request["official_path"]))
    official_path.parent.mkdir(parents=True, exist_ok=True)
    official_path.write_text(simulation.model_dump_json(indent=2), encoding="utf-8")
    graph = agent.graph_artifact()
    telemetry = agent.telemetry()
    reward = (
        simulation.reward_info.reward if simulation.reward_info is not None else 0.0
    )
    termination = str(simulation.termination_reason)
    return {
        "type": "result",
        "status": "finished",
        "simulation_id": simulation.id,
        "reward": float(reward or 0.0),
        "reward_info": _dump(simulation.reward_info),
        "termination_reason": termination,
        "incomplete": "user_stop" not in termination.lower()
        and float(reward or 0.0) < 1.0,
        "evaluator_failed": simulation.reward_info is None,
        "execution_failures": int(telemetry.get("execution_failures", 0)),
        "telemetry": telemetry,
        "runtime_metrics": _runtime_metrics(agent, duration_seconds),
        "tool_surface": holder["tool_surface"],
        "official_path": str(official_path),
        "agent_path": str(request["agent_path"]),
        "graph_path": str(request["graph_path"]) if graph is not None else None,
    }


def _aggregate(request: Mapping[str, Any]) -> dict[str, Any]:
    from tau2.data_model.simulation import Results, SimulationRun
    from tau2.runner import get_tasks
    from tau2.runner.helpers import get_info

    task_ids = list(dict.fromkeys(str(value) for value in request["task_ids"]))
    request = {**request, "task_ids": task_ids}
    tasks = get_tasks(
        task_set_name="banking_knowledge",
        task_split_name=str(request["task_split_name"]),
        task_ids=task_ids,
        num_tasks=None,
    )
    simulations = [
        SimulationRun.model_validate_json(Path(path).read_text(encoding="utf-8"))
        for path in request["official_paths"]
    ]
    config = _text_config(request, aggregate=True)
    results = Results(info=get_info(config), tasks=tasks, simulations=simulations)
    output_path = Path(str(request["output_path"]))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    results.save(output_path, format="json")
    return {
        "type": "aggregate",
        "domain": "banking_knowledge",
        "output_path": str(output_path),
        "tasks": len(tasks),
        "simulations": len(simulations),
    }


def main() -> int:
    try:
        request = json.loads(sys.stdin.readline().lstrip("\ufeff"))
        if request.get("type") == "inspect":
            response = _inspect(request)
        elif request.get("type") == "run":
            response = _run(request)
        elif request.get("type") == "aggregate":
            response = _aggregate(request)
        else:
            raise ValueError(f"unsupported request type: {request.get('type')!r}")
        _emit(response)
        return 0
    except Exception as exc:  # noqa: BLE001 - worker boundary returns structured failure
        _emit({"type": "error", "error": f"{type(exc).__name__}: {exc}"})
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
