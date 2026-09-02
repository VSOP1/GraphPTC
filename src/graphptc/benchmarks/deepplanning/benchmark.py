from __future__ import annotations

import copy
import dataclasses
import hashlib
import json
import math
import re
import shutil
import subprocess
import threading
import time
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from openai import OpenAI

from ...config import ExperimentConfig
from .runtime import DeepPlanningProgramRuntime
from ...graph.adaptation import GoalGraphAdaptation
from ...graph.hooks import GraphAgentHooks, extend_ptc_spec_with_graph_control
from ...model import OpenAIChatModel
from ...agents.original_ptc import OriginalPTCAgent, PTC_TOOL_SPEC


DEEPPLANNING_OFFICIAL_COMMIT = "31a4d36d123688581a9e9744427272b33ce940e0"
DEEPPLANNING_DATA_REVISION = "213876cce679f993a476d01042e13d111c0e3648"
DEEPPLANNING_LICENSE = "Apache-2.0"

PTC_GUIDANCE = """Use programmatic_tool_call for all Python and official-tool work. The official
DeepPlanning tools listed below are Python functions in the persistent task sandbox. Call them with
keyword arguments. Combine mechanically predictable searches, filtering, joins, and calculations
in one coherent block, and print only compact decision-relevant results. Tool and Python state
persist between blocks. Official tools may return mappings, lists, or text; inspect the returned
shape before indexing it. Do not fabricate tool results. Once the task is complete, return the final
answer without another programmatic_tool_call. Finish Travel with the exact official
<plan>...</plan> schema; for Shopping, mutate the official cart to the requested optimal contents
and then give a concise final answer."""

GRAPH_GUIDANCE = """Graph control uses the benchmark-neutral GraphPTC contract. Declare CONTINUE,
PATCH, or REPLAN with target task and an observable expected change. GRAPH_ASSESSMENT and each
GRAPH_DELTA summarize only goals, constraints, tool calls, artifacts, state effects, failures, and
the next action."""

USER_PROMPT = "{question}"


def _demo(graph: bool) -> tuple[dict[str, Any], ...]:
    arguments: dict[str, Any] = {
        "code": "items = [3, 1, 2]\nprint({'sorted_items': sorted(items)})",
    }
    if graph:
        arguments.update(action="CONTINUE", target="task", expected_change="derive a sorted intermediate result")
    output = "{'sorted_items': [1, 2, 3]}"
    if graph:
        output += '\n\nGRAPH_DELTA {"declared_action":{"action":"CONTINUE","target":"task"},"action_verification":{"realized":true}}'
    return (
        {"role": "user", "content": "PTC organization demonstration only: sort these provided values."},
        {"role": "assistant", "content": "I will compute the deterministic intermediate result.", "tool_calls": [{"id": "deepplanning_demo_1", "type": "function", "function": {"name": "programmatic_tool_call", "arguments": json.dumps(arguments)}}]},
        {"role": "tool", "tool_call_id": "deepplanning_demo_1", "content": output},
        {"role": "assistant", "content": "The sorted values are 1, 2, and 3."},
    )


@dataclass(frozen=True)
class DeepPlanningTask:
    domain: str
    sample_id: str
    query: str
    language: str | None = None
    level: int | None = None

    @property
    def key(self) -> str:
        qualifier = self.language if self.domain == "travel" else f"level{self.level}"
        return f"{self.domain}-{qualifier}-{self.sample_id}"


@dataclass(frozen=True)
class DeepPlanningRunSummary:
    selected: int
    processed: int
    runner_failures: int
    execution_failure_tasks: int
    execution_failure_blocks: int
    incomplete_tasks: int
    model_calls: int
    tool_calls: int
    input_tokens: int
    output_tokens: int
    duration_seconds: float
    run_signature: str
    leaderboard_equivalent_repetitions: bool

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


class _EmptySearchTools:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []


def inspect_deepplanning(config: ExperimentConfig) -> dict[str, Any]:
    app = config.deepplanning
    root = Path(app.root)
    _validate_protocol(config)
    tasks = load_deepplanning_tasks(config)
    freeze = _pip_freeze(config)
    tool_counts: dict[str, int] = {}
    for task in (tasks[0], tasks[120], tasks[240], tasks[290], tasks[340]):
        runtime = _runtime(config, task, _database_dir(config, task, None))
        try:
            tool_counts[task.key.rsplit("-", 1)[0]] = len(runtime.metadata["tool_names"])
        finally:
            runtime.close()
    return {
        "official_commit": app.official_commit,
        "data_revision": app.data_revision,
        "code_license": DEEPPLANNING_LICENSE,
        "data_license": DEEPPLANNING_LICENSE,
        "task_counts": _counts(task.key.rsplit("-", 1)[0] for task in tasks),
        "total_tasks": len(tasks),
        "tool_counts": tool_counts,
        "python": _run_text((app.python_command, "--version")),
        "dependency_freeze_sha256": _sha256(freeze),
        "dependency_freeze": freeze,
        "official_root": str(root.resolve()),
        "max_model_calls_documented": app.max_model_calls,
        "shopping_implementation_max_calls": app.max_model_calls * 2,
        "travel_conversion_default": "qwen-plus via DashScope",
        "travel_conversion_selected": config.model.model,
        "official_agent_total_attempts": 30,
        "official_agent_retry_backoff_seconds": 1.5,
        "selected_agent_total_attempts": config.model.max_retries + 1,
        "selected_sdk_hidden_retries": 0,
        "official_conversion_retries": 30,
        "selected_conversion_retries": 30,
        "official_unified_workers_per_domain": 50,
        "official_travel_script_workers": 40,
        "official_cli_default_workers": {"travel": 10, "shopping": 5},
        "selected_workers_per_arm": app.workers,
        "selected_total_concurrency": app.workers * 2,
        "leaderboard_runs": app.run_count,
    }


def load_deepplanning_tasks(config: ExperimentConfig) -> list[DeepPlanningTask]:
    base = Path(config.deepplanning.root) / "benchmark" / "deepplanning"
    tasks: list[DeepPlanningTask] = []
    for language in ("zh", "en"):
        path = base / "travelplanning" / "data" / f"travelplanning_query_{language}.json"
        for row in json.loads(path.read_text(encoding="utf-8")):
            tasks.append(DeepPlanningTask("travel", str(row["id"]), str(row["query"]), language=language))
    for level in (1, 2, 3):
        path = base / "shoppingplanning" / "data" / f"level_{level}_query_meta.json"
        for row in json.loads(path.read_text(encoding="utf-8")):
            tasks.append(DeepPlanningTask("shopping", str(row["id"]), str(row["query"]), level=level))
    expected = 2 * config.deepplanning.expected_travel_tasks_per_language + sum(config.deepplanning.expected_shopping_tasks)
    if len(tasks) != expected:
        raise ValueError(f"expected {expected} DeepPlanning tasks, found {len(tasks)}")
    return tasks


def probe_deepplanning_api(
    config: ExperimentConfig,
    *,
    concurrencies: Sequence[int] = (10, 20, 40),
    waves: int = 2,
    output: Path | None = None,
) -> dict[str, Any]:
    """Measure raw provider stability without task prompts or transport retries."""
    _validate_protocol(config)
    levels = tuple(int(value) for value in concurrencies)
    if not levels or any(value <= 0 for value in levels):
        raise ValueError("probe concurrencies must be positive")
    if tuple(sorted(set(levels))) != levels:
        raise ValueError("probe concurrencies must be unique and increasing")
    if waves <= 0:
        raise ValueError("probe waves must be positive")
    client = OpenAI(
        api_key=config.require_api_key(config.model.api_key_env),
        base_url=config.model.base_url,
        max_retries=0,
        timeout=config.model.timeout_seconds,
    )
    request: dict[str, Any] = {
        "model": config.model.model,
        "messages": [{"role": "user", "content": "Reply with exactly OK."}],
        "max_completion_tokens": 8,
        "temperature": 0.0,
    }
    if config.model.thinking:
        request["extra_body"] = {"thinking": {"type": config.model.thinking}}

    def one(level: int, index: int) -> dict[str, Any]:
        started = time.perf_counter()
        try:
            response = client.chat.completions.create(**request)
            message = response.choices[0].message
            content = (message.content or "").strip()
            if not content and not (getattr(message, "tool_calls", None) or []):
                raise ValueError("empty response")
            usage = getattr(response, "usage", None)
            return {
                "level": level,
                "index": index,
                "success": True,
                "duration_ms": (time.perf_counter() - started) * 1_000,
                "input_tokens": getattr(usage, "prompt_tokens", 0) or 0,
                "output_tokens": getattr(usage, "completion_tokens", 0) or 0,
            }
        except Exception as exc:
            return {
                "level": level,
                "index": index,
                "success": False,
                "duration_ms": (time.perf_counter() - started) * 1_000,
                "status_code": getattr(exc, "status_code", None),
                "error_type": type(exc).__name__,
                "error": str(exc),
            }

    reports: list[dict[str, Any]] = []
    highest_stable: int | None = None
    for level in levels:
        total = level * waves
        with ThreadPoolExecutor(max_workers=level) as executor:
            attempts = list(executor.map(lambda index: one(level, index), range(total)))
        successes = [item for item in attempts if item["success"]]
        durations = sorted(float(item["duration_ms"]) for item in attempts)
        stable = len(successes) == total
        reports.append(
            {
                "concurrency": level,
                "waves": waves,
                "requests": total,
                "successes": len(successes),
                "failures": total - len(successes),
                "stable": stable,
                "latency_ms": {
                    "p50": _percentile(durations, 0.50),
                    "p95": _percentile(durations, 0.95),
                    "max": max(durations),
                },
                "error_types": _counts(
                    str(item.get("error_type")) for item in attempts if not item["success"]
                ),
                "attempts": attempts,
            }
        )
        if not stable:
            break
        highest_stable = level
    report = {
        "probe": "deepplanning-provider-stability-v1",
        "created_at": time.time(),
        "model": config.model.model,
        "base_url": config.model.base_url,
        "transport_retries": 0,
        "task_prompts_used": False,
        "levels_requested": list(levels),
        "waves": waves,
        "levels": reports,
        "highest_stable_total_concurrency": highest_stable,
        "recommended_workers_per_arm": highest_stable // 2 if highest_stable else None,
    }
    target = output or config.deepplanning.results_dir / "api-probes" / f"probe-{int(report['created_at'])}.json"
    _write_json_atomic(target, report)
    report["output"] = str(target)
    return report


def run_deepplanning_benchmark(
    config: ExperimentConfig,
    *,
    task_keys: Sequence[str] = (),
    domains: Sequence[str] = (),
    run_index: int = 0,
    run_label: str = "full",
    limit: int | None = None,
    restart: bool = False,
    progress: Callable[[Mapping[str, Any]], None] | None = None,
) -> DeepPlanningRunSummary:
    _validate_protocol(config)
    app = config.deepplanning
    if run_index < 0 or run_index >= app.run_count:
        raise ValueError(f"run_index must be in [0, {app.run_count - 1}]")
    tasks = load_deepplanning_tasks(config)
    if domains:
        allowed = set(domains)
        tasks = [task for task in tasks if task.domain in allowed or task.key.rsplit("-", 1)[0] in allowed]
    if task_keys:
        by_key = {task.key: task for task in tasks}
        unknown = sorted(set(task_keys) - set(by_key))
        if unknown:
            raise ValueError(f"unknown DeepPlanning task keys: {unknown}")
        tasks = [by_key[key] for key in task_keys]
    if limit is not None:
        tasks = tasks[:limit]
    arm = _arm(config)
    if not re.fullmatch(r"[a-z0-9][a-z0-9-]*", run_label):
        raise ValueError("run_label must contain only lowercase letters, digits, and hyphens")
    run_root = app.results_dir / run_label / ("graphptc" if arm == "GraphPTC" else "fewshot-ptc") / f"run-{run_index}"
    results_path = run_root / "results.jsonl"
    progress_path = run_root / "progress.jsonl"
    shopping_roots = _prepare_shopping_databases(config, run_root, tasks, restart)
    inspection = inspect_deepplanning(config)
    signature = _sha256({
        "config": dataclasses.asdict(config), "inspection": inspection,
        "tasks": [task.key for task in tasks], "run_index": run_index, "run_label": run_label,
        "arm": arm, "prompt": PTC_GUIDANCE, "demo": _demo(config.runtime.graph_adaptation_mode == "generic"),
    })
    run_root.mkdir(parents=True, exist_ok=True)
    if restart:
        results_path.unlink(missing_ok=True)
        progress_path.unlink(missing_ok=True)
        shutil.rmtree(run_root / "artifacts", ignore_errors=True)
        shutil.rmtree(run_root / "graphs", ignore_errors=True)
    existing = _read_jsonl(results_path)
    if any(row.get("run_signature") != signature for row in existing):
        raise ValueError("existing DeepPlanning results use another run signature")
    completed = {str(row["task_key"]) for row in existing if row.get("status") in {"finished", "failed"}}
    pending = [task for task in tasks if task.key not in completed]
    lock = threading.Lock()
    started = time.perf_counter()

    def append(path: Path, row: Mapping[str, Any]) -> None:
        with lock:
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(dict(row), ensure_ascii=False, default=repr) + "\n")

    def run_one(task: DeepPlanningTask) -> dict[str, Any]:
        event = {"task_key": task.key, "status": "started", "started_at": time.time(), "run_signature": signature}
        append(progress_path, event)
        if progress:
            progress(event)
        db = _database_dir(config, task, shopping_roots)
        runtime = _runtime(config, task, db)
        controller: GoalGraphAdaptation | None = None
        agent_result = None
        checkpoint: dict[str, Any] = {}

        def save_checkpoint(value: dict[str, Any]) -> None:
            checkpoint.clear()
            checkpoint.update(copy.deepcopy(value))
            checkpoint_path = run_root / "checkpoints" / f"{task.key}.json"
            _write_json_atomic(checkpoint_path, value)
            append(
                progress_path,
                {
                    "task_key": task.key,
                    "status": "checkpoint",
                    "next_turn": value.get("next_turn"),
                    "ptc_blocks": (value.get("agent") or {}).get("ptc_blocks"),
                    "model_requests": (value.get("agent") or {}).get("model_requests"),
                    "timestamp": time.time(),
                },
            )
        try:
            metadata = runtime.metadata
            official_prompt = str(metadata["official_prompt"])
            schema_text = json.dumps(metadata["tools"], ensure_ascii=False, separators=(",", ":"))
            system_prompt = official_prompt + "\n\n" + PTC_GUIDANCE + "\n\nOfficial tool schemas:\n" + schema_text
            if config.runtime.graph_adaptation_mode == "generic":
                system_prompt += "\n\n" + GRAPH_GUIDANCE
                controller = GoalGraphAdaptation(
                    {}, {}, task=task.query, expose_graph_api=False
                )
                hook_kwargs = GraphAgentHooks.from_controller(controller).agent_kwargs()
                hook_kwargs["runtime_functions"] = ()
            else:
                hook_kwargs = {"runtime_functions": ()}
            model = OpenAIChatModel(config.model, config.require_api_key(config.model.api_key_env))
            agent = OriginalPTCAgent(
                model=model, search_tools=_EmptySearchTools(), runtime=config.runtime,
                system_prompt=system_prompt, user_prompt_template=USER_PROMPT,
                ptc_tool_spec=_ptc_spec(config), demonstration_messages=_demo(controller is not None),
                program_runtime=runtime, checkpoint_callback=save_checkpoint,
                **hook_kwargs,
            )
            agent_result = agent.run(task.query)
            if controller is not None:
                controller.finish(answered=agent_result.status == "success")
            messages = list(checkpoint.get("messages", []))
            if not messages or messages[-1].get("role") != "assistant" or messages[-1].get("content") != agent_result.answer:
                messages.append({"role": "assistant", "content": agent_result.answer})
            artifact_dir = run_root / "artifacts" / task.key
            artifact_dir.mkdir(parents=True, exist_ok=True)
            _write_official_output(task, agent_result.answer, messages, artifact_dir, db, run_root, arm)
            trajectory = {
                "method": arm, "benchmark_version": "DeepPlanning v1.1", "task": dataclasses.asdict(task),
                "official_runtime": metadata, "messages": messages, "agent": agent_result.to_dict(),
                "runtime": runtime.telemetry(),
            }
            (artifact_dir / "trajectory.json").write_text(json.dumps(trajectory, ensure_ascii=False, indent=2, default=repr), encoding="utf-8")
            graph_path = None
            if controller is not None:
                graph_path = run_root / "graphs" / f"{task.key}.json"
                graph_path.parent.mkdir(parents=True, exist_ok=True)
                graph_path.write_text(json.dumps(controller.graph_artifact(), ensure_ascii=False, indent=2, default=repr), encoding="utf-8")
            row = {
                "task_key": task.key, "domain": task.domain, "language": task.language, "level": task.level,
                "status": "finished", "run_signature": signature, "agent": agent_result.to_dict(),
                "runtime": runtime.telemetry(), "artifact_dir": str(artifact_dir),
                "graph_path": str(graph_path) if graph_path else None,
                "graph_telemetry": controller.telemetry() if controller else None,
            }
        except Exception as exc:
            row = {
                "task_key": task.key, "domain": task.domain, "language": task.language, "level": task.level,
                "status": "failed", "run_signature": signature, "error": f"{type(exc).__name__}: {exc}",
                "agent": agent_result.to_dict() if agent_result else None, "runtime": runtime.telemetry(),
                "graph_telemetry": controller.telemetry() if controller else None,
            }
        finally:
            runtime.close()
        row["finished_at"] = time.time()
        append(results_path, row)
        append(progress_path, {key: row.get(key) for key in ("task_key", "domain", "language", "level", "status", "finished_at")})
        if progress:
            progress(row)
        return row

    with ThreadPoolExecutor(max_workers=max(1, app.workers)) as executor:
        futures = [executor.submit(run_one, task) for task in pending]
        for future in as_completed(futures):
            future.result()
    terminal = {row["task_key"]: row for row in _read_jsonl(results_path) if row.get("task_key") in {task.key for task in tasks}}
    summary = _summarize(tasks, terminal, signature, time.perf_counter() - started, app.run_count)
    (run_root / "run.json").write_text(json.dumps({"summary": summary.to_dict(), "inspection": inspection, "arm": arm, "run_index": run_index}, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def evaluate_deepplanning_benchmark(config: ExperimentConfig, *, run_index: int = 0, run_label: str = "full") -> dict[str, Any]:
    _validate_protocol(config)
    app = config.deepplanning
    arm = _arm(config)
    run_root = app.results_dir / run_label / ("graphptc" if arm == "GraphPTC" else "fewshot-ptc") / f"run-{run_index}"
    rows = [row for row in _read_jsonl(run_root / "results.jsonl") if row.get("status") in {"finished", "failed"}]
    if not rows:
        raise ValueError(f"no DeepPlanning results found for {arm} run {run_index}")
    logs = run_root / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    script = str(Path(__file__).with_name("official.py"))
    failures = {"conversion": 0, "evaluator": 0, "aggregation": 0}
    commands: list[dict[str, Any]] = []
    for language in ("zh", "en"):
        result_dir = run_root / "official" / "travel" / "results" / f"{arm}_{language}"
        if not (result_dir / "reports").exists():
            continue
        conversion = [
            app.python_command, script, "convert-travel", "--root", app.root,
            "--result-dir", str(result_dir), "--language", language,
            "--model", config.model.model, "--base-url", str(config.model.base_url),
            "--api-key-env", config.model.api_key_env, "--timeout", str(config.model.timeout_seconds),
            "--max-retries", "30",
        ]
        rc = _logged_command(conversion, logs / f"travel-{language}-conversion.log")
        commands.append({"stage": "conversion", "language": language, "return_code": rc, "command": conversion})
        failures["conversion"] += int(rc != 0)
        if rc != 0:
            continue
        evaluation = [
            app.python_command, script, "evaluate-travel", "--root", app.root,
            "--result-dir", str(result_dir), "--language", language,
            "--workers", str(max(1, app.workers)),
        ]
        rc = _logged_command(evaluation, logs / f"travel-{language}-evaluation.log")
        commands.append({"stage": "evaluation", "language": language, "return_code": rc, "command": evaluation})
        failures["evaluator"] += int(rc != 0)
    shopping_report_root = run_root / "official" / "shopping" / "result_report"
    for level in (1, 2, 3):
        ids = [str(row["task_key"]).rsplit("-", 1)[-1] for row in rows if row.get("domain") == "shopping" and int(row.get("level") or 0) == level]
        if not ids:
            continue
        database = run_root / "official" / "shopping" / f"database_level{level}"
        output = shopping_report_root / f"database_{arm}_level{level}_{run_index}"
        command = [
            app.python_command, script, "evaluate-shopping", "--root", app.root,
            "--database-dir", str(database), "--output-dir", str(output),
        ]
        for sample_id in ids:
            command.extend(("--case-id", sample_id))
        rc = _logged_command(command, logs / f"shopping-level{level}-evaluation.log")
        commands.append({"stage": "evaluation", "level": level, "return_code": rc, "command": command})
        failures["evaluator"] += int(rc != 0)
    aggregate_output = run_root / "official" / "aggregate.json"
    command = [
        app.python_command, script, "aggregate", "--root", app.root,
        "--method", arm, "--travel-results-dir", str(run_root / "official" / "travel" / "results"),
        "--shopping-report-dir", str(shopping_report_root), "--output", str(aggregate_output),
    ]
    rc = _logged_command(command, logs / "aggregation.log")
    commands.append({"stage": "aggregation", "return_code": rc, "command": command})
    failures["aggregation"] += int(rc != 0)
    report = {
        "method": arm, "run_index": run_index, "failures": failures, "commands": commands,
        "aggregate": json.loads(aggregate_output.read_text(encoding="utf-8")) if aggregate_output.exists() else None,
        "conversion_model": config.model.model, "conversion_max_retries": 30,
        "official_evaluator": True,
    }
    (run_root / "evaluation.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def _runtime(config: ExperimentConfig, task: DeepPlanningTask, database_dir: Path) -> DeepPlanningProgramRuntime:
    app = config.deepplanning
    command = (app.python_command, str(Path(__file__).with_name("worker.py")))
    request = {
        "official_root": app.root, "domain": task.domain, "sample_id": task.sample_id,
        "language": task.language, "level": task.level, "database_dir": str(database_dir),
    }
    return DeepPlanningProgramRuntime(worker_command=command, request=request, timeout_seconds=config.runtime.code_timeout_seconds)


def _percentile(values: Sequence[float], quantile: float) -> float:
    if not values:
        return 0.0
    index = min(len(values) - 1, max(0, round((len(values) - 1) * quantile)))
    return float(values[index])


def _database_dir(config: ExperimentConfig, task: DeepPlanningTask, shopping_roots: Mapping[int, Path] | None) -> Path:
    base = Path(config.deepplanning.root) / "benchmark" / "deepplanning"
    if task.domain == "travel":
        return base / "travelplanning" / "database" / f"database_{task.language}"
    if shopping_roots is not None:
        return shopping_roots[int(task.level or 0)]
    return base / "shoppingplanning" / f"database_level{task.level}"


def _prepare_shopping_databases(config: ExperimentConfig, run_root: Path, tasks: Sequence[DeepPlanningTask], restart: bool) -> dict[int, Path]:
    levels = sorted({int(task.level) for task in tasks if task.domain == "shopping" and task.level is not None})
    official = Path(config.deepplanning.root) / "benchmark" / "deepplanning" / "shoppingplanning"
    roots: dict[int, Path] = {}
    for level in levels:
        target = run_root / "official" / "shopping" / f"database_level{level}"
        if restart and target.exists():
            shutil.rmtree(target)
        if not target.exists():
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copytree(official / f"database_level{level}", target)
        roots[level] = target
    return roots


def _write_official_output(task: DeepPlanningTask, answer: str, messages: list[dict[str, Any]], artifact_dir: Path, database_dir: Path, run_root: Path, arm: str) -> None:
    if task.domain == "travel":
        match = re.findall(r"<plan>(.*?)</plan>", answer or "", flags=re.DOTALL | re.IGNORECASE)
        plan = "\n\n".join(value.strip() for value in match if value.strip())
        (artifact_dir / "report.txt").write_text(plan, encoding="utf-8")
        official_report = run_root / "official" / "travel" / "results" / f"{arm}_{task.language}" / "reports" / f"id_{task.sample_id}.txt"
        official_report.parent.mkdir(parents=True, exist_ok=True)
        official_report.write_text(plan, encoding="utf-8")
    else:
        case_dir = database_dir / f"case_{task.sample_id}"
        (case_dir / "messages.json").write_text(json.dumps({"step": len(messages), "description": "GraphPTC completed", "messages": messages}, ensure_ascii=False, indent=2, default=repr), encoding="utf-8")
        shutil.copy2(case_dir / "cart.json", artifact_dir / "cart.json")
        shutil.copy2(case_dir / "messages.json", artifact_dir / "messages.json")


def _ptc_spec(config: ExperimentConfig) -> dict[str, Any]:
    if config.runtime.graph_adaptation_mode == "off":
        return copy.deepcopy(PTC_TOOL_SPEC)
    if config.runtime.graph_adaptation_mode != "generic":
        raise ValueError("DeepPlanning graph_adaptation_mode must be off or generic")
    return extend_ptc_spec_with_graph_control(
        PTC_TOOL_SPEC,
        include_input_artifacts=False,
        target_description="Use task for this DeepPlanning episode.",
    )


def _validate_protocol(config: ExperimentConfig) -> None:
    app = config.deepplanning
    if app.official_commit != DEEPPLANNING_OFFICIAL_COMMIT or app.data_revision != DEEPPLANNING_DATA_REVISION:
        raise ValueError("DeepPlanning code or data revision differs from the audited v1.1 protocol")
    if config.runtime.graph_adaptation_mode not in {"off", "generic"}:
        raise ValueError("DeepPlanning graph_adaptation_mode must be off or generic")
    if config.runtime.max_stdout_chars != 8000:
        raise ValueError("DeepPlanning max_stdout_chars must be 8000")
    if config.runtime.max_turns != app.max_model_calls or config.runtime.max_ptc_blocks != app.max_model_calls - 1:
        raise ValueError("DeepPlanning total model-call budget must be 400")
    if not math.isinf(config.runtime.task_timeout_seconds):
        raise ValueError("DeepPlanning must not impose a non-official per-task wall-clock limit")
    if (
        config.model.max_retries != 29
        or config.model.retry_backoff_seconds != 1.5
        or not config.model.retry_all_errors
    ):
        raise ValueError("DeepPlanning model calls must use the audited 30-attempt, 1.5-second retry policy")


def compare_deepplanning_configs(graph: ExperimentConfig, baseline: ExperimentConfig) -> dict[str, Any]:
    graph_data, baseline_data = dataclasses.asdict(graph), dataclasses.asdict(baseline)
    graph_data["runtime"]["graph_adaptation_mode"] = "off"
    if graph_data != baseline_data:
        raise ValueError("DeepPlanning arms differ beyond graph_adaptation_mode")
    if graph.runtime.graph_adaptation_mode != "generic" or baseline.runtime.graph_adaptation_mode != "off":
        raise ValueError("expected GraphPTC generic and Fewshot PTC off")
    return {"matched": True, "only_difference": "runtime.graph_adaptation_mode"}


def compare_deepplanning_benchmarks(
    graph: ExperimentConfig,
    baseline: ExperimentConfig,
    *,
    run_label: str = "full",
    run_index: int = 0,
    output: Path | None = None,
) -> dict[str, Any]:
    compare_deepplanning_configs(graph, baseline)
    base = graph.deepplanning.results_dir / run_label
    graph_root = base / "graphptc" / f"run-{run_index}"
    baseline_root = base / "fewshot-ptc" / f"run-{run_index}"
    graph_rows = _terminal_rows(graph_root)
    baseline_rows = _terminal_rows(baseline_root)
    if set(graph_rows) != set(baseline_rows):
        raise ValueError("DeepPlanning result arms do not have identical task keys")
    graph_scores = _official_task_scores(graph_root, "GraphPTC")
    baseline_scores = _official_task_scores(baseline_root, "Fewshot PTC")
    pairs = []
    for key in sorted(graph_rows):
        graph_row, baseline_row = graph_rows[key], baseline_rows[key]
        graph_score, baseline_score = graph_scores.get(key), baseline_scores.get(key)
        pairs.append(
            {
                "task_key": key,
                "domain": graph_row.get("domain"),
                "language": graph_row.get("language"),
                "level": graph_row.get("level"),
                "graphptc_score": graph_score,
                "fewshot_ptc_score": baseline_score,
                "score_delta": (
                    graph_score - baseline_score
                    if graph_score is not None and baseline_score is not None
                    else None
                ),
                "graphptc_agent_status": (graph_row.get("agent") or {}).get("status"),
                "fewshot_ptc_agent_status": (baseline_row.get("agent") or {}).get("status"),
                "graphptc_model_calls": (graph_row.get("agent") or {}).get("model_requests", 0),
                "fewshot_ptc_model_calls": (baseline_row.get("agent") or {}).get("model_requests", 0),
                "graphptc_tool_calls": (graph_row.get("runtime") or {}).get("tool_calls", 0),
                "fewshot_ptc_tool_calls": (baseline_row.get("runtime") or {}).get("tool_calls", 0),
            }
        )
    graph_summary = _summary_from_rows(graph_rows, graph_root)
    baseline_summary = _summary_from_rows(baseline_rows, baseline_root)
    graph_control = _graph_control_summary(graph_rows)
    complete_four_runs = run_label == "full" and all(
        (base / arm / f"run-{index}" / "evaluation.json").exists()
        for arm in ("graphptc", "fewshot-ptc")
        for index in range(graph.deepplanning.run_count)
    )
    report = {
        "benchmark": "Qwen/DeepPlanning v1.1",
        "run_label": run_label,
        "run_index": run_index,
        "leaderboard_equivalent": complete_four_runs,
        "matched_config": compare_deepplanning_configs(graph, baseline),
        "arms": {"GraphPTC": graph_summary, "Fewshot PTC": baseline_summary},
        "official": {
            "GraphPTC": _load_json(graph_root / "official" / "aggregate.json"),
            "Fewshot PTC": _load_json(baseline_root / "official" / "aggregate.json"),
        },
        "paired": pairs,
        "paired_score": _paired_score_summary(pairs),
        "graph_control": graph_control,
        "operational_deltas": {
            key: graph_summary[key] - baseline_summary[key]
            for key in ("incomplete_tasks", "execution_failure_tasks", "execution_failure_blocks", "model_calls", "tool_calls", "input_tokens", "output_tokens", "duration_seconds")
        },
    }
    destination = output or base / f"paired-run-{run_index}.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def _summarize(tasks: Sequence[DeepPlanningTask], terminal: Mapping[str, Mapping[str, Any]], signature: str, duration: float, run_count: int) -> DeepPlanningRunSummary:
    rows = [terminal.get(task.key, {}) for task in tasks]
    agents = [row.get("agent") or {} for row in rows]
    blocks = [block for agent in agents for block in agent.get("blocks", [])]
    usages = [agent.get("usage", {}) for agent in agents]
    return DeepPlanningRunSummary(
        selected=len(tasks), processed=len(terminal), runner_failures=sum(row.get("status") == "failed" for row in rows),
        execution_failure_tasks=sum(any(not block.get("success", False) for block in agent.get("blocks", [])) for agent in agents),
        execution_failure_blocks=sum(not block.get("success", False) for block in blocks),
        incomplete_tasks=sum(agent.get("status") != "success" for agent in agents),
        model_calls=sum(int(agent.get("model_requests", 0)) for agent in agents),
        tool_calls=sum(int((row.get("runtime") or {}).get("tool_calls", 0)) for row in rows),
        input_tokens=sum(int(usage.get("input_tokens", 0)) for usage in usages),
        output_tokens=sum(int(usage.get("output_tokens", 0)) for usage in usages),
        duration_seconds=duration, run_signature=signature,
        leaderboard_equivalent_repetitions=False,
    )


def _terminal_rows(root: Path) -> dict[str, dict[str, Any]]:
    rows = _read_jsonl(root / "results.jsonl")
    return {str(row["task_key"]): row for row in rows if row.get("status") in {"finished", "failed"}}


def _summary_from_rows(rows: Mapping[str, Mapping[str, Any]], root: Path) -> dict[str, Any]:
    values = list(rows.values())
    agents = [row.get("agent") or {} for row in values]
    blocks = [block for agent in agents for block in agent.get("blocks", [])]
    usage = [agent.get("usage") or {} for agent in agents]
    return {
        "tasks": len(values),
        "runner_failures": sum(row.get("status") == "failed" for row in values),
        "incomplete_tasks": sum(agent.get("status") != "success" for agent in agents),
        "execution_failure_tasks": sum(any(not block.get("success", False) for block in agent.get("blocks", [])) for agent in agents),
        "execution_failure_blocks": sum(not block.get("success", False) for block in blocks),
        "model_calls": sum(int(agent.get("model_requests", 0)) for agent in agents),
        "tool_calls": sum(int((row.get("runtime") or {}).get("tool_calls", 0)) for row in values),
        "input_tokens": sum(int(item.get("input_tokens", 0)) for item in usage),
        "output_tokens": sum(int(item.get("output_tokens", 0)) for item in usage),
        "duration_seconds": float(((_load_json(root / "run.json") or {}).get("summary") or {}).get("duration_seconds", 0)),
        "conversion_failures": int((_load_json(root / "evaluation.json") or {}).get("failures", {}).get("conversion", 0)),
        "evaluator_failures": int((_load_json(root / "evaluation.json") or {}).get("failures", {}).get("evaluator", 0)),
    }


def _official_task_scores(root: Path, method: str) -> dict[str, float]:
    scores: dict[str, float] = {}
    for language in ("zh", "en"):
        evaluation = root / "official" / "travel" / "results" / f"{method}_{language}" / "evaluation"
        for path in evaluation.glob("id_*_score.json"):
            data = _load_json(path) or {}
            scores[f"travel-{language}-{data.get('sample_id')}"] = float((data.get("scores") or {}).get("case_acc", 0.0))
    reports = root / "official" / "shopping" / "result_report"
    for level in (1, 2, 3):
        path = reports / f"database_{method}_level{level}_{root.name.removeprefix('run-')}" / "summary_report.json"
        data = _load_json(path) or {}
        for item in data.get("case_results", []):
            sample_id = str(item.get("case_name", "")).removeprefix("case_")
            scores[f"shopping-level{level}-{sample_id}"] = float(item.get("case_score", 0.0))
    return scores


def _paired_score_summary(pairs: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    comparable = [float(item["score_delta"]) for item in pairs if item.get("score_delta") is not None]
    return {
        "comparable_tasks": len(comparable),
        "graphptc_wins": sum(value > 0 for value in comparable),
        "ties": sum(value == 0 for value in comparable),
        "fewshot_ptc_wins": sum(value < 0 for value in comparable),
        "mean_delta": sum(comparable) / len(comparable) if comparable else None,
    }


def _graph_control_summary(rows: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    telemetry = [row.get("graph_telemetry") or {} for row in rows.values()]
    distribution: dict[str, int] = {}
    for item in telemetry:
        for action, count in (item.get("action_distribution") or {}).items():
            distribution[str(action)] = distribution.get(str(action), 0) + int(count)
    adaptive_actions = distribution.get("PATCH", 0) + distribution.get("REPLAN", 0)
    return {
        "graph_assessment_present": all(bool(item) for item in telemetry),
        "graph_delta_observations": sum(int(item.get("observation_calls", 0)) for item in telemetry),
        "realized_graph_deltas": sum(int(item.get("realized_graph_deltas", 0)) for item in telemetry),
        "missed_graph_deltas": sum(int(item.get("missed_graph_deltas", 0)) for item in telemetry),
        "action_distribution": distribution,
        "patch_or_replan_actions": adaptive_actions,
        "observed_later_action_change": adaptive_actions > 0,
        "causal_influence_proven": False,
        "conclusion": (
            "GRAPH_DELTA was model-visible, but no PATCH or REPLAN action followed; no actual adaptive influence was observed."
            if adaptive_actions == 0
            else "A PATCH or REPLAN followed model-visible GRAPH_DELTA, but this matched smoke has no no-delta ablation and does not prove causality."
        ),
    }


def _load_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _pip_freeze(config: ExperimentConfig) -> list[str]:
    return _run_text((config.deepplanning.python_command, "-m", "pip", "freeze")).splitlines()


def _run_text(command: Sequence[str]) -> str:
    result = subprocess.run(list(command), check=True, capture_output=True, text=True, encoding="utf-8")
    return result.stdout.strip() or result.stderr.strip()


def _logged_command(command: Sequence[str], log_path: Path) -> int:
    result = subprocess.run(list(command), capture_output=True, text=True, encoding="utf-8", errors="replace")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(result.stdout + result.stderr, encoding="utf-8")
    return result.returncode


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _write_json_atomic(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=repr), encoding="utf-8")
    temporary.replace(path)


def _sha256(value: Any) -> str:
    raw = json.dumps(value, ensure_ascii=False, sort_keys=True, default=str, separators=(",", ":")).encode()
    return hashlib.sha256(raw).hexdigest()


def _counts(values: Any) -> dict[str, int]:
    result: dict[str, int] = {}
    for value in values:
        result[value] = result.get(value, 0) + 1
    return result


def _arm(config: ExperimentConfig) -> str:
    return "GraphPTC" if config.runtime.graph_adaptation_mode == "generic" else "Fewshot PTC"
