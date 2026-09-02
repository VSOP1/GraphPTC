from __future__ import annotations

import copy
import dataclasses
import hashlib
import json
import subprocess
import threading
from collections.abc import Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from ...config import ExperimentConfig
from ...graph.adaptation import GoalGraphAdaptation
from ...graph.hooks import GraphAgentHooks, extend_ptc_spec_with_graph_control
from .runtime import InterCodeProgramRuntime
from ...model import OpenAIChatModel
from ...agents.original_ptc import OriginalPTCAgent, PTC_TOOL_SPEC


INTERCODE_SYSTEM_PROMPT = """You are playing the official InterCode {environment_name} multi-turn game through
programmatic tool calling. Your only directly callable model tool is programmatic_tool_call. Its
Python source runs in one persistent task namespace.

Inside that namespace, call {tool_signature}. Each call is one official InterCode action. The adapter
returns the execution observation, the official reward in [0, 1], whether the action executed, the
action number, and the remaining action budget. As in the official Try Again experiment, the harness
evaluates the reward immediately after every action and terminates on reward 1 or after 10 actions.
Do not call submit yourself. Every environment action must be made through {tool_name}(); imports,
open(), pathlib, os, subprocess, and other host access are unavailable.

Use a PTC block as one coherent execution phase. It may contain multiple calls, loops, conditionals,
parsing, and aggregation when later operations follow mechanically from earlier outputs. Stop the
block when a new semantic decision is required. Print only compact information needed for that
decision. Never answer in prose while actions remain. Do not access host files, environment variables,
the host shell, or external networks; interact only through {tool_name}()."""

GRAPH_GUIDANCE = """The graph-control fields describe the intended dependency update for this PTC block. Use
CONTINUE for a new execution phase, PATCH for correcting failed code or commands, and REPLAN when
changing the dependency path. Use target `task` and state an observable expected_change."""

USER_PROMPT = 'Query: "{question}"'
FINALIZE_PROMPT = (
    "The official InterCode action budget is exhausted. Do not emit another tool call. "
    "Return a short completion marker."
)


class EmptySearchTools:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []


@dataclass(frozen=True)
class InterCodeRunSummary:
    selected: int
    processed: int
    successes: int
    success_rate: float
    error_percentage: float
    mean_actions: float
    execution_failure_tasks: int
    runner_failures: int
    run_signature: str

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


def inspect_intercode(config: ExperimentConfig) -> dict[str, Any]:
    response = _worker_request(
        config.intercode.worker_command,
        {"type": "inspect", "root": config.intercode.root},
    )
    inspection = {key: value for key, value in response.items() if key != "type"}
    _validate_alignment(config, inspection)
    return inspection


def run_intercode_benchmark(
    config: ExperimentConfig,
    *,
    task_ids: Sequence[str] = (),
    limit: int | None = None,
    restart: bool = False,
) -> InterCodeRunSummary:
    app = config.intercode
    inspection = inspect_intercode(config)
    available = {
        str(item["task_id"]): dict(item)
        for item in inspection["tasks"]
        if isinstance(item, dict)
    }
    if task_ids:
        unknown = sorted(set(task_ids) - available.keys())
        if unknown:
            raise ValueError(f"unknown InterCode task IDs: {unknown[:5]}")
        selected = [available[task_id] for task_id in task_ids]
    else:
        selected = list(available.values())
    if limit is not None:
        selected = selected[:limit]
    signature = _run_signature(config, inspection, [item["task_id"] for item in selected])
    app.results_path.parent.mkdir(parents=True, exist_ok=True)
    app.artifact_dir.mkdir(parents=True, exist_ok=True)
    app.graph_dir.mkdir(parents=True, exist_ok=True)
    if restart and app.results_path.exists():
        app.results_path.unlink()
    existing = _read_jsonl(app.results_path)
    if any(item.get("run_signature") != signature for item in existing):
        raise ValueError("existing InterCode results use another run signature")
    terminal = _terminal_records(existing)
    pending = [item for item in selected if item["task_id"] not in terminal]
    write_lock = threading.Lock()

    def append(record: dict[str, Any]) -> None:
        with write_lock, app.results_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False, default=repr) + "\n")

    def run_one(spec: Mapping[str, Any]) -> dict[str, Any]:
        task_id = str(spec["task_id"])
        append({"task_id": task_id, "status": "started", "run_signature": signature})
        record = _run_one(config, spec, signature)
        append(record)
        return record

    if pending:
        with ThreadPoolExecutor(max_workers=min(app.workers, len(pending))) as executor:
            futures = [executor.submit(run_one, item) for item in pending]
            for future in as_completed(futures):
                future.result()

    records = _terminal_records(_read_jsonl(app.results_path))
    selected_records = [records[str(item["task_id"])] for item in selected]
    summary = _summarize(selected_records, signature)
    report = {
        "schema_version": 1,
        "benchmark": "intercode",
        "official_alignment": {key: value for key, value in inspection.items() if key != "tasks"},
        "model": dataclasses.asdict(config.model),
        "runtime": dataclasses.asdict(config.runtime),
        "summary": summary.to_dict(),
        "scoring": _score_records(selected_records),
    }
    app.report_path.parent.mkdir(parents=True, exist_ok=True)
    app.report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, default=repr), encoding="utf-8"
    )
    return summary


def evaluate_intercode_benchmark(config: ExperimentConfig) -> dict[str, Any]:
    inspection = inspect_intercode(config)
    expected = config.intercode.expected_bash_tasks + config.intercode.expected_sql_tasks
    records = _terminal_records(_read_jsonl(config.intercode.results_path))
    if len(records) != expected:
        raise ValueError(
            f"InterCode results are incomplete: expected {expected}, found {len(records)}"
        )
    report = {
        "schema_version": 1,
        "benchmark": "intercode",
        "official_alignment": {key: value for key, value in inspection.items() if key != "tasks"},
        "model": dataclasses.asdict(config.model),
        "runtime": dataclasses.asdict(config.runtime),
        "summary": _summarize(list(records.values()), next(iter(records.values()))["run_signature"]).to_dict(),
        "scoring": _score_records(list(records.values())),
    }
    config.intercode.report_path.parent.mkdir(parents=True, exist_ok=True)
    config.intercode.report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, default=repr), encoding="utf-8"
    )
    return report


def compare_intercode_benchmarks(
    graph_config: ExperimentConfig,
    baseline_config: ExperimentConfig,
    output_path: Path,
) -> dict[str, Any]:
    _validate_arm_pair(graph_config, baseline_config)
    graph = _terminal_records(_read_jsonl(graph_config.intercode.results_path))
    baseline = _terminal_records(_read_jsonl(baseline_config.intercode.results_path))
    expected = graph_config.intercode.expected_bash_tasks + graph_config.intercode.expected_sql_tasks
    if set(graph) != set(baseline) or len(graph) != expected:
        raise ValueError("InterCode paired results do not contain the same complete task IDs")
    graph_scoring = _score_records(list(graph.values()))
    baseline_scoring = _score_records(list(baseline.values()))
    graph_success = {key: _official_success(value) for key, value in graph.items()}
    baseline_success = {key: _official_success(value) for key, value in baseline.items()}
    wins = sum(graph_success[key] and not baseline_success[key] for key in graph)
    losses = sum(baseline_success[key] and not graph_success[key] for key in graph)
    report = {
        "schema_version": 1,
        "benchmark": "intercode",
        "tasks": expected,
        "graphptc": graph_scoring,
        "baseline": baseline_scoring,
        "difference": {
            "overall_success_rate": graph_scoring["overall"]["success_rate"]
            - baseline_scoring["overall"]["success_rate"],
            "bash_success_rate": graph_scoring["bash"]["success_rate"]
            - baseline_scoring["bash"]["success_rate"],
            "sql_success_rate": graph_scoring["sql"]["success_rate"]
            - baseline_scoring["sql"]["success_rate"],
            "paired_wins": wins,
            "paired_losses": losses,
            "paired_ties": expected - wins - losses,
        },
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return report


def _run_one(
    config: ExperimentConfig, spec: Mapping[str, Any], signature: str
) -> dict[str, Any]:
    task_id = str(spec["task_id"])
    environment = str(spec["environment"])
    arm = "graphptc" if config.runtime.graph_adaptation_mode == "generic" else "baseline"
    runtime = InterCodeProgramRuntime(
        worker_command=config.intercode.worker_command,
        root=config.intercode.root,
        task_id=task_id,
        environment=environment,
        data_path=str(spec["data_path"]),
        data_index=int(spec["data_index"]),
        image_name=str(spec["image_name"]),
        container_prefix=f"graphptc-intercode-{arm}-{_artifact_key(task_id)}",
        max_actions=config.intercode.max_actions,
        timeout_seconds=config.runtime.code_timeout_seconds,
    )
    controller: GoalGraphAdaptation | None = None
    agent_result = None
    evaluation: dict[str, Any] | None = None
    metadata: dict[str, Any] = {}
    try:
        metadata = runtime.metadata
        if config.runtime.graph_adaptation_mode == "generic":
            controller = GoalGraphAdaptation(
                {}, {}, task=str(metadata["query"]), expose_graph_api=False
            )
            hooks = GraphAgentHooks.from_controller(controller).agent_kwargs()
            hooks["runtime_functions"] = ()
        else:
            hooks = {"runtime_functions": ()}
        hooks["message_projection_callback"] = _dialogue_projection(environment)
        system_prompt = _system_prompt(environment, graph=config.runtime.graph_adaptation_mode == "generic")
        agent = OriginalPTCAgent(
            model=OpenAIChatModel(
                config.model, config.require_api_key(config.model.api_key_env)
            ),
            search_tools=EmptySearchTools(),  # type: ignore[arg-type]
            runtime=config.runtime,
            system_prompt=system_prompt,
            user_prompt_template=USER_PROMPT,
            finalize_prompt=FINALIZE_PROMPT,
            ptc_tool_spec=_ptc_spec(config, environment),
            demonstration_messages=(),
            program_runtime=runtime,
            **hooks,
        )
        agent_result = agent.run(str(metadata["query"]))
        evaluation = runtime.evaluate()
        if controller is not None:
            controller.finish(answered=bool(evaluation.get("success")))
        graph_path: Path | None = None
        if controller is not None:
            graph_path = config.intercode.graph_dir / f"{_artifact_key(task_id)}.json"
            graph_path.write_text(
                json.dumps(controller.graph_artifact(), ensure_ascii=False, indent=2, default=repr),
                encoding="utf-8",
            )
        artifact = config.intercode.artifact_dir / _artifact_key(task_id)
        artifact.mkdir(parents=True, exist_ok=True)
        (artifact / "execution.json").write_text(
            json.dumps(agent_result.to_dict(), ensure_ascii=False, indent=2, default=repr),
            encoding="utf-8",
        )
        return {
            "schema_version": 1,
            "benchmark": "intercode",
            "task_id": task_id,
            "status": "finished",
            "run_signature": signature,
            "environment": environment,
            "filesystem": metadata.get("filesystem"),
            "hardness": metadata.get("hardness"),
            "database": metadata.get("database"),
            "official_evaluation": evaluation,
            "agent": agent_result.to_dict(),
            "graph_path": str(graph_path) if graph_path is not None else None,
            "graph_telemetry": controller.telemetry() if controller is not None else None,
        }
    except Exception as exc:
        return {
            "schema_version": 1,
            "benchmark": "intercode",
            "task_id": task_id,
            "status": "failed",
            "run_signature": signature,
            "environment": environment,
            "filesystem": spec.get("filesystem"),
            "hardness": spec.get("hardness"),
            "database": spec.get("database"),
            "error": f"{type(exc).__name__}: {exc}",
            "official_evaluation": evaluation,
            "agent": agent_result.to_dict() if agent_result is not None else None,
            "graph_telemetry": controller.telemetry() if controller is not None else None,
        }
    finally:
        runtime.close()


def _system_prompt(environment: str, *, graph: bool) -> str:
    if environment == "bash":
        prompt = INTERCODE_SYSTEM_PROMPT.format(
            environment_name="Bash",
            tool_signature="bash(command=...) to execute a Bash command",
            tool_name="bash",
        )
    elif environment == "sql":
        prompt = INTERCODE_SYSTEM_PROMPT.format(
            environment_name="SQL",
            tool_signature="sql(query=...) to execute a MySQL query",
            tool_name="sql",
        )
        prompt += (
            "\n\nYou initially know only the natural-language query. Use SHOW TABLES and "
            "DESCRIBE when schema discovery is needed."
        )
    else:
        raise ValueError(f"unsupported InterCode environment: {environment}")
    return prompt + ("\n\n" + GRAPH_GUIDANCE if graph else "")


def _ptc_spec(config: ExperimentConfig, environment: str) -> dict[str, Any]:
    tool = "bash" if environment == "bash" else "sql"
    spec = copy.deepcopy(PTC_TOOL_SPEC)
    spec["function"]["description"] = (
        f"Execute one coherent Python phase in the persistent InterCode-{environment} "
        f"namespace. The {tool} function and mutable state dictionary are available."
    )
    spec["function"]["parameters"]["properties"]["code"]["description"] = (
        f"Python source for one InterCode phase; call {tool} multiple times only when "
        "the subsequent operations are mechanically determined."
    )
    if config.runtime.graph_adaptation_mode == "off":
        return spec
    return extend_ptc_spec_with_graph_control(
        spec,
        include_input_artifacts=False,
        target_description="Use `task` for this InterCode episode.",
    )


def _dialogue_projection(environment: str):
    limit = 7 if environment == "bash" else 5

    def project(messages: list[dict[str, Any]]) -> None:
        if len(messages) > limit + 1:
            messages[:] = [messages[0], *messages[-limit:]]

    return project


def _validate_alignment(config: ExperimentConfig, inspection: Mapping[str, Any]) -> None:
    app = config.intercode
    mismatches: dict[str, Any] = {}
    expected = {
        "official_commit": app.official_commit,
        "bash_tasks": app.expected_bash_tasks,
        "sql_tasks": app.expected_sql_tasks,
    }
    for key, value in expected.items():
        if inspection.get(key) != value:
            mismatches[key] = {"expected": value, "actual": inspection.get(key)}
    protocol = inspection.get("official_protocol", {})
    if not isinstance(protocol, Mapping) or protocol.get("max_turns") != app.max_actions:
        mismatches["max_actions"] = {
            "expected": app.max_actions,
            "actual": protocol.get("max_turns") if isinstance(protocol, Mapping) else None,
        }
    images = inspection.get("images", {})
    if not isinstance(images, Mapping) or not all(images.values()):
        mismatches["images"] = images
    if app.prompt_variant != "intercode-ptc-zero-shot":
        mismatches["prompt_variant"] = app.prompt_variant
    if config.model.temperature != 0.0:
        mismatches["temperature"] = config.model.temperature
    if config.runtime.max_turns != 11 or config.runtime.max_ptc_blocks != 10:
        mismatches["ptc_turn_budget"] = {
            "max_turns": config.runtime.max_turns,
            "max_ptc_blocks": config.runtime.max_ptc_blocks,
        }
    if config.runtime.graph_adaptation_mode not in {"off", "generic"}:
        mismatches["graph_adaptation_mode"] = config.runtime.graph_adaptation_mode
    if mismatches:
        raise ValueError(f"InterCode official alignment mismatch: {mismatches}")


def _validate_arm_pair(graph: ExperimentConfig, baseline: ExperimentConfig) -> None:
    if graph.model != baseline.model:
        raise ValueError("InterCode arms must use the same model configuration")
    if replace(graph.runtime, graph_adaptation_mode="off") != baseline.runtime:
        raise ValueError("InterCode arms may only differ in graph adaptation mode")
    graph_app = replace(
        graph.intercode,
        results_path=baseline.intercode.results_path,
        report_path=baseline.intercode.report_path,
        artifact_dir=baseline.intercode.artifact_dir,
        graph_dir=baseline.intercode.graph_dir,
    )
    if graph_app != baseline.intercode:
        raise ValueError("InterCode arms may only differ in output paths")
    if graph.runtime.graph_adaptation_mode != "generic":
        raise ValueError("InterCode GraphPTC arm must use generic graph adaptation")
    if baseline.runtime.graph_adaptation_mode != "off":
        raise ValueError("InterCode baseline arm must disable graph adaptation")


def _score_records(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    bash = [item for item in records if item.get("environment") == "bash"]
    sql = [item for item in records if item.get("environment") == "sql"]
    return {
        "overall": _score_slice(records),
        "bash": _score_slice(bash),
        "sql": _score_slice(sql),
        "bash_by_filesystem": {
            f"fs{number}": _score_slice(
                [item for item in bash if item.get("filesystem") == number]
            )
            for number in range(1, 5)
        },
        "sql_by_hardness": {
            hardness: _score_slice(
                [item for item in sql if str(item.get("hardness", "")).lower() == hardness]
            )
            for hardness in ("easy", "medium", "hard", "extra")
        },
    }


def _score_slice(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    count = len(records)
    successes = sum(_official_success(item) for item in records)
    actions = sum(_evaluation_int(item, "actions") for item in records)
    invalid = sum(_evaluation_int(item, "invalid_actions") for item in records)
    return {
        "tasks": count,
        "successes": successes,
        "success_rate": 100.0 * successes / count if count else 0.0,
        "error_percentage": 100.0 * invalid / actions if actions else 0.0,
        "mean_actions": actions / count if count else 0.0,
        "mean_reward": (
            sum(_evaluation_float(item, "max_reward") for item in records) / count
            if count
            else 0.0
        ),
        "runner_failures": sum(item.get("status") != "finished" for item in records),
    }


def _summarize(
    records: Sequence[Mapping[str, Any]], signature: str
) -> InterCodeRunSummary:
    score = _score_slice(records)
    return InterCodeRunSummary(
        selected=len(records),
        processed=len(records),
        successes=int(score["successes"]),
        success_rate=float(score["success_rate"]),
        error_percentage=float(score["error_percentage"]),
        mean_actions=float(score["mean_actions"]),
        execution_failure_tasks=sum(
            bool((item.get("agent") or {}).get("blocks"))
            and any(not block.get("success", False) for block in (item.get("agent") or {}).get("blocks", []))
            for item in records
        ),
        runner_failures=int(score["runner_failures"]),
        run_signature=signature,
    )


def _official_success(record: Mapping[str, Any]) -> bool:
    evaluation = record.get("official_evaluation")
    return bool(evaluation.get("success")) if isinstance(evaluation, Mapping) else False


def _evaluation_int(record: Mapping[str, Any], key: str) -> int:
    evaluation = record.get("official_evaluation")
    return int(evaluation.get(key, 0) or 0) if isinstance(evaluation, Mapping) else 0


def _evaluation_float(record: Mapping[str, Any], key: str) -> float:
    evaluation = record.get("official_evaluation")
    return float(evaluation.get(key, 0.0) or 0.0) if isinstance(evaluation, Mapping) else 0.0


def _run_signature(
    config: ExperimentConfig, inspection: Mapping[str, Any], task_ids: Sequence[str]
) -> str:
    payload = {
        "official_commit": inspection.get("official_commit"),
        "model": dataclasses.asdict(config.model),
        "runtime": dataclasses.asdict(config.runtime),
        "intercode": dataclasses.asdict(config.intercode),
        "task_ids": list(task_ids),
    }
    encoded = json.dumps(payload, sort_keys=True, default=str, separators=(",", ":"))
    return hashlib.sha256(encoded.encode()).hexdigest()


def _worker_request(command: Sequence[str], request: Mapping[str, Any]) -> dict[str, Any]:
    completed = subprocess.run(
        command,
        input=json.dumps(dict(request), ensure_ascii=True) + "\n",
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=120,
        check=False,
    )
    lines = [line for line in completed.stdout.splitlines() if line.strip()]
    if not lines:
        raise RuntimeError(
            f"InterCode worker returned no response: {completed.stderr[-2000:]}"
        )
    response = json.loads(lines[-1])
    if response.get("type") == "error":
        raise RuntimeError(str(response.get("error")))
    if completed.returncode != 0:
        raise RuntimeError(
            f"InterCode worker exited {completed.returncode}: {completed.stderr[-2000:]}"
        )
    return response


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _terminal_records(records: Sequence[Mapping[str, Any]]) -> dict[str, dict[str, Any]]:
    terminal: dict[str, dict[str, Any]] = {}
    for record in records:
        if record.get("status") in {"finished", "failed"}:
            terminal[str(record["task_id"])] = dict(record)
    return terminal


def _artifact_key(task_id: str) -> str:
    return task_id.replace(":", "-").replace("/", "-").replace("\\", "-")
