from __future__ import annotations

import dataclasses
import hashlib
import json
import subprocess
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Sequence

from .appworld_runtime import AppWorldProgramRuntime
from .config import ExperimentConfig
from .goal_adaptation import GoalGraphAdaptation
from .graph_agent import GraphAgentHooks, extend_ptc_spec_with_graph_control
from .model import OpenAIChatModel
from .ptc import OriginalPTCAgent, PTC_TOOL_SPEC


APPWORLD_SYSTEM_PROMPT = """You are an autonomous AppWorld agent. Your only directly callable tool is
programmatic_tool_call. Its code is executed directly in one persistent AppWorld Python shell for
this task; variables and imports persist across blocks and are reset for the next task.

Inside the shell, `apis` exposes AppWorld APIs and `requester` is available. Discover capabilities
on demand with `apis.api_docs.show_app_descriptions()`,
`apis.api_docs.show_api_descriptions(app_name=...)`, and
`apis.api_docs.show_api_doc(app_name=..., api_name=...)`. Use the Supervisor APIs when account or
profile information is needed. Choose APIs, parameters, and programs yourself from the instruction
and live documentation. Print only compact information needed for the next decision.

When the task is actually done, call `apis.supervisor.complete_task()` inside the program. Pass its
optional `answer` only for answer-seeking instructions. Do not call save_state or load_state. Do not
access files, environment variables, the shell, or external networks. AppWorld's safety rules and
execution limits apply. API/runtime failures are observations to diagnose and repair; do not claim
completion until the required state change or answer has been produced."""

APPWORLD_USER_PROMPT = """Complete this AppWorld instruction on behalf of the supervisor:

<instruction>{question}</instruction>"""

APPWORLD_PTC_SPEC = {
    **PTC_TOOL_SPEC,
    "function": {
        **PTC_TOOL_SPEC["function"],
        "description": (
            "Execute this Python source directly in the task's persistent AppWorld shell. "
            "The globals `apis` and `requester` are available."
        ),
    },
}


class EmptySearchTools:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []


@dataclass(frozen=True)
class AppWorldRunSummary:
    selected: int
    completed: int
    official_failures: int
    execution_failure_tasks: int
    incomplete_tasks: int
    evaluator_failures: int
    run_signature: str

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


def inspect_appworld(config: ExperimentConfig) -> dict[str, Any]:
    response = _worker_request(
        config.appworld.worker_command,
        {
            "type": "inspect",
            "root": config.appworld.root,
            "dataset_name": config.appworld.dataset_name,
        },
    )
    return {key: value for key, value in response.items() if key != "type"}


def run_appworld_benchmark(
    config: ExperimentConfig,
    *,
    limit: int | None = None,
    task_ids: Sequence[str] = (),
    restart: bool = False,
    progress: Callable[[dict[str, Any]], None] | None = None,
) -> AppWorldRunSummary:
    app = config.appworld
    inspection = inspect_appworld(config)
    available = list(inspection["task_ids"])
    selected = list(task_ids) if task_ids else available
    unknown = sorted(set(selected) - set(available))
    if unknown:
        raise ValueError(f"task IDs are not in {app.dataset_name}: {unknown}")
    if limit is not None:
        selected = selected[:limit]
    if not task_ids and limit is None and len(selected) != app.expected_tasks:
        raise ValueError(
            f"expected {app.expected_tasks} {app.dataset_name} tasks, found {len(selected)}"
        )
    signature_payload = _signature_payload(config, inspection, selected)
    signature = _sha256(signature_payload)
    output_path = app.results_path
    output_path.parent.mkdir(parents=True, exist_ok=True)
    app.graph_dir.mkdir(parents=True, exist_ok=True)
    if restart and output_path.exists():
        output_path.unlink()
    records = _read_jsonl(output_path)
    mismatched = {
        record.get("run_signature")
        for record in records
        if record.get("run_signature") != signature
    }
    if mismatched:
        raise ValueError("existing AppWorld results use another run signature")
    seen = {str(record.get("task_id")) for record in records}
    pending = [task_id for task_id in selected if task_id not in seen]
    write_lock = threading.Lock()

    def append(record: dict[str, Any]) -> None:
        with write_lock:
            with output_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, ensure_ascii=False, default=repr) + "\n")
        if progress is not None:
            progress(record)

    def run_one(task_id: str) -> dict[str, Any]:
        append({"task_id": task_id, "status": "started", "run_signature": signature})
        runtime = AppWorldProgramRuntime(
            worker_command=app.worker_command,
            root=app.root,
            task_id=task_id,
            experiment_name=app.experiment_name,
            timeout_seconds=config.runtime.code_timeout_seconds,
        )
        controller: GoalGraphAdaptation | None = None
        evaluator_error: str | None = None
        evaluation: dict[str, Any] | None = None
        try:
            instruction = str(runtime.metadata["instruction"])
            controller = GoalGraphAdaptation(
                {}, {}, task=instruction, expose_graph_api=False
            )
            hooks = GraphAgentHooks.from_controller(controller)
            hook_kwargs = hooks.agent_kwargs()
            hook_kwargs["runtime_functions"] = ()
            model = OpenAIChatModel(
                config.model,
                config.require_api_key(config.model.api_key_env),
            )
            agent = OriginalPTCAgent(
                model=model,
                search_tools=EmptySearchTools(),  # type: ignore[arg-type]
                runtime=config.runtime,
                system_prompt=APPWORLD_SYSTEM_PROMPT,
                user_prompt_template=APPWORLD_USER_PROMPT,
                ptc_tool_spec=extend_ptc_spec_with_graph_control(
                    APPWORLD_PTC_SPEC,
                    include_input_artifacts=False,
                    target_description="Use `task` for this AppWorld episode.",
                ),
                program_runtime=runtime,
                **hook_kwargs,
            )
            agent_result = agent.run(instruction)
            try:
                evaluation = runtime.evaluate()
            except Exception as exc:
                evaluator_error = f"{type(exc).__name__}: {exc}"
            graph_path = app.graph_dir / f"{task_id}.json"
            graph_path.write_text(
                json.dumps(controller.graph_artifact(), ensure_ascii=False, indent=2, default=repr),
                encoding="utf-8",
            )
            record = {
                "task_id": task_id,
                "status": "finished",
                "run_signature": signature,
                "instruction": instruction,
                "agent": agent_result.to_dict(),
                "task_completed": runtime.task_completed,
                "execution_failures": sum(not block.success for block in agent_result.blocks),
                "official_evaluation": evaluation,
                "evaluator_error": evaluator_error,
                "appworld": runtime.metadata,
                "graph_path": str(graph_path),
                "graph_telemetry": controller.telemetry(),
            }
        except Exception as exc:
            record = {
                "task_id": task_id,
                "status": "failed",
                "run_signature": signature,
                "error": f"{type(exc).__name__}: {exc}",
                "task_completed": runtime.task_completed,
                "execution_failures": 0,
                "official_evaluation": evaluation,
                "evaluator_error": evaluator_error,
            }
        finally:
            runtime.close()
        append(record)
        return record

    if app.workers == 1:
        for task_id in pending:
            run_one(task_id)
    else:
        with ThreadPoolExecutor(max_workers=app.workers) as executor:
            futures = {executor.submit(run_one, task_id): task_id for task_id in pending}
            for future in as_completed(futures):
                future.result()

    finished = {
        record["task_id"]: record
        for record in _read_jsonl(output_path)
        if record.get("status") in {"finished", "failed"}
    }
    selected_records = [finished[value] for value in selected if value in finished]
    summary = _summarize(selected, selected_records, signature)
    app.report_path.parent.mkdir(parents=True, exist_ok=True)
    app.report_path.write_text(
        json.dumps(
            {
                "summary": summary.to_dict(),
                "run_signature_payload": signature_payload,
                "tasks": selected_records,
            },
            ensure_ascii=False,
            indent=2,
            default=repr,
        ),
        encoding="utf-8",
        errors="replace",
    )
    return summary


def evaluate_appworld_benchmark(config: ExperimentConfig) -> dict[str, Any]:
    records = [
        record
        for record in _read_jsonl(config.appworld.results_path)
        if record.get("status") == "finished"
    ]
    task_ids = [str(record["task_id"]) for record in records]
    if not task_ids:
        raise ValueError("no finished AppWorld tasks to evaluate")
    response = _worker_request(
        config.appworld.worker_command,
        {
            "type": "evaluate_tasks",
            "root": config.appworld.root,
            "task_ids": task_ids,
            "experiment_name": config.appworld.experiment_name,
        },
        timeout=300,
    )
    evaluation = response["evaluation"]
    report_path = config.appworld.report_path
    report = (
        json.loads(report_path.read_text(encoding="utf-8"))
        if report_path.exists()
        else {}
    )
    report["official_evaluation"] = evaluation
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, default=repr),
        encoding="utf-8",
    )
    return evaluation


def _summarize(
    selected: list[str], records: list[dict[str, Any]], signature: str
) -> AppWorldRunSummary:
    return AppWorldRunSummary(
        selected=len(selected),
        completed=len(records),
        official_failures=sum(
            not bool((record.get("official_evaluation") or {}).get("success"))
            for record in records
            if record.get("evaluator_error") is None
        ),
        execution_failure_tasks=sum(bool(record.get("execution_failures")) for record in records),
        incomplete_tasks=sum(not bool(record.get("task_completed")) for record in records),
        evaluator_failures=sum(bool(record.get("evaluator_error")) for record in records),
        run_signature=signature,
    )


def _worker_request(
    command: Sequence[str], payload: dict[str, Any], *, timeout: float = 60
) -> dict[str, Any]:
    if not command:
        raise ValueError("[appworld].worker_command is required")
    completed = subprocess.run(
        tuple(command),
        input=json.dumps(payload, ensure_ascii=True) + "\n",
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=timeout,
        check=False,
    )
    lines = [line for line in completed.stdout.splitlines() if line.strip()]
    if completed.returncode != 0 or not lines:
        raise RuntimeError(
            f"AppWorld worker request failed ({completed.returncode}): {completed.stderr[-2000:]}"
        )
    response = json.loads(lines[-1])
    if response.get("type") == "error":
        raise RuntimeError(str(response.get("error")))
    return response


def _signature_payload(
    config: ExperimentConfig, inspection: dict[str, Any], task_ids: list[str]
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "benchmark": "appworld",
        "model": dataclasses.asdict(config.model),
        "runtime": dataclasses.asdict(config.runtime),
        "appworld": {
            "root": config.appworld.root,
            "dataset_name": config.appworld.dataset_name,
            "experiment_name": config.appworld.experiment_name,
            "worker_command": list(config.appworld.worker_command),
            "workers": config.appworld.workers,
            "prompt_variant": config.appworld.prompt_variant,
        },
        "environment": {key: value for key, value in inspection.items() if key != "task_ids"},
        "task_ids": task_ids,
        "graphptc_commit": _git_commit(),
        "graphptc_source_hash": _source_hash(),
    }


def _git_commit() -> str:
    completed = subprocess.run(
        ("git", "rev-parse", "HEAD"), capture_output=True, text=True, check=True
    )
    return completed.stdout.strip()


def _source_hash() -> str:
    digest = hashlib.sha256()
    source_root = Path(__file__).resolve().parent
    for path in sorted(source_root.rglob("*.py")):
        digest.update(path.relative_to(source_root).as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _sha256(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
