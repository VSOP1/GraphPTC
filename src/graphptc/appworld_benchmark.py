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
from .experiments.appworld_ptc_fewshot import APPWORLD_PTC_FEW_SHOT_MESSAGES
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

APPWORLD_PTC_SEMANTIC_PROMPT = APPWORLD_SYSTEM_PROMPT + """

Follow AppWorld's general operating contract. Work autonomously, never invent API names, argument
names, IDs, credentials, or other values, and avoid changes beyond the instruction. Do not ask the
supervisor to perform a step. Before first using an API, read its live API documentation; it
describes both parameters and response shapes. Resolve the supervisor's personal/account details
through Supervisor APIs and references to other people through phone contacts. Obtain the current
date or time from the environment, never from model memory. A reference to the file system means the
file-system app, not OS access; assume the single default time zone unless the instruction says
otherwise. Store every API result needed later in a Python variable.
Only printed stdout is visible to the next model turn; an unprinted return value produces only a generic
success message. Pass authentication values such as a returned access token explicitly whenever the
API documentation requires them; successful login does not create implicit authentication state.
When an API is paginated, use its documented pagination parameters and process all relevant pages.
Submit an answer as the minimal direct value requested, using numeric values for counts.

Treat each PTC block as one semantically coherent phase, not as a wrapper around one API call and not
as an attempt to solve every uncertain step in one monolithic program. Put mechanically foreseeable
API calls and Python loops, pagination, filtering, joins, and aggregation in the same block. Print a
compact derived result, not raw collections or secrets. Return to the model for a new semantic
decision, an execution failure that needs repair, or task completion. There is no fixed number of
blocks or calls.

The graph-control fields declare intent rather than prove progress. Use `CONTINUE` for a new task
effect, `PATCH` when correcting a failed block, and `REPLAN` when changing the dependency path.
`target` names the affected graph node and `expected_change` states the observable delta. Select
`INSPECT` only when the tool schema provides an executable inspection request; a label by itself is
not a graph query."""

APPWORLD_INSPECTION_GUIDANCE = """When `INSPECT` is available, include an `inspection` request in the
same programmatic_tool_call. Use `frontier` for the current bounded dependency frontier or `trace`
with a graph `node_id` for its bounded neighborhood. The host executes this read-only query after
the submitted code is projected, and `GRAPH_DELTA.inspection_result` is available only to the next
model turn. The query does not execute AppWorld APIs or change AppWorld state."""

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
        "parameters": {
            **PTC_TOOL_SPEC["function"]["parameters"],
            "properties": {
                **PTC_TOOL_SPEC["function"]["parameters"]["properties"],
                "code": {
                    **PTC_TOOL_SPEC["function"]["parameters"]["properties"]["code"],
                    "description": (
                        "Exact Python source executed directly in the persistent AppWorld shell. "
                        "The globals `apis` and `requester` are available."
                    ),
                },
            },
        },
    },
}


def _appworld_prompt_bundle(
    variant: str,
    *,
    graph_inspection_enabled: bool = False,
) -> tuple[str, tuple[dict[str, Any], ...]]:
    if variant == "appworld-general":
        prompt, demonstrations = APPWORLD_SYSTEM_PROMPT, ()
    elif variant == "appworld-ptc-semantics":
        prompt, demonstrations = APPWORLD_PTC_SEMANTIC_PROMPT, ()
    elif variant == "appworld-ptc-fewshot":
        prompt, demonstrations = APPWORLD_PTC_SEMANTIC_PROMPT, APPWORLD_PTC_FEW_SHOT_MESSAGES
    else:
        raise ValueError(f"unsupported AppWorld prompt variant: {variant!r}")
    if graph_inspection_enabled:
        prompt += "\n\n" + APPWORLD_INSPECTION_GUIDANCE
    return prompt, demonstrations


def _appworld_ptc_spec(config: ExperimentConfig) -> dict[str, Any]:
    return extend_ptc_spec_with_graph_control(
        APPWORLD_PTC_SPEC,
        include_input_artifacts=False,
        include_inspection=config.runtime.graph_inspection_enabled,
        target_description="Use `task` for this AppWorld episode.",
    )


class EmptySearchTools:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []


@dataclass(frozen=True)
class AppWorldRunSummary:
    selected: int
    processed: int
    task_completed: int
    official_failures: int
    execution_failure_tasks: int
    execution_failure_blocks: int
    incomplete_tasks: int
    evaluator_failures: int
    runner_failures: int
    inspection_declared: int
    inspection_succeeded: int
    inspection_failed: int
    inspection_results_returned: int
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
    system_prompt, demonstration_messages = _appworld_prompt_bundle(
        app.prompt_variant,
        graph_inspection_enabled=config.runtime.graph_inspection_enabled,
    )
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
    seen = _terminal_task_ids(records)
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
        agent_result = None
        instruction: str | None = None
        runtime_metadata: dict[str, Any] = {}
        evaluator_error: str | None = None
        evaluation: dict[str, Any] | None = None
        try:
            runtime_metadata = runtime.metadata
            instruction = str(runtime_metadata["instruction"])
            controller = GoalGraphAdaptation(
                {},
                {},
                task=instruction,
                expose_graph_api=False,
                host_inspection_enabled=config.runtime.graph_inspection_enabled,
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
                system_prompt=system_prompt,
                user_prompt_template=APPWORLD_USER_PROMPT,
                ptc_tool_spec=_appworld_ptc_spec(config),
                demonstration_messages=demonstration_messages,
                program_runtime=runtime,
                **hook_kwargs,
            )
            agent_result = agent.run(instruction)
            controller.finish(answered=runtime.task_completed)
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
                "appworld": runtime_metadata,
                "graph_path": str(graph_path),
                "graph_telemetry": controller.telemetry(),
            }
        except Exception as exc:
            record = {
                "task_id": task_id,
                "status": "failed",
                "run_signature": signature,
                "error": f"{type(exc).__name__}: {exc}",
                "instruction": instruction,
                "agent": agent_result.to_dict() if agent_result is not None else None,
                "task_completed": runtime.task_completed,
                "execution_failures": (
                    sum(not block.success for block in agent_result.blocks)
                    if agent_result is not None
                    else 0
                ),
                "official_evaluation": evaluation,
                "evaluator_error": evaluator_error,
                "appworld": runtime_metadata or None,
                "graph_telemetry": controller.telemetry() if controller is not None else None,
            }
        finally:
            try:
                runtime.close()
            except Exception as exc:
                record["status"] = "failed"
                record["close_error"] = f"{type(exc).__name__}: {exc}"
        final_runtime = runtime.telemetry()
        if final_runtime.get("termination_confirmed") is False:
            record["status"] = "failed"
            record.setdefault(
                "close_error",
                final_runtime.get("close_error") or "worker termination was not confirmed",
            )
        record["runtime_final"] = final_runtime
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
                "resolved_config": _resolved_config(config),
                "resolved_config_sha256": _sha256(_resolved_config(config)),
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
    report_path = config.appworld.report_path
    if not report_path.exists():
        raise ValueError("AppWorld run report does not exist")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    saved_payload = report.get("run_signature_payload")
    if not isinstance(saved_payload, dict):
        raise ValueError("AppWorld run report has no signature payload")
    task_ids = [str(value) for value in saved_payload.get("task_ids", ())]
    if not task_ids:
        raise ValueError("no AppWorld task IDs in the saved run signature")
    inspection = inspect_appworld(config)
    expected_payload = _signature_payload(config, inspection, task_ids)
    expected_signature = _sha256(expected_payload)
    saved_signature = str((report.get("summary") or {}).get("run_signature", ""))
    if saved_signature != expected_signature or saved_payload != expected_payload:
        raise ValueError("saved AppWorld run signature does not match current configuration")

    terminal = {
        str(record.get("task_id")): record
        for record in _read_jsonl(config.appworld.results_path)
        if record.get("status") in {"finished", "failed"}
    }
    if set(terminal) != set(task_ids):
        raise ValueError("AppWorld results do not match the saved run task IDs")
    if any(record.get("status") != "finished" for record in terminal.values()):
        raise ValueError("AppWorld runner failures must be resolved before evaluation")
    if any(record.get("run_signature") != expected_signature for record in terminal.values()):
        raise ValueError("AppWorld results contain another run signature")

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
    if response.get("appworld_version") != inspection.get("appworld_version"):
        raise ValueError("AppWorld evaluator code version changed during evaluation")
    if response.get("data_version") != inspection.get("data_version"):
        raise ValueError("AppWorld evaluator data version changed during evaluation")
    report["official_evaluation"] = evaluation
    report["official_evaluation_provenance"] = {
        "run_signature": expected_signature,
        "appworld_version": response.get("appworld_version"),
        "data_version": response.get("data_version"),
        "task_ids": task_ids,
    }
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
        processed=len(records),
        task_completed=sum(bool(record.get("task_completed")) for record in records),
        official_failures=sum(
            not bool((record.get("official_evaluation") or {}).get("success"))
            for record in records
            if record.get("status") == "finished"
            and record.get("official_evaluation") is not None
            and record.get("evaluator_error") is None
        ),
        execution_failure_tasks=sum(bool(record.get("execution_failures")) for record in records),
        execution_failure_blocks=sum(int(record.get("execution_failures") or 0) for record in records),
        incomplete_tasks=sum(not bool(record.get("task_completed")) for record in records),
        evaluator_failures=sum(bool(record.get("evaluator_error")) for record in records),
        runner_failures=sum(record.get("status") == "failed" for record in records),
        inspection_declared=_inspection_total(records, "declared"),
        inspection_succeeded=_inspection_total(records, "succeeded"),
        inspection_failed=_inspection_total(records, "failed"),
        inspection_results_returned=_inspection_total(records, "results_returned"),
        run_signature=signature,
    )


def _terminal_task_ids(records: Sequence[dict[str, Any]]) -> set[str]:
    return {
        str(record.get("task_id"))
        for record in records
        if record.get("status") in {"finished", "failed"}
    }


def _inspection_total(records: Sequence[dict[str, Any]], key: str) -> int:
    return sum(
        int(((record.get("graph_telemetry") or {}).get("inspection") or {}).get(key, 0))
        for record in records
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
    model = dataclasses.asdict(config.model)
    runtime = dataclasses.asdict(config.runtime)
    appworld = {
        "root": config.appworld.root,
        "dataset_name": config.appworld.dataset_name,
        "experiment_name": config.appworld.experiment_name,
        "worker_command": list(config.appworld.worker_command),
        "workers": config.appworld.workers,
        "prompt_variant": config.appworld.prompt_variant,
    }
    system_prompt, demonstrations = _appworld_prompt_bundle(
        config.appworld.prompt_variant,
        graph_inspection_enabled=config.runtime.graph_inspection_enabled,
    )
    return {
        "schema_version": 2,
        "benchmark": "appworld",
        "model": model,
        "runtime": runtime,
        "appworld": appworld,
        "behavior_config_sha256": _sha256(
            {"model": model, "runtime": runtime, "appworld": appworld}
        ),
        "prompt": {
            "variant": config.appworld.prompt_variant,
            "system_prompt_sha256": _sha256(system_prompt),
            "demonstrations_sha256": _sha256(demonstrations),
            "tool_spec_sha256": _sha256(_appworld_ptc_spec(config)),
        },
        "environment": {key: value for key, value in inspection.items() if key != "task_ids"},
        "task_ids": task_ids,
        "graphptc_commit": _git_commit(),
        "graphptc_git_dirty": _git_dirty(),
        "graphptc_source_hash": _source_hash(),
    }


def _git_commit() -> str:
    completed = subprocess.run(
        ("git", "rev-parse", "HEAD"), capture_output=True, text=True, check=True
    )
    return completed.stdout.strip()


def _git_dirty() -> bool:
    completed = subprocess.run(
        ("git", "status", "--porcelain", "--untracked-files=all"),
        capture_output=True,
        text=True,
        check=True,
    )
    return bool(completed.stdout.strip())


def _source_hash() -> str:
    digest = hashlib.sha256()
    source_root = Path(__file__).resolve().parent
    for path in sorted(source_root.rglob("*.py")):
        digest.update(path.relative_to(source_root).as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _resolved_config(config: ExperimentConfig) -> dict[str, Any]:
    appworld = dataclasses.asdict(config.appworld)
    for key in ("results_path", "report_path", "graph_dir"):
        appworld[key] = str(appworld[key])
    appworld["worker_command"] = list(appworld["worker_command"])
    return {
        "model": dataclasses.asdict(config.model),
        "runtime": dataclasses.asdict(config.runtime),
        "appworld": appworld,
    }


def _sha256(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
