from __future__ import annotations

import copy
import dataclasses
import hashlib
import json
import re
import subprocess
import threading
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from .alfworld_runtime import AlfWorldProgramRuntime
from .config import ExperimentConfig
from .experiments.alfworld_ptc_fewshot import ALFWORLD_PTC_FEW_SHOT_MESSAGES
from .goal_adaptation import GoalGraphAdaptation
from .graph_agent import GraphAgentHooks, extend_ptc_spec_with_graph_control
from .model import OpenAIChatModel
from .ptc import PTC_TOOL_SPEC, OriginalPTCAgent

ALFWORLD_SYSTEM_PROMPT = """You are an autonomous agent in one official ALFWorld text episode. Your only directly
callable tool is programmatic_tool_call. Its Python code runs in one persistent task shell.
Variables persist across PTC blocks and reset between tasks.

Inside the shell, call act(command) to submit an exact text action to the official AlfredTWEnv.
It returns observation, done, step, and steps_remaining. The mutable state dictionary contains the
latest values. The official generation action space is used, so no admissible-command list is
provided. Derive exact object and receptacle names from observations. Common command forms include
look, inventory, go to ..., open ..., close ..., take ... from ..., move ... to ..., examine ...,
and the documented clean/heat/cool actions. In AlfredTWEnv, place an inventory object with the
exact form move OBJECT to RECEPTACLE. Do not invent object identifiers.

Use Python loops and conditionals when several mechanically determined actions belong together,
but stop a block for a new semantic decision or a failed action. Print only the compact observation
needed by the next turn. Stop when done is true. Do not access files, environment variables, the
shell, or external networks."""

ALFWORLD_PTC_GUIDANCE = """Treat each PTC block as one semantically coherent phase rather than a wrapper around one
environment step. The graph-control fields declare intent, not proven progress. Use CONTINUE for a
new state transition, PATCH to correct failed code or a failed action sequence, and REPLAN when
changing the dependency path. Use target `task` and state an observable expected_change."""

ALFWORLD_USER_PROMPT = """Complete this ALFWorld episode using only its live observations:

<episode>{question}</episode>"""

ALFWORLD_PTC_SPEC = {
    **PTC_TOOL_SPEC,
    "function": {
        **PTC_TOOL_SPEC["function"],
        "description": (
            "Execute Python in the persistent ALFWorld task shell. "
            "The function act(command) and mutable state dictionary are available."
        ),
        "parameters": {
            **PTC_TOOL_SPEC["function"]["parameters"],
            "properties": {
                **PTC_TOOL_SPEC["function"]["parameters"]["properties"],
                "code": {
                    **PTC_TOOL_SPEC["function"]["parameters"]["properties"]["code"],
                    "description": (
                        "Python source executed in the persistent ALFWorld shell; call "
                        "act(command) for official environment steps."
                    ),
                },
            },
        },
    },
}


class EmptySearchTools:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []


@dataclass(frozen=True)
class AlfWorldRunSummary:
    selected: int
    processed: int
    successes: int
    success_rate: float
    mean_goal_condition_success_rate: float
    mean_steps: float
    episode_done: int
    execution_failure_tasks: int
    execution_failure_blocks: int
    evaluator_failures: int
    runner_failures: int
    run_signature: str

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


def _prompt_bundle(
    variant: str, *, graph_adaptation_mode: str
) -> tuple[str, tuple[dict[str, Any], ...]]:
    _validate_control_mode(graph_adaptation_mode)
    if variant != "alfworld-ptc-fewshot":
        raise ValueError(f"unsupported ALFWorld prompt variant: {variant!r}")
    prompt = ALFWORLD_SYSTEM_PROMPT
    demonstrations = copy.deepcopy(ALFWORLD_PTC_FEW_SHOT_MESSAGES)
    if graph_adaptation_mode == "generic":
        prompt += "\n\n" + ALFWORLD_PTC_GUIDANCE
    else:
        demonstrations = _without_graph_fields(demonstrations)
    return prompt, tuple(demonstrations)


def _ptc_spec(config: ExperimentConfig) -> dict[str, Any]:
    _validate_control_mode(config.runtime.graph_adaptation_mode)
    if config.runtime.graph_inspection_enabled:
        raise ValueError("ALFWorld graph inspection is not implemented")
    if config.runtime.graph_adaptation_mode == "off":
        return copy.deepcopy(ALFWORLD_PTC_SPEC)
    return extend_ptc_spec_with_graph_control(
        ALFWORLD_PTC_SPEC,
        include_input_artifacts=False,
        include_inspection=False,
        target_description="Use `task` for this ALFWorld episode.",
    )


def _without_graph_fields(
    messages: Sequence[dict[str, Any]],
) -> tuple[dict[str, Any], ...]:
    cleaned = copy.deepcopy(tuple(messages))
    for message in cleaned:
        for call in message.get("tool_calls", ()):
            function = call.get("function", {})
            if function.get("name") != "programmatic_tool_call":
                continue
            arguments = json.loads(function["arguments"])
            for key in ("action", "target", "expected_change", "inspection"):
                arguments.pop(key, None)
            function["arguments"] = json.dumps(arguments)
        if message.get("role") == "tool" and isinstance(message.get("content"), str):
            message["content"] = message["content"].split("\n\nGRAPH_DELTA ", 1)[0]
    return cleaned


def _validate_control_mode(mode: str) -> None:
    if mode not in {"off", "generic"}:
        raise ValueError("runtime.graph_adaptation_mode must be one of off, generic")


def validate_alfworld_alignment(
    config: ExperimentConfig, inspection: Mapping[str, Any]
) -> None:
    app = config.alfworld
    defaults = inspection.get("official_defaults")
    if not isinstance(defaults, Mapping):
        raise ValueError("ALFWorld inspection has no official defaults")  # noqa: TRY004
    expected = {
        "env_type": "AlfredTWEnv",
        "domain_randomization": False,
        "task_types": [1, 2, 3, 4, 5, 6],
        "random_seed": app.seed,
        "training_method": "dagger",
        "eval_batch_size": app.workers,
        "dagger_action_space": "generation",
        "max_steps": app.max_steps,
        "num_eval_games": -1,
    }
    mismatches = {
        key: {"expected": value, "actual": defaults.get(key)}
        for key, value in expected.items()
        if defaults.get(key) != value
    }
    if inspection.get("alfworld_version") != app.official_version:
        mismatches["alfworld_version"] = {
            "expected": app.official_version,
            "actual": inspection.get("alfworld_version"),
        }
    if inspection.get("split") != app.split:
        mismatches["split"] = {
            "expected": app.split,
            "actual": inspection.get("split"),
        }
    if inspection.get("adapter_batch_size") != 1:
        mismatches["adapter_batch_size"] = {
            "expected": 1,
            "actual": inspection.get("adapter_batch_size"),
        }
    if inspection.get("placement_command") != "move OBJECT to RECEPTACLE":
        mismatches["placement_command"] = {
            "expected": "move OBJECT to RECEPTACLE",
            "actual": inspection.get("placement_command"),
        }
    if config.model.temperature != 0.0:
        mismatches["agent_temperature"] = {
            "expected": 0.0,
            "actual": config.model.temperature,
        }
    if config.runtime.graph_inspection_enabled:
        mismatches["graph_inspection_enabled"] = {
            "expected": False,
            "actual": True,
        }
    if mismatches:
        raise ValueError(f"ALFWorld official alignment mismatch: {mismatches}")


def validate_alfworld_arm_pair(
    graph: ExperimentConfig, baseline: ExperimentConfig
) -> None:
    if graph.model != baseline.model:
        raise ValueError("ALFWorld arms must use the same model configuration")
    if replace(graph.runtime, graph_adaptation_mode="off") != baseline.runtime:
        raise ValueError("ALFWorld arms may only differ in graph adaptation mode")
    graph_app = replace(
        graph.alfworld,
        results_path=baseline.alfworld.results_path,
        report_path=baseline.alfworld.report_path,
        graph_dir=baseline.alfworld.graph_dir,
    )
    if graph_app != baseline.alfworld:
        raise ValueError("ALFWorld arms may only differ in output paths")


def inspect_alfworld(config: ExperimentConfig) -> dict[str, Any]:
    app = config.alfworld
    response = _worker_request(
        app.worker_command,
        {
            "type": "inspect",
            "data_root": app.data_root,
            "config_path": app.official_config_path,
            "split": app.split,
        },
    )
    inspection = {key: value for key, value in response.items() if key != "type"}
    validate_alfworld_alignment(config, inspection)
    return inspection


def run_alfworld_benchmark(
    config: ExperimentConfig,
    *,
    limit: int | None = None,
    task_ids: Sequence[str] = (),
    restart: bool = False,
    progress: Callable[[dict[str, Any]], None] | None = None,
) -> AlfWorldRunSummary:
    app = config.alfworld
    system_prompt, demonstrations = _prompt_bundle(
        app.prompt_variant,
        graph_adaptation_mode=config.runtime.graph_adaptation_mode,
    )
    inspection = inspect_alfworld(config)
    available = [str(value) for value in inspection["task_ids"]]
    selected = list(task_ids) if task_ids else available
    unknown = sorted(set(selected) - set(available))
    if unknown:
        raise ValueError(f"task IDs are not in {app.split}: {unknown}")
    if limit is not None:
        selected = selected[:limit]
    if not task_ids and limit is None and len(selected) != app.expected_tasks:
        raise ValueError(
            f"expected {app.expected_tasks} {app.split} tasks, found {len(selected)}"
        )
    signature_payload = _signature_payload(config, inspection, selected)
    signature = _sha256(signature_payload)
    app.results_path.parent.mkdir(parents=True, exist_ok=True)
    app.graph_dir.mkdir(parents=True, exist_ok=True)
    if restart and app.results_path.exists():
        app.results_path.unlink()
    records = _read_jsonl(app.results_path)
    if any(record.get("run_signature") != signature for record in records):
        raise ValueError("existing ALFWorld results use another run signature")
    terminal = _terminal_records(records)
    pending = [task_id for task_id in selected if task_id not in terminal]
    write_lock = threading.Lock()

    def append(record: dict[str, Any]) -> None:
        with write_lock, app.results_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False, default=repr) + "\n")
        if progress is not None:
            progress(record)

    def run_one(task_id: str) -> dict[str, Any]:
        append({"task_id": task_id, "status": "started", "run_signature": signature})
        runtime = AlfWorldProgramRuntime(
            worker_command=app.worker_command,
            data_root=app.data_root,
            official_config_path=app.official_config_path,
            split=app.split,
            task_id=task_id,
            seed=app.seed,
            max_steps=app.max_steps,
            timeout_seconds=config.runtime.code_timeout_seconds,
        )
        controller: GoalGraphAdaptation | None = None
        agent_result = None
        metadata: dict[str, Any] = {}
        evaluation: dict[str, Any] | None = None
        evaluator_error: str | None = None
        try:
            metadata = runtime.metadata
            question = json.dumps(
                {
                    "task": metadata["task"],
                    "initial_state": metadata["initial_state"],
                },
                ensure_ascii=False,
            )
            if config.runtime.graph_adaptation_mode == "generic":
                controller = GoalGraphAdaptation(
                    {}, {}, task=str(metadata["task"]), expose_graph_api=False
                )
                hooks = GraphAgentHooks.from_controller(controller)
                hook_kwargs = hooks.agent_kwargs()
                hook_kwargs["runtime_functions"] = ()
            else:
                hook_kwargs = {"runtime_functions": ()}
            model = OpenAIChatModel(
                config.model, config.require_api_key(config.model.api_key_env)
            )
            agent = OriginalPTCAgent(
                model=model,
                search_tools=EmptySearchTools(),  # type: ignore[arg-type]
                runtime=config.runtime,
                system_prompt=system_prompt,
                user_prompt_template=ALFWORLD_USER_PROMPT,
                ptc_tool_spec=_ptc_spec(config),
                demonstration_messages=demonstrations,
                program_runtime=runtime,
                **hook_kwargs,
            )
            agent_result = agent.run(question)
            try:
                evaluation = runtime.evaluate()
            except Exception as exc:  # noqa: BLE001 - evaluator failure is recorded per task.
                evaluator_error = f"{type(exc).__name__}: {exc}"
            if controller is not None:
                controller.finish(answered=bool((evaluation or {}).get("success")))
            graph_path: Path | None = None
            if controller is not None:
                graph_path = app.graph_dir / f"{_artifact_key(task_id)}.json"
                graph_path.write_text(
                    json.dumps(
                        controller.graph_artifact(),
                        ensure_ascii=False,
                        indent=2,
                        default=repr,
                    ),
                    encoding="utf-8",
                )
            record = {
                "task_id": task_id,
                "status": "finished",
                "run_signature": signature,
                "agent": agent_result.to_dict(),
                "episode_done": runtime.task_completed,
                "execution_failures": sum(
                    not block.success for block in agent_result.blocks
                ),
                "official_evaluation": evaluation,
                "evaluator_error": evaluator_error,
                "alfworld": metadata,
                "graph_path": str(graph_path) if graph_path is not None else None,
                "graph_telemetry": controller.telemetry()
                if controller is not None
                else None,
            }
        except Exception as exc:  # noqa: BLE001 - runner failure remains in the denominator.
            record = {
                "task_id": task_id,
                "status": "failed",
                "run_signature": signature,
                "error": f"{type(exc).__name__}: {exc}",
                "agent": agent_result.to_dict() if agent_result is not None else None,
                "episode_done": runtime.task_completed,
                "execution_failures": (
                    sum(not block.success for block in agent_result.blocks)
                    if agent_result is not None
                    else 0
                ),
                "official_evaluation": evaluation,
                "evaluator_error": evaluator_error,
                "alfworld": metadata or None,
                "graph_telemetry": controller.telemetry()
                if controller is not None
                else None,
            }
        finally:
            try:
                runtime.close()
            except Exception as exc:  # noqa: BLE001 - close is best-effort and recorded.
                record["status"] = "failed"
                record["close_error"] = f"{type(exc).__name__}: {exc}"
        final_runtime = runtime.telemetry()
        if final_runtime.get("termination_confirmed") is False:
            record["status"] = "failed"
            record.setdefault(
                "close_error",
                final_runtime.get("close_error")
                or "worker termination was not confirmed",
            )
        record["runtime_final"] = final_runtime
        append(record)
        return record

    if app.workers == 1:
        for task_id in pending:
            run_one(task_id)
    else:
        with ThreadPoolExecutor(max_workers=app.workers) as executor:
            futures = [executor.submit(run_one, task_id) for task_id in pending]
            for future in as_completed(futures):
                future.result()

    finished = _terminal_records(_read_jsonl(app.results_path))
    selected_records = [finished[value] for value in selected if value in finished]
    summary = _summarize(selected, selected_records, signature)
    app.report_path.parent.mkdir(parents=True, exist_ok=True)
    report = {
        "summary": summary.to_dict(),
        "run_signature_payload": signature_payload,
        "resolved_config": _resolved_config(config),
        "resolved_config_sha256": _sha256(_resolved_config(config)),
        "tasks": selected_records,
    }
    app.report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, default=repr),
        encoding="utf-8",
    )
    return summary


def evaluate_alfworld_benchmark(config: ExperimentConfig) -> dict[str, Any]:
    app = config.alfworld
    if not app.report_path.exists():
        raise ValueError("ALFWorld run report does not exist")
    report = json.loads(app.report_path.read_text(encoding="utf-8"))
    payload = report.get("run_signature_payload")
    if not isinstance(payload, dict):
        raise ValueError("ALFWorld run report has no signature payload")  # noqa: TRY004
    task_ids = [str(value) for value in payload.get("task_ids", ())]
    inspection = inspect_alfworld(config)
    expected_payload = _signature_payload(config, inspection, task_ids)
    signature = _sha256(expected_payload)
    if (
        payload != expected_payload
        or (report.get("summary") or {}).get("run_signature") != signature
    ):
        raise ValueError(
            "saved ALFWorld run signature does not match current configuration"
        )
    terminal = _terminal_records(_read_jsonl(app.results_path))
    if set(terminal) != set(task_ids):
        raise ValueError("ALFWorld results do not match the saved run task IDs")
    summary = _summarize(task_ids, [terminal[value] for value in task_ids], signature)
    report["official_evaluation"] = summary.to_dict()
    app.report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, default=repr),
        encoding="utf-8",
    )
    return summary.to_dict()


def _summarize(
    selected: Sequence[str], records: Sequence[Mapping[str, Any]], signature: str
) -> AlfWorldRunSummary:
    evaluations = [
        item.get("official_evaluation")
        for item in records
        if item.get("status") == "finished"
        and isinstance(item.get("official_evaluation"), Mapping)
        and not item.get("evaluator_error")
    ]
    successes = sum(bool(item.get("success")) for item in evaluations)
    return AlfWorldRunSummary(
        selected=len(selected),
        processed=len(records),
        successes=successes,
        success_rate=successes / len(selected) if selected else 0.0,
        mean_goal_condition_success_rate=(
            sum(
                float(item.get("goal_condition_success_rate", 0.0))
                for item in evaluations
            )
            / len(selected)
            if selected
            else 0.0
        ),
        mean_steps=(
            sum(float(item.get("steps", 0.0)) for item in evaluations) / len(selected)
            if selected
            else 0.0
        ),
        episode_done=sum(bool(item.get("episode_done")) for item in records),
        execution_failure_tasks=sum(
            bool(item.get("execution_failures")) for item in records
        ),
        execution_failure_blocks=sum(
            int(item.get("execution_failures", 0)) for item in records
        ),
        evaluator_failures=sum(bool(item.get("evaluator_error")) for item in records),
        runner_failures=sum(item.get("status") == "failed" for item in records),
        run_signature=signature,
    )


def _terminal_records(
    records: Sequence[Mapping[str, Any]],
) -> dict[str, dict[str, Any]]:
    return {
        str(item["task_id"]): dict(item)
        for item in records
        if item.get("status") in {"finished", "failed"}
    }


def _artifact_key(task_id: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9_-]+", "-", task_id).strip("-")[:80] or "task"
    return f"{slug}-{hashlib.sha256(task_id.encode()).hexdigest()[:12]}"


def _worker_request(
    command: Sequence[str], payload: Mapping[str, Any], *, timeout: float = 120
) -> dict[str, Any]:
    if not command:
        raise ValueError("[alfworld].worker_command is required")
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
        detail = completed.stderr[-2000:] or completed.stdout[-2000:]
        raise RuntimeError(
            f"ALFWorld worker request failed ({completed.returncode}): {detail}"
        )
    response = json.loads(lines[-1])
    if response.get("type") == "error":
        raise RuntimeError(str(response.get("error")))
    return response


def _signature_payload(
    config: ExperimentConfig, inspection: Mapping[str, Any], task_ids: Sequence[str]
) -> dict[str, Any]:
    model = dataclasses.asdict(config.model)
    runtime = dataclasses.asdict(config.runtime)
    app = {
        "data_root": config.alfworld.data_root,
        "official_config_path": config.alfworld.official_config_path,
        "split": config.alfworld.split,
        "worker_command": list(config.alfworld.worker_command),
        "workers": config.alfworld.workers,
        "seed": config.alfworld.seed,
        "max_steps": config.alfworld.max_steps,
        "prompt_variant": config.alfworld.prompt_variant,
        "official_version": config.alfworld.official_version,
    }
    prompt, demonstrations = _prompt_bundle(
        config.alfworld.prompt_variant,
        graph_adaptation_mode=config.runtime.graph_adaptation_mode,
    )
    environment = {key: value for key, value in inspection.items() if key != "task_ids"}
    return {
        "schema_version": 1,
        "benchmark": "alfworld",
        "model": model,
        "runtime": runtime,
        "alfworld": app,
        "behavior_config_sha256": _sha256(
            {"model": model, "runtime": runtime, "alfworld": app}
        ),
        "prompt": {
            "variant": config.alfworld.prompt_variant,
            "system_prompt_sha256": _sha256(prompt),
            "demonstrations_sha256": _sha256(demonstrations),
            "tool_spec_sha256": _sha256(_ptc_spec(config)),
        },
        "environment": environment,
        "task_ids": list(task_ids),
        "graphptc_commit": _git_commit(),
        "graphptc_git_dirty": _git_dirty(),
        "graphptc_source_hash": _source_hash(),
    }


def _resolved_config(config: ExperimentConfig) -> dict[str, Any]:
    app = dataclasses.asdict(config.alfworld)
    for key in ("results_path", "report_path", "graph_dir"):
        app[key] = str(app[key])
    app["worker_command"] = list(app["worker_command"])
    return {
        "model": dataclasses.asdict(config.model),
        "runtime": dataclasses.asdict(config.runtime),
        "alfworld": app,
    }


def _git_commit() -> str:
    return subprocess.run(
        ("git", "rev-parse", "HEAD"), capture_output=True, text=True, check=True
    ).stdout.strip()


def _git_dirty() -> bool:
    return bool(
        subprocess.run(
            ("git", "status", "--porcelain", "--untracked-files=all"),
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    )


def _source_hash() -> str:
    digest = hashlib.sha256()
    root = Path(__file__).resolve().parent
    for path in sorted(root.rglob("*.py")):
        digest.update(path.relative_to(root).as_posix().encode())
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _sha256(value: Any) -> str:
    payload = json.dumps(
        value, ensure_ascii=False, sort_keys=True, default=str, separators=(",", ":")
    )
    return hashlib.sha256(payload.encode()).hexdigest()


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line
    ]
