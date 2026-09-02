from __future__ import annotations

import copy
import dataclasses
import json
import os
import subprocess
import time
from collections import defaultdict
from collections.abc import Mapping, Sequence
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .runtime import APIFlowProgramRuntime, _redact
from ...config import ExperimentConfig
from ...graph.adaptation import GoalGraphAdaptation
from ...graph.hooks import GraphAgentHooks, extend_ptc_spec_with_graph_control
from ...graph.diagnostics import graph_delta_sequence
from ...model import OpenAIChatModel
from ...agents.original_ptc import OriginalPTCAgent, PTC_TOOL_SPEC
from ...graph.tool_effects import ToolEffectContract


APIFLOW_TOOLS = ("read", "write", "edit", "search", "execute", "clarify", "report_blocked")

PTC_OVERLAY = """

# Programmatic tool calling

Your only directly callable model tool is `programmatic_tool_call`. Its Python source runs in one
persistent namespace for this task. The seven APIFlow tools described above are Python globals with
the same keyword arguments as the official tools. JSON tool results are returned as Python dicts or
lists; textual confirmations remain strings.

Inside PTC code, call every wrapper with keyword arguments even where the official synopsis is
compact: for example `read(entity_ref={...})`, `write(entity_ref={...}, content={...})`, and
`execute(request={...})`. Do not pass wrapper arguments positionally or introspect wrapper objects.

Use PTC blocks to organize mechanically related API operations, filtering, joins, and state checks.
Only printed stdout is visible to your next model turn, so print compact decision-relevant values.
Do not access files, environment variables, the shell, or the network except through the seven
official wrappers. Stop with a concise final answer in exactly the format requested by the task.
"""

USER_PROMPT = """Complete this APIFlow-Bench task autonomously:

<task>{question}</task>"""

APIFLOW_PTC_SPEC = {
    **PTC_TOOL_SPEC,
    "function": {
        **PTC_TOOL_SPEC["function"],
        "description": (
            "Execute one coherent Python program directly in this task's persistent namespace. "
            "The seven official APIFlow tools are globals."
        ),
    },
}


def _demo_messages() -> tuple[dict[str, Any], ...]:
    return (
        {
            "role": "user",
            "content": "PTC organization example only: inventory a workspace before deciding what to change.",
        },
        {
            "role": "assistant",
            "content": "I will inspect the available entities and variables in one compact block.",
            "tool_calls": [
                {
                    "id": "apiflow_demo_1",
                    "type": "function",
                    "function": {
                        "name": "programmatic_tool_call",
                        "arguments": json.dumps(
                            {
                                "code": (
                                    "items = search(query='', kind=None)\n"
                                    "print({'count': len(items), 'sample': items[:8]})"
                                )
                            }
                        ),
                    },
                }
            ],
        },
        {
            "role": "tool",
            "tool_call_id": "apiflow_demo_1",
            "content": "{'count': 3, 'sample': ['spec:api', {'kind': 'variable', 'id': 'credential', 'scope': 'vault'}, 'request:draft']}",
        },
        {
            "role": "assistant",
            "content": "The inventory is complete; I would now read the relevant spec and credential before acting.",
        },
    )


class _EmptySearchTools:
    calls: list[dict[str, Any]] = []


@dataclass(frozen=True)
class APIFlowRunSummary:
    selected: int
    processed: int
    passed: int
    failed: int
    runner_failures: int
    tool_calls: int
    input_tokens: int
    output_tokens: int

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


def inspect_apiflow(config: ExperimentConfig) -> dict[str, Any]:
    _validate_config(config)
    frozen_manifest = _load_manifest(config)
    inspection = _official_request(
        config.apiflow.official_worker_command,
        {"type": "inspect", "root": config.apiflow.root},
        timeout=120,
    )
    tasks = list(inspection.get("tasks") or [])
    if len(tasks) != config.apiflow.expected_tasks:
        raise ValueError(
            f"expected {config.apiflow.expected_tasks} APIFlow tasks, found {len(tasks)}"
        )
    if len({task["task_id"] for task in tasks}) != len(tasks):
        raise ValueError("APIFlow task IDs are not unique")
    python = config.apiflow.official_worker_command[0]
    command = [
        python,
        str(Path(config.apiflow.root) / "scripts" / "bank_sha256.py"),
        str(config.apiflow.bank_path),
    ]
    env = {**os.environ, "PYTHONUTF8": "1"}
    completed = subprocess.run(
        command, capture_output=True, text=True, encoding="utf-8", env=env, timeout=120
    )
    bank_sha256 = completed.stdout.strip().splitlines()[-1] if completed.returncode == 0 else ""
    expected_bank_sha256 = str(frozen_manifest["bank_sha256"])
    if bank_sha256 != expected_bank_sha256:
        raise ValueError(
            f"APIFlow bank hash mismatch: expected {expected_bank_sha256}, got {bank_sha256}"
        )
    manifest = {
        "schema_version": 1,
        "benchmark": "APIFlow-Bench",
        "release": "1.0",
        "bank_sha256": bank_sha256,
        "expected_tasks": config.apiflow.expected_tasks,
        "epochs": config.apiflow.epochs,
        "environment": {
            "python": inspection.get("python"),
            "apiflow_bench": inspection.get("apiflow_bench"),
            "inspect_ai": inspection.get("inspect_ai"),
        },
        "tasks": tasks,
    }
    config.apiflow.task_manifest_path.parent.mkdir(parents=True, exist_ok=True)
    config.apiflow.task_manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return manifest


def run_apiflow_benchmark(
    config: ExperimentConfig,
    *,
    task_ids: Sequence[str] = (),
    limit: int | None = None,
) -> APIFlowRunSummary:
    _validate_config(config)
    manifest = _load_manifest(config)
    selected = list(manifest["tasks"])
    if task_ids:
        wanted = set(task_ids)
        selected = [task for task in selected if task["task_id"] in wanted]
        missing = wanted - {task["task_id"] for task in selected}
        if missing:
            raise ValueError(f"unknown APIFlow task IDs: {sorted(missing)}")
    if limit is not None:
        selected = selected[:limit]
    existing = _terminal_records(config.apiflow.results_path)
    pending = [
        (task, epoch)
        for task in selected
        for epoch in range(config.apiflow.epochs)
        if _record_key(task["task_id"], epoch) not in existing
    ]
    if config.apiflow.workers == 1:
        for task, epoch in pending:
            _progress(
                config.apiflow.progress_path,
                {"task_id": task["task_id"], "epoch": epoch, "status": "started"},
            )
            record = _run_one(config, task, epoch)
            _record_completed_trial(config, record, existing)
    elif pending:
        _run_concurrent(config, pending, existing)
    records = [
        existing[_record_key(task["task_id"], epoch)]
        for task in selected
        for epoch in range(config.apiflow.epochs)
        if _record_key(task["task_id"], epoch) in existing
    ]
    summary = _summarize(selected, records)
    report = _build_report(config, manifest, records, summary)
    config.apiflow.report_path.parent.mkdir(parents=True, exist_ok=True)
    config.apiflow.report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, default=repr), encoding="utf-8"
    )
    return summary


def evaluate_apiflow_benchmark(config: ExperimentConfig) -> dict[str, Any]:
    report = json.loads(config.apiflow.report_path.read_text(encoding="utf-8"))
    expected = config.apiflow.expected_tasks * config.apiflow.epochs
    if report.get("summary", {}).get("processed") != expected:
        raise ValueError(f"APIFlow report is incomplete: expected {expected} terminal trials")
    return report


def compare_apiflow_benchmarks(
    graph_config: ExperimentConfig,
    baseline_config: ExperimentConfig,
    output_path: Path,
) -> dict[str, Any]:
    _validate_arm_pair(graph_config, baseline_config)
    graph = evaluate_apiflow_benchmark(graph_config)
    baseline = evaluate_apiflow_benchmark(baseline_config)
    graph_records = {_record_key(r["task_id"], r["epoch"]): r for r in graph["trials"]}
    baseline_records = {
        _record_key(r["task_id"], r["epoch"]): r for r in baseline["trials"]
    }
    if graph_records.keys() != baseline_records.keys():
        raise ValueError("APIFlow arm reports do not contain identical task/epoch pairs")
    pairs = [(graph_records[key], baseline_records[key]) for key in sorted(graph_records)]
    report = {
        "schema_version": 1,
        "benchmark": "APIFlow-Bench 1.0",
        "epochs": graph_config.apiflow.epochs,
        "temperature": graph_config.model.temperature,
        "overall": _paired_metrics(pairs),
        "per_kind": {
            kind: _paired_metrics([p for p in pairs if p[0]["task"]["kind"] == kind])
            for kind in ("solo", "chain")
        },
        "per_axis": {
            axis: _paired_metrics([p for p in pairs if p[0]["task"]["axis"] == axis])
            for axis in sorted({p[0]["task"]["axis"] for p in pairs})
        },
        "graph_delta_mechanism": {
            "tasks_with_delta": sum(
                int((r.get("graph_delta_sequence") or {}).get("graph_deltas", 0)) > 0
                for r in graph_records.values()
            ),
            "graph_deltas": sum(
                int((r.get("graph_delta_sequence") or {}).get("graph_deltas", 0))
                for r in graph_records.values()
            ),
            "deltas_preceding_later_action": sum(
                int((r.get("graph_delta_sequence") or {}).get("deltas_preceding_later_action", 0))
                for r in graph_records.values()
            ),
            "causal_influence_established": False,
        },
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def _run_one(
    config: ExperimentConfig, task: Mapping[str, Any], epoch: int
) -> dict[str, Any]:
    task_id = str(task["task_id"])
    runtime: APIFlowProgramRuntime | None = None
    controller: GoalGraphAdaptation | None = None
    agent_result = None
    evaluation: dict[str, Any] | None = None
    messages: list[dict[str, Any]] = []
    error: str | None = None
    started = time.time()
    try:
        runtime = APIFlowProgramRuntime(
            worker_command=config.apiflow.official_worker_command,
            root=config.apiflow.root,
            task_id=task_id,
            timeout_seconds=config.runtime.code_timeout_seconds,
        )
        instruction = str(runtime.metadata["instruction"])
        hooks: dict[str, Any]
        if config.runtime.graph_adaptation_mode == "generic":
            functions = {function.__name__: function for function in runtime.functions}
            controller = GoalGraphAdaptation(
                functions,
                _contracts(),
                task=instruction,
                expose_graph_api=False,
            )
            hooks = GraphAgentHooks.from_controller(controller).agent_kwargs()
        else:
            hooks = {"runtime_functions": runtime.functions}
        latest_checkpoint: dict[str, Any] = {}

        def checkpoint(value: dict[str, Any]) -> None:
            latest_checkpoint.clear()
            latest_checkpoint.update(copy.deepcopy(value))

        model = OpenAIChatModel(
            config.model, config.require_api_key(config.model.api_key_env)
        )
        agent = OriginalPTCAgent(
            model=model,
            search_tools=_EmptySearchTools(),  # type: ignore[arg-type]
            runtime=config.runtime,
            system_prompt=str(runtime.metadata["system_prompt"]) + PTC_OVERLAY,
            user_prompt_template=USER_PROMPT,
            ptc_tool_spec=_ptc_spec(config),
            demonstration_messages=_demo_messages(),
            program_runtime=runtime,
            checkpoint_callback=checkpoint,
            **hooks,
        )
        agent_result = agent.run(instruction)
        messages = list(latest_checkpoint.get("messages") or [])
        evaluation = runtime.evaluate(agent_result.answer)
    except BaseException as exc:
        error = f"{type(exc).__name__}: {exc}"
    finally:
        if runtime is not None:
            try:
                runtime.close()
            except BaseException as exc:
                error = error or f"runtime close {type(exc).__name__}: {exc}"
    artifact = config.apiflow.artifact_dir / f"{task_id}-epoch{epoch}"
    artifact.mkdir(parents=True, exist_ok=True)
    if evaluation is not None:
        (artifact / "evaluation.json").write_text(
            json.dumps(evaluation, ensure_ascii=False, indent=2, default=repr), encoding="utf-8"
        )
    if agent_result is not None:
        (artifact / "execution.json").write_text(
            json.dumps(
                {
                    "agent": agent_result.to_dict(),
                    "runtime": runtime.telemetry() if runtime is not None else None,
                    "messages": messages,
                },
                ensure_ascii=False,
                indent=2,
                default=repr,
            ),
            encoding="utf-8",
        )
    if controller is not None:
        controller.finish(answered=agent_result is not None and agent_result.status == "success")
        config.apiflow.graph_dir.mkdir(parents=True, exist_ok=True)
        graph_artifact = _redact_secret_strings(
            controller.graph_artifact(), runtime.secret_values if runtime is not None else ()
        )
        (config.apiflow.graph_dir / f"{task_id}-epoch{epoch}.json").write_text(
            json.dumps(graph_artifact, ensure_ascii=False, indent=2, default=repr),
            encoding="utf-8",
        )
    compact_evaluation = None
    if evaluation is not None:
        compact_evaluation = {
            key: evaluation.get(key)
            for key in ("passed", "reason", "score", "predicates", "sub_predicates")
        }
    record = {
        "task_id": task_id,
        "epoch": epoch,
        "task": dict(task),
        "status": "finished" if error is None else "failed",
        "passed": bool((evaluation or {}).get("passed")) and error is None,
        "agent": agent_result.to_dict() if agent_result is not None else None,
        "evaluation": compact_evaluation,
        "runtime": runtime.telemetry() if runtime is not None else None,
        "graph_telemetry": controller.telemetry() if controller is not None else None,
        "graph_delta_sequence": graph_delta_sequence(messages),
        "error": error,
        "duration_seconds": time.time() - started,
    }
    (artifact / "meta.json").write_text(
        json.dumps(record, ensure_ascii=False, indent=2, default=repr), encoding="utf-8"
    )
    return record


def _run_one_to_artifact(
    config: ExperimentConfig, task: Mapping[str, Any], epoch: int
) -> str:
    record = _run_one(config, task, epoch)
    return str(config.apiflow.artifact_dir / f"{record['task_id']}-epoch{epoch}" / "meta.json")


def _run_concurrent(
    config: ExperimentConfig,
    pending: Sequence[tuple[Mapping[str, Any], int]],
    existing: dict[str, dict[str, Any]],
) -> None:
    jobs = iter(pending)
    with ProcessPoolExecutor(max_workers=config.apiflow.workers) as executor:
        in_flight: dict[Any, tuple[Mapping[str, Any], int]] = {}

        def submit_next() -> bool:
            try:
                task, epoch = next(jobs)
            except StopIteration:
                return False
            _progress(
                config.apiflow.progress_path,
                {"task_id": task["task_id"], "epoch": epoch, "status": "started"},
            )
            future = executor.submit(_run_one_to_artifact, config, task, epoch)
            in_flight[future] = (task, epoch)
            return True

        for _ in range(min(config.apiflow.workers, len(pending))):
            submit_next()
        while in_flight:
            done, _ = wait(in_flight, return_when=FIRST_COMPLETED)
            for future in done:
                in_flight.pop(future)
                record = json.loads(Path(future.result()).read_text(encoding="utf-8"))
                _record_completed_trial(config, record, existing)
                submit_next()


def _record_completed_trial(
    config: ExperimentConfig,
    record: dict[str, Any],
    existing: dict[str, dict[str, Any]],
) -> None:
    _append_jsonl(config.apiflow.results_path, record)
    existing[_record_key(record["task_id"], record["epoch"])] = record
    _progress(
        config.apiflow.progress_path,
        {
            "task_id": record["task_id"],
            "epoch": record["epoch"],
            "status": record["status"],
            "passed": record["passed"],
        },
    )


def _build_report(
    config: ExperimentConfig,
    manifest: Mapping[str, Any],
    records: Sequence[Mapping[str, Any]],
    summary: APIFlowRunSummary,
) -> dict[str, Any]:
    outcomes = [
        {"world": record["task"]["world"], "passed": bool(record["passed"])}
        for record in records
    ]
    official_score = _official_request(
        config.apiflow.official_worker_command,
        {"type": "score", "outcomes": outcomes},
        timeout=120,
    )
    return {
        "schema_version": 1,
        "benchmark": "APIFlow-Bench 1.0",
        "evaluation_label": "post-release custom-agent reproduction",
        "summary": summary.to_dict(),
        "official_cluster_bootstrap_score": official_score,
        "bank": {
            "release": manifest["release"],
            "bank_sha256": manifest["bank_sha256"],
            "environment": manifest["environment"],
        },
        "configuration": {
            "epochs": config.apiflow.epochs,
            "temperature": config.model.temperature,
            "model": config.model.model,
            "thinking": config.model.thinking,
            "max_completion_tokens": config.model.max_completion_tokens,
            "model_max_retries": config.model.max_retries,
            "retry_all_errors": config.model.retry_all_errors,
            "task_timeout_seconds": config.runtime.task_timeout_seconds,
            "code_timeout_seconds": config.runtime.code_timeout_seconds,
            "workers": config.apiflow.workers,
            "graph_adaptation_mode": config.runtime.graph_adaptation_mode,
            "prompt_variant": config.apiflow.prompt_variant,
        },
        "known_bank_limitations": {
            "public_tasks_and_transcripts": True,
            "replay_verified_tasks": 465,
            "unverified_solo_tasks": ["v56-w06-sub06", "v56-w18-sub15"],
            "known_chain_instruction_defects_retained": True,
        },
        "trials": list(records),
    }


def _summarize(
    selected: Sequence[Mapping[str, Any]], records: Sequence[Mapping[str, Any]]
) -> APIFlowRunSummary:
    passed = sum(bool(record.get("passed")) for record in records)
    return APIFlowRunSummary(
        selected=len(selected),
        processed=len(records),
        passed=passed,
        failed=len(records) - passed,
        runner_failures=sum(record.get("status") != "finished" for record in records),
        tool_calls=sum(int((record.get("runtime") or {}).get("tool_calls", 0)) for record in records),
        input_tokens=sum(int(((record.get("agent") or {}).get("usage") or {}).get("input_tokens", 0)) for record in records),
        output_tokens=sum(int(((record.get("agent") or {}).get("usage") or {}).get("output_tokens", 0)) for record in records),
    )


def _paired_metrics(pairs: Sequence[tuple[Mapping[str, Any], Mapping[str, Any]]]) -> dict[str, Any]:
    graph_passes = sum(bool(a.get("passed")) for a, _ in pairs)
    baseline_passes = sum(bool(b.get("passed")) for _, b in pairs)
    total = len(pairs)
    return {
        "total": total,
        "graph_passed": graph_passes,
        "baseline_passed": baseline_passes,
        "graph_pass_rate": graph_passes / total if total else 0.0,
        "baseline_pass_rate": baseline_passes / total if total else 0.0,
        "absolute_delta": (graph_passes - baseline_passes) / total if total else 0.0,
        "graph_wins": sum(bool(a.get("passed")) and not bool(b.get("passed")) for a, b in pairs),
        "graph_losses": sum(not bool(a.get("passed")) and bool(b.get("passed")) for a, b in pairs),
        "ties": sum(bool(a.get("passed")) == bool(b.get("passed")) for a, b in pairs),
    }


def _contracts() -> dict[str, ToolEffectContract]:
    return {
        name: ToolEffectContract(
            name=name,
            effect="read" if name in {"read", "search"} else "write",
            normalize_arguments=_redact,
            normalize_artifact=_redact,
        )
        for name in APIFLOW_TOOLS
    }


def _ptc_spec(config: ExperimentConfig) -> dict[str, Any]:
    if config.runtime.graph_adaptation_mode == "off":
        return copy.deepcopy(APIFLOW_PTC_SPEC)
    if config.runtime.graph_adaptation_mode != "generic":
        raise ValueError("APIFlow graph adaptation must be off or generic")
    return extend_ptc_spec_with_graph_control(
        APIFLOW_PTC_SPEC,
        include_input_artifacts=False,
        target_description="Use task for this APIFlow episode.",
    )


def _validate_config(config: ExperimentConfig) -> None:
    if config.model.model != "mimo-v2.5":
        raise ValueError("APIFlow model must be mimo-v2.5")
    if config.model.temperature != 1:
        raise ValueError("APIFlow temperature must be one")
    if config.model.max_retries != -1 or config.model.retry_all_errors:
        raise ValueError("APIFlow must retry only transport failures until the task deadline")
    if config.model.thinking != "disabled":
        raise ValueError("APIFlow thinking must be disabled")
    if config.apiflow.epochs != 1:
        raise ValueError("APIFlow evaluation is frozen to one epoch")
    if config.apiflow.workers < 1:
        raise ValueError("APIFlow workers must be positive")
    if config.runtime.task_timeout_seconds != 7200:
        raise ValueError("APIFlow full-bank task timeout must be 7200 seconds")
    if config.runtime.graph_adaptation_mode not in {"off", "generic"}:
        raise ValueError("APIFlow graph adaptation must be off or generic")
    if not config.apiflow.official_worker_command:
        raise ValueError("APIFlow official worker command is required")
    if config.apiflow.prompt_variant != "apiflow-ptc-fewshot":
        raise ValueError("unsupported APIFlow prompt variant")


def _validate_arm_pair(graph: ExperimentConfig, baseline: ExperimentConfig) -> None:
    _validate_config(graph)
    _validate_config(baseline)
    if graph.model != baseline.model:
        raise ValueError("APIFlow arms use different model configs")
    graph_runtime = vars(graph.runtime) | {"graph_adaptation_mode": "off"}
    if graph_runtime != vars(baseline.runtime):
        raise ValueError("APIFlow arms differ outside graph adaptation")
    if graph.runtime.graph_adaptation_mode != "generic" or baseline.runtime.graph_adaptation_mode != "off":
        raise ValueError("APIFlow arm roles are invalid")
    ignored = {"results_path", "report_path", "artifact_dir", "graph_dir", "progress_path"}
    left = {k: v for k, v in vars(graph.apiflow).items() if k not in ignored}
    right = {k: v for k, v in vars(baseline.apiflow).items() if k not in ignored}
    if left != right:
        raise ValueError("APIFlow arm benchmark configs differ")


def _load_manifest(config: ExperimentConfig) -> dict[str, Any]:
    if not config.apiflow.task_manifest_path.exists():
        raise ValueError("APIFlow manifest is missing; run inspect-apiflow first")
    manifest = json.loads(config.apiflow.task_manifest_path.read_text(encoding="utf-8"))
    bank_sha256 = manifest.get("bank_sha256")
    if (
        not isinstance(bank_sha256, str)
        or len(bank_sha256) != 64
        or any(character not in "0123456789abcdef" for character in bank_sha256.lower())
    ):
        raise ValueError("APIFlow manifest has no valid frozen bank SHA-256")
    return manifest


def _official_request(
    command: Sequence[str], payload: Mapping[str, Any], *, timeout: float
) -> dict[str, Any]:
    env = {**os.environ, "PYTHONUTF8": "1"}
    completed = subprocess.run(
        tuple(command),
        input=(
            json.dumps(dict(payload), ensure_ascii=True)
            + "\n"
            + json.dumps({"type": "close"})
            + "\n"
        ),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        env=env,
        timeout=timeout,
    )
    lines = [line for line in completed.stdout.splitlines() if line.strip()]
    if completed.returncode != 0 or not lines:
        raise RuntimeError((completed.stderr or completed.stdout)[-2000:])
    response = json.loads(lines[0])
    if response.get("type") == "error":
        raise RuntimeError(str(response.get("error")))
    return response


def _record_key(task_id: str, epoch: int) -> str:
    return f"{task_id}:epoch{epoch}"


def _terminal_records(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    records: dict[str, dict[str, Any]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        key = _record_key(record["task_id"], int(record["epoch"]))
        if key in records:
            raise ValueError(f"duplicate APIFlow terminal record: {key}")
        records[key] = record
    return records


def _append_jsonl(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(dict(value), ensure_ascii=False, default=repr) + "\n")


def _progress(path: Path, value: Mapping[str, Any]) -> None:
    _append_jsonl(path, {"timestamp": time.time(), **dict(value)})


def _redact_secret_strings(value: Any, secrets: Sequence[str]) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _redact_secret_strings(item, secrets) for key, item in value.items()}
    if isinstance(value, list):
        return [_redact_secret_strings(item, secrets) for item in value]
    if isinstance(value, str):
        redacted = value
        for secret in secrets:
            if secret:
                redacted = redacted.replace(secret, "<redacted>")
        return redacted
    return value
