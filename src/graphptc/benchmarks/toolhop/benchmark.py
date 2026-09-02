from __future__ import annotations

import ast
import copy
import dataclasses
import hashlib
import json
import math
import operator
import os
import subprocess
import time
from collections.abc import Mapping, Sequence
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ...config import ExperimentConfig
from ...graph.adaptation import GoalGraphAdaptation
from ...graph.hooks import GraphAgentHooks, extend_ptc_spec_with_graph_control
from ...graph.diagnostics import graph_delta_sequence
from ...model import OpenAIChatModel
from ...agents.original_ptc import OriginalPTCAgent, PTC_TOOL_SPEC
from ...graph.tool_effects import ToolEffectContract
from .runtime import ToolHopProgramRuntime


TOOLHOP_COMMIT = "b439d7279af359fda46e8117ae4f0245b75f5c6b"

OFFICIAL_MANDATORY_PROMPT = """You will be asked a question with some tools, and should provide a short final answer.
Please note that you must call the tool at every step, you must not use your own knowledge. Your final answer must also be returned from the tool.
If the final answer is a date, format is as follows: YYYY-MM-DD (ISO standard)
If the final nswer is a name, format it as follows: Firstname Lastname
If the final answer contains any number, format it as a number, not a word, and only output that number. Do not include leading 0s.

Please provide the final answer in the following format: <answer>final answer here</answer>
Answer as short as possible.
Question: {question}"""

SYSTEM_PROMPT = """You are solving the official ToolHop Mandatory scenario through programmatic tool calling.

Your only directly callable model tool is `programmatic_tool_call`. Its Python source runs in one
persistent namespace for this task. Every function in <runtime_tool_definitions> is available as a
Python global with the exact documented name and keyword parameters. Tool results are ordinary
Python values.

Use each PTC block as a coherent program: call one or more relevant tools, and use Python for
mechanically related loops, filtering, joins, aggregation, and selection. State persists across
blocks. Only printed stdout is visible on the next turn, so print compact decision-relevant values.
Call wrappers with keyword arguments. Do not introspect wrappers or access files, environment
variables, the shell, or the network outside the provided tools. Follow the Mandatory requirement
that every factual step and the final answer come from tool output.

<runtime_tool_definitions>
{tools}
</runtime_tool_definitions>"""

FINALIZE_PROMPT = """The ToolHop tool-use phase is complete and tools are unavailable. Return the
short final answer supported by the tool outputs. Do not emit tool-call markup, Python, a plan, or
<result> tags. Use exactly <answer>final answer here</answer>, as required by ToolHop."""

TOOLHOP_PTC_SPEC = {
    **PTC_TOOL_SPEC,
    "function": {
        **PTC_TOOL_SPEC["function"],
        "description": (
            "Execute one coherent Python program directly in this task's persistent "
            "namespace. The task's official ToolHop functions are globals."
        ),
    },
}


def _demo_messages() -> tuple[dict[str, Any], ...]:
    return (
        {
            "role": "user",
            "content": (
                "PTC organization example only, using hypothetical tools: obtain several "
                "records and derive one value from their outputs."
            ),
        },
        {
            "role": "assistant",
            "content": "I will keep the dependent calls and deterministic selection in one block.",
            "tool_calls": [
                {
                    "id": "toolhop_demo_1",
                    "type": "function",
                    "function": {
                        "name": "programmatic_tool_call",
                        "arguments": json.dumps(
                            {
                                "code": (
                                    "rows = [lookup_record(key=k) for k in ['a', 'b']]\n"
                                    "valid = [row for row in rows if row.get('active')]\n"
                                    "print({'values': [row['value'] for row in valid]})"
                                )
                            }
                        ),
                    },
                }
            ],
        },
        {
            "role": "tool",
            "tool_call_id": "toolhop_demo_1",
            "content": "{'values': [7]}",
        },
        {
            "role": "assistant",
            "content": "I would return the short answer supported by the printed tool-derived value.",
        },
    )


class _EmptySearchTools:
    calls: list[dict[str, Any]] = []


@dataclass(frozen=True)
class ToolHopRunSummary:
    selected: int
    processed: int
    official_passed: int
    strict_passed: int
    runner_failures: int
    execution_failures: int
    tool_failure_tasks: int
    tool_failures: int
    missing_final_answer: int
    tool_calls: int
    input_tokens: int
    output_tokens: int

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


def inspect_toolhop(config: ExperimentConfig) -> dict[str, Any]:
    _validate_config(config)
    frozen_manifest = _load_manifest(config)
    dataset_sha256 = _sha256(config.toolhop.dataset_path)
    expected_data_sha256 = str(frozen_manifest["data_sha256"])
    if dataset_sha256 != expected_data_sha256:
        raise ValueError(
            f"ToolHop data hash mismatch: expected {expected_data_sha256}, "
            f"got {dataset_sha256}"
        )
    commit = subprocess.run(
        ("git", "-C", config.toolhop.root, "rev-parse", "HEAD"),
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=30,
    ).stdout.strip()
    if commit != config.toolhop.official_commit:
        raise ValueError(
            f"ToolHop commit mismatch: expected {config.toolhop.official_commit}, got {commit}"
        )
    tasks = _load_dataset(config.toolhop.dataset_path)
    if len(tasks) != config.toolhop.expected_tasks:
        raise ValueError(
            f"expected {config.toolhop.expected_tasks} ToolHop tasks, found {len(tasks)}"
        )
    ids = [str(task["id"]) for task in tasks]
    if len(set(ids)) != len(ids):
        raise ValueError("ToolHop task IDs are not unique")
    inspection = _worker_request(config.toolhop.official_worker_command, {"type": "inspect"})
    packages = ((inspection.get("environment") or {}).get("packages") or {})
    missing = [name for name, version in packages.items() if version is None]
    if missing:
        raise ValueError(f"ToolHop worker dependencies are missing: {missing}")
    manifest = {
        "schema_version": 1,
        "benchmark": "ToolHop",
        "scenario": config.toolhop.scenario,
        "official_commit": commit,
        "data_sha256": dataset_sha256,
        "expected_tasks": config.toolhop.expected_tasks,
        "epochs": config.toolhop.epochs,
        "environment": inspection.get("environment"),
        "tasks": [
            {
                "task_id": str(task["id"]),
                "domain": task.get("domain"),
                "answer_type": task.get("answer_type"),
                "subtasks": len(task.get("sub_task") or ()),
                "tools": len(task.get("tools") or {}),
            }
            for task in tasks
        ],
    }
    path = config.toolhop.task_manifest_path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest


def run_toolhop_benchmark(
    config: ExperimentConfig,
    *,
    task_ids: Sequence[str] = (),
    limit: int | None = None,
) -> ToolHopRunSummary:
    _validate_config(config)
    manifest = _load_manifest(config)
    tasks = _load_dataset(config.toolhop.dataset_path)
    by_id = {str(task["id"]): task for task in tasks}
    selected_ids = [str(task["task_id"]) for task in manifest["tasks"]]
    if task_ids:
        wanted = set(task_ids)
        missing = wanted - by_id.keys()
        if missing:
            raise ValueError(f"unknown ToolHop task IDs: {sorted(missing)}")
        selected_ids = [task_id for task_id in selected_ids if task_id in wanted]
    if limit is not None:
        selected_ids = selected_ids[:limit]
    selected = [by_id[task_id] for task_id in selected_ids]
    existing = _terminal_records(config.toolhop.results_path)
    pending = [
        (task, epoch)
        for task in selected
        for epoch in range(config.toolhop.epochs)
        if _record_key(str(task["id"]), epoch) not in existing
    ]
    if config.toolhop.workers == 1:
        for task, epoch in pending:
            _progress(config.toolhop.progress_path, task, epoch, "started")
            _record_completed(config, _run_one(config, task, epoch), existing)
    elif pending:
        _run_concurrent(config, pending, existing)
    _rescore_existing_records(config, existing)
    records = [
        existing[_record_key(str(task["id"]), epoch)]
        for task in selected
        for epoch in range(config.toolhop.epochs)
        if _record_key(str(task["id"]), epoch) in existing
    ]
    summary = _summarize(selected, records)
    report = _build_report(config, manifest, records, summary)
    config.toolhop.report_path.parent.mkdir(parents=True, exist_ok=True)
    config.toolhop.report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, default=repr), encoding="utf-8"
    )
    return summary


def evaluate_toolhop_benchmark(config: ExperimentConfig) -> dict[str, Any]:
    report = json.loads(config.toolhop.report_path.read_text(encoding="utf-8"))
    expected = config.toolhop.expected_tasks * config.toolhop.epochs
    if int((report.get("summary") or {}).get("processed", -1)) != expected:
        raise ValueError(f"ToolHop report is incomplete: expected {expected} terminal trials")
    return report


def compare_toolhop_benchmarks(
    graph_config: ExperimentConfig,
    baseline_config: ExperimentConfig,
    output_path: Path,
) -> dict[str, Any]:
    _validate_arm_pair(graph_config, baseline_config)
    graph = evaluate_toolhop_benchmark(graph_config)
    baseline = evaluate_toolhop_benchmark(baseline_config)
    graph_records = {_record_key(r["task_id"], r["epoch"]): r for r in graph["trials"]}
    base_records = {_record_key(r["task_id"], r["epoch"]): r for r in baseline["trials"]}
    if graph_records.keys() != base_records.keys():
        raise ValueError("ToolHop arms do not contain identical task/epoch pairs")
    pairs = [(graph_records[key], base_records[key]) for key in sorted(graph_records)]
    report = {
        "schema_version": 1,
        "benchmark": "ToolHop",
        "scenario": "Mandatory",
        "protocol_label": "post-release custom-agent PTC reproduction",
        "official": _paired_metrics(pairs, "official_passed"),
        "strict": _paired_metrics(pairs, "strict_passed"),
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
                int(
                    (r.get("graph_delta_sequence") or {}).get(
                        "deltas_preceding_later_action", 0
                    )
                )
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
    task_id = str(task["id"])
    runtime: ToolHopProgramRuntime | None = None
    controller: GoalGraphAdaptation | None = None
    agent_result = None
    messages: list[dict[str, Any]] = []
    score: dict[str, Any] | None = None
    error: str | None = None
    started = time.time()
    try:
        runtime = ToolHopProgramRuntime(
            worker_command=config.toolhop.official_worker_command,
            task=task,
            timeout_seconds=config.runtime.code_timeout_seconds,
        )
        functions = {function.__name__: function for function in runtime.functions}
        hooks: dict[str, Any]
        if config.runtime.graph_adaptation_mode == "generic":
            controller = GoalGraphAdaptation(
                functions,
                _contracts(functions),
                task=str(task["question"]),
                expose_graph_api=False,
            )
            hooks = GraphAgentHooks.from_controller(controller).agent_kwargs()
        else:
            hooks = {"runtime_functions": runtime.functions}
        latest_checkpoint: dict[str, Any] = {}

        def checkpoint(value: dict[str, Any]) -> None:
            latest_checkpoint.clear()
            latest_checkpoint.update(copy.deepcopy(value))

        agent = OriginalPTCAgent(
            model=OpenAIChatModel(
                config.model, config.require_api_key(config.model.api_key_env)
            ),
            search_tools=_EmptySearchTools(),  # type: ignore[arg-type]
            runtime=config.runtime,
            system_prompt=SYSTEM_PROMPT.format(
                tools=json.dumps(list((task.get("tools") or {}).values()), ensure_ascii=False)
            ),
            user_prompt_template="{question}",
            finalize_prompt=FINALIZE_PROMPT,
            ptc_tool_spec=_ptc_spec(config),
            demonstration_messages=_demo_messages(),
            program_runtime=runtime,
            checkpoint_callback=checkpoint,
            **hooks,
        )
        instruction = OFFICIAL_MANDATORY_PROMPT.format(question=task["question"])
        agent_result = agent.run(instruction)
        messages = list(latest_checkpoint.get("messages") or [])
        last_tool_output = (
            agent_result.blocks[-1].stdout if agent_result.blocks else None
        )
        score = score_answer(str(task["answer"]), agent_result.answer, last_tool_output)
    except BaseException as exc:
        error = f"{type(exc).__name__}: {exc}"
    finally:
        if controller is not None:
            controller.finish(
                answered=agent_result is not None and agent_result.status == "success"
            )
        if runtime is not None:
            try:
                runtime.close()
            except BaseException as exc:
                error = error or f"runtime close {type(exc).__name__}: {exc}"

    artifact = config.toolhop.artifact_dir / f"{task_id}-epoch{epoch}"
    artifact.mkdir(parents=True, exist_ok=True)
    if agent_result is not None:
        (artifact / "execution.json").write_text(
            json.dumps(
                {
                    "agent": agent_result.to_dict(),
                    "messages": messages,
                    "api_calls": list(runtime.calls) if runtime is not None else [],
                },
                ensure_ascii=False,
                indent=2,
                default=repr,
            ),
            encoding="utf-8",
        )
    if score is not None:
        (artifact / "evaluation.json").write_text(
            json.dumps(score, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    if controller is not None:
        config.toolhop.graph_dir.mkdir(parents=True, exist_ok=True)
        (config.toolhop.graph_dir / f"{task_id}-epoch{epoch}.json").write_text(
            json.dumps(
                controller.graph_artifact(), ensure_ascii=False, indent=2, default=repr
            ),
            encoding="utf-8",
        )
    runtime_data = runtime.telemetry() if runtime is not None else None
    record = {
        "task_id": task_id,
        "epoch": epoch,
        "task": {
            "domain": task.get("domain"),
            "answer_type": task.get("answer_type"),
            "tools": len(task.get("tools") or {}),
        },
        "status": "finished" if error is None else "failed",
        "official_passed": bool((score or {}).get("official_passed")) and error is None,
        "strict_passed": bool((score or {}).get("strict_passed")) and error is None,
        "evaluation": score,
        "agent": agent_result.to_dict() if agent_result is not None else None,
        "runtime": runtime_data,
        "graph_telemetry": controller.telemetry() if controller is not None else None,
        "graph_delta_sequence": graph_delta_sequence(messages),
        "error": error,
        "duration_seconds": time.time() - started,
    }
    (artifact / "meta.json").write_text(
        json.dumps(record, ensure_ascii=False, indent=2, default=repr), encoding="utf-8"
    )
    return record


def score_answer(
    ground_truth_text: str, final_answer: str, last_tool_output: str | None
) -> dict[str, Any]:
    solution = _extract_answer(final_answer)
    official_passed = False
    official_branch: str | None = None
    try:
        ground_truth = _official_eval(ground_truth_text.strip())
    except (ValueError, SyntaxError):
        if _official_contains(ground_truth_text, solution):
            official_passed = True
            official_branch = "answer_substring"
    else:
        try:
            candidate = _official_eval(solution.strip())
        except (ValueError, SyntaxError):
            pass
        else:
            if ground_truth == candidate:
                official_passed = True
                official_branch = "literal_equality"
    if last_tool_output is not None and _official_contains(
        ground_truth_text, last_tool_output
    ):
        official_passed = True
        official_branch = "last_tool_output"
    strict_passed = _strict_equal(ground_truth_text, solution)
    return {
        "official_passed": official_passed,
        "official_branch": official_branch,
        "strict_passed": strict_passed,
        "ground_truth": ground_truth_text,
        "extracted_answer": solution,
        "last_tool_output": last_tool_output,
        "scorer": "official ToolHop rule reimplementation with safe expression evaluation",
    }


def _extract_answer(value: str) -> str:
    output = str(value or "")
    if "<answer>" in output:
        output = output.split("<answer>")[-1]
        if "</answer>" in output:
            output = output.split("</answer>")[0]
    return output.strip()


def _official_contains(expected: str, actual: str) -> bool:
    needle = str(expected).removesuffix(".0").lower()
    haystack = str(actual).removesuffix(".0").replace(",", "").lower()
    return needle in haystack


def _strict_equal(expected: str, actual: str) -> bool:
    try:
        return ast.literal_eval(expected.strip()) == ast.literal_eval(actual.strip())
    except (ValueError, SyntaxError):
        return expected.strip().casefold() == actual.strip().casefold()


_BINARY_OPERATORS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}
_UNARY_OPERATORS = {ast.UAdd: operator.pos, ast.USub: operator.neg}


def _official_eval(source: str) -> Any:
    """Evaluate ToolHop answer expressions without executing arbitrary model code."""
    tree = ast.parse(source, mode="eval")

    def evaluate(node: ast.AST) -> Any:
        if isinstance(node, ast.Expression):
            return evaluate(node.body)
        if isinstance(node, ast.Constant):
            return node.value
        if isinstance(node, ast.List):
            return [evaluate(value) for value in node.elts]
        if isinstance(node, ast.Tuple):
            return tuple(evaluate(value) for value in node.elts)
        if isinstance(node, ast.Set):
            return {evaluate(value) for value in node.elts}
        if isinstance(node, ast.Dict):
            return {
                evaluate(key): evaluate(value)
                for key, value in zip(node.keys, node.values)
            }
        if isinstance(node, ast.BinOp) and type(node.op) in _BINARY_OPERATORS:
            return _BINARY_OPERATORS[type(node.op)](
                evaluate(node.left), evaluate(node.right)
            )
        if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY_OPERATORS:
            return _UNARY_OPERATORS[type(node.op)](evaluate(node.operand))
        raise ValueError(f"unsupported answer expression: {type(node).__name__}")

    return evaluate(tree)


def _contracts(functions: Mapping[str, Any]) -> dict[str, ToolEffectContract]:
    return {
        name: ToolEffectContract(name=name, effect="read") for name in functions
    }


def _ptc_spec(config: ExperimentConfig) -> dict[str, Any]:
    if config.runtime.graph_adaptation_mode == "off":
        return copy.deepcopy(TOOLHOP_PTC_SPEC)
    if config.runtime.graph_adaptation_mode != "generic":
        raise ValueError("ToolHop graph adaptation must be off or generic")
    return extend_ptc_spec_with_graph_control(
        TOOLHOP_PTC_SPEC,
        include_input_artifacts=False,
        target_description="Use task for this ToolHop episode.",
    )


def _run_one_to_artifact(
    config: ExperimentConfig, task: Mapping[str, Any], epoch: int
) -> str:
    record = _run_one(config, task, epoch)
    return str(
        config.toolhop.artifact_dir
        / f"{record['task_id']}-epoch{epoch}"
        / "meta.json"
    )


def _run_concurrent(
    config: ExperimentConfig,
    pending: Sequence[tuple[Mapping[str, Any], int]],
    existing: dict[str, dict[str, Any]],
) -> None:
    jobs = iter(pending)
    with ProcessPoolExecutor(max_workers=config.toolhop.workers) as executor:
        in_flight: dict[Any, tuple[Mapping[str, Any], int]] = {}

        def submit_next() -> bool:
            try:
                task, epoch = next(jobs)
            except StopIteration:
                return False
            _progress(config.toolhop.progress_path, task, epoch, "started")
            future = executor.submit(_run_one_to_artifact, config, task, epoch)
            in_flight[future] = (task, epoch)
            return True

        for _ in range(min(config.toolhop.workers, len(pending))):
            submit_next()
        while in_flight:
            done, _ = wait(in_flight, return_when=FIRST_COMPLETED)
            for future in done:
                task, epoch = in_flight.pop(future)
                try:
                    record = json.loads(
                        Path(future.result()).read_text(encoding="utf-8")
                    )
                except BaseException as exc:
                    record = _runner_failure_record(task, epoch, exc)
                _record_completed(config, record, existing)
                submit_next()


def _runner_failure_record(
    task: Mapping[str, Any], epoch: int, exc: BaseException
) -> dict[str, Any]:
    return {
        "task_id": str(task["id"]),
        "epoch": epoch,
        "task": {
            "domain": task.get("domain"),
            "answer_type": task.get("answer_type"),
            "tools": len(task.get("tools") or {}),
        },
        "status": "failed",
        "official_passed": False,
        "strict_passed": False,
        "evaluation": None,
        "agent": None,
        "runtime": None,
        "graph_telemetry": None,
        "graph_delta_sequence": {},
        "error": f"worker {type(exc).__name__}: {exc}",
        "duration_seconds": 0,
    }


def _record_completed(
    config: ExperimentConfig,
    record: dict[str, Any],
    existing: dict[str, dict[str, Any]],
) -> None:
    _append_jsonl(config.toolhop.results_path, record)
    existing[_record_key(record["task_id"], int(record["epoch"]))] = record
    _append_jsonl(
        config.toolhop.progress_path,
        {
            "timestamp": time.time(),
            "task_id": record["task_id"],
            "epoch": record["epoch"],
            "status": record["status"],
            "official_passed": record["official_passed"],
        },
    )


def _progress(
    path: Path, task: Mapping[str, Any], epoch: int, status: str
) -> None:
    _append_jsonl(
        path,
        {
            "timestamp": time.time(),
            "task_id": str(task["id"]),
            "epoch": epoch,
            "status": status,
        },
    )


def _summarize(
    selected: Sequence[Mapping[str, Any]], records: Sequence[Mapping[str, Any]]
) -> ToolHopRunSummary:
    return ToolHopRunSummary(
        selected=len(selected),
        processed=len(records),
        official_passed=sum(bool(record.get("official_passed")) for record in records),
        strict_passed=sum(bool(record.get("strict_passed")) for record in records),
        runner_failures=sum(record.get("status") != "finished" for record in records),
        execution_failures=sum(
            any(
                not bool(block.get("success"))
                for block in ((record.get("agent") or {}).get("blocks") or [])
            )
            for record in records
        ),
        tool_failure_tasks=sum(
            int((record.get("runtime") or {}).get("failed_tool_calls", 0)) > 0
            for record in records
        ),
        tool_failures=sum(
            int((record.get("runtime") or {}).get("failed_tool_calls", 0))
            for record in records
        ),
        missing_final_answer=sum(
            not bool(((record.get("agent") or {}).get("answer") or "").strip())
            for record in records
        ),
        tool_calls=sum(
            int((record.get("runtime") or {}).get("tool_calls", 0)) for record in records
        ),
        input_tokens=sum(
            int((((record.get("agent") or {}).get("usage") or {}).get("input_tokens", 0)))
            for record in records
        ),
        output_tokens=sum(
            int((((record.get("agent") or {}).get("usage") or {}).get("output_tokens", 0)))
            for record in records
        ),
    )


def _build_report(
    config: ExperimentConfig,
    manifest: Mapping[str, Any],
    records: Sequence[Mapping[str, Any]],
    summary: ToolHopRunSummary,
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "benchmark": "ToolHop",
        "scenario": "Mandatory",
        "protocol_label": "post-release custom-agent PTC reproduction",
        "official_commit": manifest["official_commit"],
        "data_sha256": manifest["data_sha256"],
        "environment": manifest["environment"],
        "summary": summary.to_dict(),
        "scores": {
            "official_accuracy": (
                summary.official_passed / summary.processed if summary.processed else 0.0
            ),
            "strict_accuracy": (
                summary.strict_passed / summary.processed if summary.processed else 0.0
            ),
        },
        "configuration": {
            "model": config.model.model,
            "temperature": config.model.temperature,
            "max_completion_tokens": config.model.max_completion_tokens,
            "max_retries": config.model.max_retries,
            "max_turns": config.runtime.max_turns,
            "max_ptc_blocks": config.runtime.max_ptc_blocks,
            "code_timeout_seconds": config.runtime.code_timeout_seconds,
            "task_timeout_seconds": config.runtime.task_timeout_seconds,
            "max_stdout_chars": config.runtime.max_stdout_chars,
            "workers": config.toolhop.workers,
            "epochs": config.toolhop.epochs,
            "prompt_variant": config.toolhop.prompt_variant,
            "graph_adaptation_mode": config.runtime.graph_adaptation_mode,
        },
        "scoring_notes": {
            "official_rule": (
                "ToolHop official literal equality/substring rule plus final tool-output credit"
            ),
            "expression_parser": (
                "safe AST evaluation reproduces official literal/arithmetic eval without code execution"
            ),
            "strict_rule": "whole extracted answer equality with no substring or tool-output credit",
            "leaderboard_native": False,
        },
        "trials": list(records),
    }


def _paired_metrics(
    pairs: Sequence[tuple[Mapping[str, Any], Mapping[str, Any]]], key: str
) -> dict[str, Any]:
    graph_passed = sum(bool(graph.get(key)) for graph, _ in pairs)
    baseline_passed = sum(bool(base.get(key)) for _, base in pairs)
    wins = sum(bool(graph.get(key)) and not bool(base.get(key)) for graph, base in pairs)
    losses = sum(not bool(graph.get(key)) and bool(base.get(key)) for graph, base in pairs)
    discordant = wins + losses
    p_value = min(
        1.0,
        2
        * sum(math.comb(discordant, i) for i in range(min(wins, losses) + 1))
        / (2**discordant),
    ) if discordant else 1.0
    total = len(pairs)
    return {
        "total": total,
        "graph_passed": graph_passed,
        "baseline_passed": baseline_passed,
        "graph_accuracy": graph_passed / total if total else 0.0,
        "baseline_accuracy": baseline_passed / total if total else 0.0,
        "absolute_delta": (graph_passed - baseline_passed) / total if total else 0.0,
        "graph_wins": wins,
        "graph_losses": losses,
        "ties": total - wins - losses,
        "mcnemar_exact_two_sided_p": p_value,
    }


def _validate_config(config: ExperimentConfig) -> None:
    if config.model.model != "mimo-v2.5":
        raise ValueError("ToolHop model must be mimo-v2.5")
    if config.model.temperature != 0:
        raise ValueError("ToolHop official protocol uses temperature zero")
    if config.model.max_completion_tokens != 256:
        raise ValueError("ToolHop official protocol uses max_tokens 256")
    if config.model.thinking != "disabled":
        raise ValueError("ToolHop thinking must be disabled")
    if config.model.max_retries != 2 or config.model.retry_all_errors:
        raise ValueError("ToolHop uses at most three transport attempts")
    if config.runtime.max_turns != 10 or config.runtime.max_ptc_blocks != 9:
        raise ValueError("ToolHop PTC mapping requires nine tool turns plus finalization")
    if config.runtime.code_timeout_seconds != 10:
        raise ValueError("ToolHop official tool timeout is 10 seconds")
    if config.runtime.max_stdout_chars != 8000:
        raise ValueError("ToolHop must preserve the 8k PTC stdout limit")
    if config.runtime.graph_adaptation_mode not in {"off", "generic"}:
        raise ValueError("ToolHop graph adaptation must be off or generic")
    if config.toolhop.scenario != "Mandatory":
        raise ValueError("ToolHop main track is frozen to Mandatory")
    if config.toolhop.expected_tasks != 995 or config.toolhop.epochs != 1:
        raise ValueError("ToolHop evaluation is frozen to 995 tasks and one epoch")
    if config.toolhop.workers < 1:
        raise ValueError("ToolHop workers must be positive")
    if config.toolhop.official_commit != TOOLHOP_COMMIT:
        raise ValueError("ToolHop official commit differs from the frozen commit")
    if config.toolhop.prompt_variant != "toolhop-ptc-fewshot":
        raise ValueError("unsupported ToolHop prompt variant")
    if not config.toolhop.official_worker_command:
        raise ValueError("ToolHop official worker command is required")


def _validate_arm_pair(graph: ExperimentConfig, baseline: ExperimentConfig) -> None:
    _validate_config(graph)
    _validate_config(baseline)
    if graph.model != baseline.model:
        raise ValueError("ToolHop arms use different model configs")
    graph_runtime = vars(graph.runtime) | {"graph_adaptation_mode": "off"}
    if graph_runtime != vars(baseline.runtime):
        raise ValueError("ToolHop arms differ outside graph adaptation")
    if (
        graph.runtime.graph_adaptation_mode != "generic"
        or baseline.runtime.graph_adaptation_mode != "off"
    ):
        raise ValueError("ToolHop arm roles are invalid")
    ignored = {"results_path", "report_path", "artifact_dir", "graph_dir", "progress_path"}
    left = {k: v for k, v in vars(graph.toolhop).items() if k not in ignored}
    right = {k: v for k, v in vars(baseline.toolhop).items() if k not in ignored}
    if left != right:
        raise ValueError("ToolHop arm benchmark configs differ")


def _load_manifest(config: ExperimentConfig) -> dict[str, Any]:
    if not config.toolhop.task_manifest_path.exists():
        raise ValueError("ToolHop manifest is missing; run inspect-toolhop first")
    manifest = json.loads(config.toolhop.task_manifest_path.read_text(encoding="utf-8"))
    if manifest.get("official_commit") != config.toolhop.official_commit:
        raise ValueError("ToolHop manifest commit differs from config")
    data_sha256 = manifest.get("data_sha256")
    if (
        not isinstance(data_sha256, str)
        or len(data_sha256) != 64
        or any(character not in "0123456789abcdef" for character in data_sha256.lower())
    ):
        raise ValueError("ToolHop manifest has no valid frozen data SHA-256")
    return manifest


def _load_dataset(path: Path) -> list[dict[str, Any]]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, list):
        raise ValueError("ToolHop dataset must be a JSON list")
    return [dict(task) for task in value]


def _worker_request(command: Sequence[str], payload: Mapping[str, Any]) -> dict[str, Any]:
    completed = subprocess.run(
        tuple(command),
        input=json.dumps(dict(payload), ensure_ascii=True) + "\n" + json.dumps({"type": "close"}) + "\n",
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        env={**os.environ, "PYTHONUTF8": "1"},
        timeout=60,
    )
    lines = [line for line in completed.stdout.splitlines() if line.strip()]
    if completed.returncode != 0 or not lines:
        raise RuntimeError((completed.stderr or completed.stdout)[-2000:])
    response = json.loads(lines[0])
    if response.get("type") == "error":
        raise RuntimeError(str(response.get("error")))
    return response


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _record_key(task_id: str, epoch: int) -> str:
    return f"{task_id}:epoch{epoch}"


def _rescore_existing_records(
    config: ExperimentConfig, records: dict[str, dict[str, Any]]
) -> None:
    changed = False
    for record in records.values():
        evaluation = record.get("evaluation") or {}
        ground_truth = evaluation.get("ground_truth")
        agent = record.get("agent") or {}
        if ground_truth is None or not agent:
            continue
        blocks = agent.get("blocks") or []
        last_tool_output = blocks[-1].get("stdout") if blocks else None
        score = score_answer(str(ground_truth), str(agent.get("answer") or ""), last_tool_output)
        official_passed = bool(score["official_passed"]) and record.get("status") == "finished"
        strict_passed = bool(score["strict_passed"]) and record.get("status") == "finished"
        if (
            record.get("evaluation") == score
            and record.get("official_passed") == official_passed
            and record.get("strict_passed") == strict_passed
        ):
            continue
        record["evaluation"] = score
        record["official_passed"] = official_passed
        record["strict_passed"] = strict_passed
        artifact = (
            config.toolhop.artifact_dir
            / f"{record['task_id']}-epoch{int(record['epoch'])}"
        )
        (artifact / "evaluation.json").write_text(
            json.dumps(score, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        (artifact / "meta.json").write_text(
            json.dumps(record, ensure_ascii=False, indent=2, default=repr),
            encoding="utf-8",
        )
        changed = True
    if changed:
        temporary = config.toolhop.results_path.with_suffix(".jsonl.tmp")
        temporary.parent.mkdir(parents=True, exist_ok=True)
        with temporary.open("w", encoding="utf-8") as handle:
            for record in records.values():
                handle.write(
                    json.dumps(record, ensure_ascii=False, default=repr) + "\n"
                )
        temporary.replace(config.toolhop.results_path)


def _terminal_records(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    records: dict[str, dict[str, Any]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        key = _record_key(str(record["task_id"]), int(record["epoch"]))
        if key in records:
            raise ValueError(f"duplicate ToolHop terminal record: {key}")
        records[key] = record
    return records


def _append_jsonl(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(dict(value), ensure_ascii=False, default=repr) + "\n")
