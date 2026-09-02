from __future__ import annotations

import copy
import dataclasses
import json
import os
import shutil
import subprocess
import threading
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ...agents.original_ptc import PTC_TOOL_SPEC
from ...config import ExperimentConfig
from ...graph.hooks import extend_ptc_spec_with_graph_control

TOOL_SANDBOX_PTC_BASE_PROMPT = """Follow ToolSandbox's official agent contract: do not assume values for function
arguments. Ask the user for clarification when the request is ambiguous or required information is missing. Use only
the scenario's listed functions, their exact schemas, and values established by the user or tool results. Avoid
unrequested state changes. When the request is fulfilled, answer the user concisely; the official user simulator
decides whether to continue or end the conversation.

Your only directly callable model tool is programmatic_tool_call. Its code is executed directly in ToolSandbox's
persistent Python shell for this scenario. The listed scenario functions are Python globals, and variables and imports
persist across PTC blocks. Do not wrap the program in another execute string. Store intermediate results in variables.
Only printed stdout is returned to the next model turn, so print compact decision-relevant values and never dump
secrets or large collections.

Treat each PTC block as one semantically coherent phase, not as a wrapper around one function call and not as an
attempt to solve every uncertain step in one monolithic program. Put mechanically foreseeable calls, loops,
filtering, joins, and aggregation in the same block. Return to the model for a new semantic decision, a failure that
needs repair, or a user-facing response. ToolSandbox is stateful: observe failures, preserve successful state, and
repair forward without resetting or branching the official environment."""

TOOL_SANDBOX_GRAPH_GUIDANCE = """Graph control uses the same generic contract as the other GraphPTC benchmarks.
The fields on programmatic_tool_call declare intent: use CONTINUE for a new effect, PATCH to correct a failed block,
and REPLAN when changing the dependency path. Set target to the affected existing graph node (normally task) and
expected_change to the observable result. After execution, GRAPH_DELTA reports recorded API calls, artifacts, state
effects, failures, and the next dependency frontier. It is evidence for the next action, not proof that the declared
change occurred."""

TOOL_SANDBOX_DIRECT_PROMPT = """Follow ToolSandbox's official agent contract. The model receives
the scenario's currently available functions as native tools with their authoritative schemas. Do
not assume argument values: ask the user when the request is ambiguous or required information is
missing. Use only listed functions and values established by the user or tool results. Observe and
repair tool failures without resetting the stateful environment, avoid unrequested state changes,
and answer the user concisely when the request is fulfilled. The official user simulator decides
whether to continue or end the conversation."""

TOOL_SANDBOX_PTC_SPEC = {
    **PTC_TOOL_SPEC,
    "function": {
        **PTC_TOOL_SPEC["function"],
        "description": (
            "Execute one coherent Python program directly in the current ToolSandbox "
            "scenario's persistent shell. Listed scenario functions are globals."
        ),
        "parameters": {
            **PTC_TOOL_SPEC["function"]["parameters"],
            "properties": {
                "code": {
                    "type": "string",
                    "description": (
                        "Exact Python source for ToolSandbox's persistent shell. "
                        "Use only listed scenario functions."
                    ),
                }
            },
        },
    },
}


def _demo_call(call_id: str, code: str, expected_change: str) -> dict[str, Any]:
    return {
        "role": "assistant",
        "content": "I will compute the mechanically determined intermediate result in one compact block.",
        "tool_calls": [
            {
                "id": call_id,
                "type": "function",
                "function": {
                    "name": "programmatic_tool_call",
                    "arguments": json.dumps(
                        {
                            "code": code,
                            "action": "CONTINUE",
                            "target": "task",
                            "expected_change": expected_change,
                        }
                    ),
                },
            }
        ],
    }


TOOL_SANDBOX_PTC_FEW_SHOT_MESSAGES: tuple[dict[str, Any], ...] = (
    {
        "role": "user",
        "content": (
            "PTC organization demonstration only: From these already retrieved records, "
            "report how many enabled records have distinct labels: "
            "[{'label':'a','enabled':True},{'label':'a','enabled':True},"
            "{'label':'b','enabled':False},{'label':'c','enabled':True}]."
        ),
    },
    _demo_call(
        "toolsandbox_demo_1",
        "records = [{'label':'a','enabled':True},{'label':'a','enabled':True},"
        "{'label':'b','enabled':False},{'label':'c','enabled':True}]\n"
        "answer = len({row['label'] for row in records if row['enabled']})\n"
        "print({'distinct_enabled_labels': answer})",
        "derive the requested count from the available records",
    ),
    {
        "role": "tool",
        "tool_call_id": "toolsandbox_demo_1",
        "content": (
            "{'distinct_enabled_labels': 2}\n\nGRAPH_DELTA "
            '{"declared_action":{"action":"CONTINUE","target":"task"},'
            '"action_verification":{"realized":true}}'
        ),
    },
    {"role": "assistant", "content": "There are 2 distinct enabled labels."},
)


def _toolsandbox_prompt_bundle(
    variant: str, *, graph_adaptation_mode: str
) -> tuple[str, tuple[dict[str, Any], ...]]:
    if variant == "toolsandbox-direct-tools-v1":
        if graph_adaptation_mode != "off":
            raise ValueError(
                "toolsandbox-direct-tools-v1 requires graph_adaptation_mode='off'"
            )
        return TOOL_SANDBOX_DIRECT_PROMPT, ()
    if variant != "toolsandbox-ptc-fewshot":
        raise ValueError(f"unsupported ToolSandbox prompt variant: {variant!r}")
    if graph_adaptation_mode not in {"off", "generic"}:
        raise ValueError("ToolSandbox graph_adaptation_mode must be off or generic")
    prompt = TOOL_SANDBOX_PTC_BASE_PROMPT
    demonstrations = copy.deepcopy(TOOL_SANDBOX_PTC_FEW_SHOT_MESSAGES)
    if graph_adaptation_mode == "generic":
        prompt += "\n\n" + TOOL_SANDBOX_GRAPH_GUIDANCE
    else:
        for message in demonstrations:
            for call in message.get("tool_calls", ()):
                arguments = json.loads(call["function"]["arguments"])
                for field in ("action", "target", "expected_change"):
                    arguments.pop(field, None)
                call["function"]["arguments"] = json.dumps(arguments)
            if message.get("role") == "tool":
                message["content"] = message["content"].split("\n\nGRAPH_DELTA ", 1)[0]
    return prompt, tuple(demonstrations)


def _toolsandbox_ptc_spec(config: ExperimentConfig) -> dict[str, Any]:
    if config.runtime.graph_adaptation_mode == "off":
        return copy.deepcopy(TOOL_SANDBOX_PTC_SPEC)
    if config.runtime.graph_adaptation_mode != "generic":
        raise ValueError("ToolSandbox graph_adaptation_mode must be off or generic")
    return extend_ptc_spec_with_graph_control(
        TOOL_SANDBOX_PTC_SPEC,
        include_input_artifacts=False,
        target_description="Existing graph node affected by this block; normally task.",
    )


@dataclass(frozen=True)
class ToolSandboxRunSummary:
    selected: int
    processed: int
    runner_failures: int
    execution_failure_scenarios: int
    execution_failure_blocks: int
    mean_similarity: float
    mean_milestone_similarity: float
    mean_minefield_similarity: float

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


def inspect_toolsandbox(config: ExperimentConfig) -> dict[str, Any]:
    return _worker_request(
        config.toolsandbox.worker_command,
        {
            "type": "inspect",
            "root": config.toolsandbox.root,
            "tool_backend": config.toolsandbox.tool_backend,
        },
    )


def run_toolsandbox_benchmark(
    config: ExperimentConfig,
    *,
    limit: int | None = None,
    scenario_names: Sequence[str] = (),
    restart: bool = False,
    progress: Callable[[dict[str, Any]], None] | None = None,
) -> ToolSandboxRunSummary:
    inspection = inspect_toolsandbox(config)
    available = list(inspection["scenario_names"])
    scenario_categories = dict(inspection["scenario_categories"])
    selected = list(scenario_names) if scenario_names else available
    unknown = sorted(set(selected) - set(available))
    if unknown:
        raise ValueError(f"unknown ToolSandbox scenarios: {unknown}")
    if limit is not None:
        selected = selected[:limit]
    if not scenario_names and limit is None and len(selected) != config.toolsandbox.expected_scenarios:
        raise ValueError(
            f"expected {config.toolsandbox.expected_scenarios} scenarios, found {len(selected)}"
        )

    output_path = config.toolsandbox.results_path
    output_path.parent.mkdir(parents=True, exist_ok=True)
    config.toolsandbox.artifact_dir.mkdir(parents=True, exist_ok=True)
    config.toolsandbox.graph_dir.mkdir(parents=True, exist_ok=True)
    if restart and output_path.exists():
        output_path.unlink()
    if restart:
        for name in selected:
            shutil.rmtree(
                config.toolsandbox.artifact_dir / "trajectories" / name,
                ignore_errors=True,
            )
            (config.toolsandbox.graph_dir / f"{name}.json").unlink(missing_ok=True)
    existing = _read_jsonl(output_path)
    seen = {str(item["scenario_name"]) for item in existing if item.get("status") in {"finished", "failed"}}
    pending = [name for name in selected if name not in seen]
    prompt, demonstrations = _toolsandbox_prompt_bundle(
        config.toolsandbox.prompt_variant,
        graph_adaptation_mode=config.runtime.graph_adaptation_mode,
    )
    request_base = {
        "type": "run",
        "root": config.toolsandbox.root,
        "tool_backend": config.toolsandbox.tool_backend,
        "agent_model": dataclasses.asdict(config.model),
        "user_model": dataclasses.asdict(config.user_model),
        "runtime": dataclasses.asdict(config.runtime),
        "agent_mode": (
            "direct_tools"
            if config.toolsandbox.prompt_variant == "toolsandbox-direct-tools-v1"
            else "ptc"
        ),
        "graph_adaptation_mode": config.runtime.graph_adaptation_mode,
        "system_prompt": prompt,
        "demonstration_messages": demonstrations,
        "ptc_tool_spec": (
            None
            if config.toolsandbox.prompt_variant == "toolsandbox-direct-tools-v1"
            else _toolsandbox_ptc_spec(config)
        ),
        "official_commit": inspection["git_commit"],
    }
    lock = threading.Lock()

    def append(record: dict[str, Any]) -> None:
        with lock:
            with output_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, ensure_ascii=False, default=repr) + "\n")
        if progress is not None:
            progress(record)

    def run_one(name: str) -> dict[str, Any]:
        request = dict(request_base)
        request.update(
            {
                "scenario_name": name,
                "output_directory": _as_wsl_path(config.toolsandbox.artifact_dir),
                "graph_path": _as_wsl_path(config.toolsandbox.graph_dir / f"{name}.json"),
            }
        )
        try:
            response = _worker_request(config.toolsandbox.worker_command, request)
            record = {"scenario_name": name, "status": "finished", **response}
        except Exception as exc:
            record = {
                "scenario_name": name,
                "status": "failed",
                "error": f"{type(exc).__name__}: {exc}",
                "categories": scenario_categories[name],
            }
        append(record)
        return record

    if config.toolsandbox.workers <= 1:
        for name in pending:
            run_one(name)
    else:
        with ThreadPoolExecutor(max_workers=config.toolsandbox.workers) as executor:
            futures = [executor.submit(run_one, name) for name in pending]
            for future in as_completed(futures):
                future.result()
    return _summarize(selected, _read_jsonl(output_path))


def evaluate_toolsandbox_benchmark(config: ExperimentConfig) -> dict[str, Any]:
    records = _read_jsonl(config.toolsandbox.results_path)
    terminal = [item for item in records if item.get("status") in {"finished", "failed"}]
    summary = _summarize([str(item["scenario_name"]) for item in terminal], terminal)
    inspection = inspect_toolsandbox(config)
    category_map = dict(inspection["scenario_categories"])
    category_values: dict[str, list[float]] = {}
    for item in terminal:
        categories = list(item.get("categories") or category_map[str(item["scenario_name"])])
        score = float(item.get("similarity", 0.0)) if item.get("status") == "finished" else 0.0
        for category in [*categories, "ALL_CATEGORIES"]:
            if category == "THREE_DISTRACTION_TOOLS" and set(categories) & {
                "TOOL_NAME_SCRAMBLED",
                "TOOL_DESCRIPTION_SCRAMBLED",
                "ARG_DESCRIPTION_SCRAMBLED",
                "ARG_TYPE_SCRAMBLED",
                "ARG_NAME_SCRAMBLED",
            }:
                continue
            category_values.setdefault(str(category), []).append(score)
    categories = {
        name: {"count": len(values), "mean_similarity": sum(values) / len(values)}
        for name, values in sorted(category_values.items())
    }
    report = {
        "summary": summary.to_dict(),
        "categories": categories,
        "official_commit": next(
            (item.get("official_commit") for item in terminal if item.get("official_commit")),
            None,
        ),
    }
    config.toolsandbox.report_path.parent.mkdir(parents=True, exist_ok=True)
    config.toolsandbox.report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return report


def _summarize(selected: Sequence[str], records: Sequence[dict[str, Any]]) -> ToolSandboxRunSummary:
    wanted = set(selected)
    terminal = [
        item
        for item in records
        if item.get("status") in {"finished", "failed"} and item.get("scenario_name") in wanted
    ]
    finished = [item for item in terminal if item.get("status") == "finished"]
    def mean(field: str) -> float:
        return sum(float(item.get(field, 0.0)) for item in finished) / len(selected) if selected else 0.0
    return ToolSandboxRunSummary(
        selected=len(selected),
        processed=len(terminal),
        runner_failures=sum(item.get("status") == "failed" for item in terminal),
        execution_failure_scenarios=sum(int(item.get("execution_failures", 0)) > 0 for item in finished),
        execution_failure_blocks=sum(int(item.get("execution_failures", 0)) for item in finished),
        mean_similarity=mean("similarity"),
        mean_milestone_similarity=mean("milestone_similarity"),
        mean_minefield_similarity=mean("minefield_similarity"),
    )


def _worker_request(command: Sequence[str], request: dict[str, Any]) -> dict[str, Any]:
    if not command:
        raise ValueError("toolsandbox.worker_command is required")
    env = os.environ.copy()
    key_names = {
        str(model.get("api_key_env", ""))
        for model in (request.get("agent_model", {}), request.get("user_model", {}))
        if isinstance(model, dict)
    }
    key_names.add("RAPID_API_KEY")
    shared = [f"{name}/u" for name in sorted(key_names) if name]
    current = env.get("WSLENV", "")
    env["WSLENV"] = ":".join([item for item in [current, *shared] if item])
    completed = subprocess.run(
        list(command),
        input=json.dumps(request, ensure_ascii=False),
        text=True,
        encoding="utf-8",
        errors="replace",
        capture_output=True,
        env=env,
        timeout=max(120.0, float(request.get("runtime", {}).get("task_timeout_seconds", 0)) + 120.0),
        check=False,
    )
    if completed.returncode != 0:
        detail = (completed.stdout or completed.stderr)[-4000:]
        raise RuntimeError(f"ToolSandbox worker failed ({completed.returncode}): {detail}")
    try:
        response = json.loads(completed.stdout)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"invalid ToolSandbox worker response: {completed.stdout[-1000:]}") from exc
    if not isinstance(response, dict) or response.get("type") == "error":
        raise RuntimeError(str(response.get("error", response)))
    return {key: value for key, value in response.items() if key != "type"}


def _as_wsl_path(path: Path) -> str:
    value = str(path.resolve())
    if len(value) >= 3 and value[1:3] == ":\\":
        return f"/mnt/{value[0].lower()}/{value[3:].replace(chr(92), '/')}"
    return value


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
