from __future__ import annotations

import copy
import dataclasses
import hashlib
import json
import shutil
import subprocess
import threading
import urllib.parse
import urllib.request
from collections import defaultdict
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ...agents.direct_tools import DirectToolAgent
from ...agents.original_ptc import PTC_TOOL_SPEC, OriginalPTCAgent
from ...config import ExperimentConfig
from ...graph.adaptation import GoalGraphAdaptation
from ...graph.hooks import GraphAgentHooks, extend_ptc_spec_with_graph_control
from ...model import OpenAIChatModel
from ...runtime.provenance import git_commit, git_dirty
from .runtime import AgentDiffProgramRuntime

AGENT_DIFF_OFFICIAL_COMMIT = "3bb9c40707df23d89e5dbc0e40c424ba38c69ff8"
AGENT_DIFF_DATASET_FILES = {
    "train.jsonl": "612004cbc184569173bd75dd329ccdfbed948474bac92361dd229bfd5a529846",
    "test.jsonl": "032dbd7fc40a052ed36cb1062294bddb254d1c34b6263c91912cf0edd6b93060",
    "all_numbered.jsonl": "d1ea9b5316d9b674cae8e450292cd17572d1aae46575a32bc23f9f4bea044df0",
}
AGENT_DIFF_DATASET_COUNTS = {"train": 179, "test": 45, "all": 224}

SERVICE_CONTEXT = {
    "slack": ("Slack", "https://slack.com/api", ""),
    "box": ("Box", "https://api.box.com/2.0", ""),
    "linear": ("Linear", "https://api.linear.app/graphql", ""),
    "calendar": (
        "Google Calendar",
        "https://www.googleapis.com/calendar/v3",
        "Current date/time: Sunday, June 17, 2018 at 00:01, America/Los_Angeles.",
    ),
}

AGENT_DIFF_BASE_PROMPT = """Complete the user's task against the current Agent-Diff service. The
session context gives the service and its official API base URL. Authentication is intercepted by
the official sandbox, so use placeholder bearer tokens when an endpoint requires one. Interact via
Python requests using the real service URL. Discover uncertain endpoints or identifiers through
API requests and parse responses carefully. Do not assume that a state-changing request succeeded;
inspect its response and verify through the public API when needed. Avoid unrelated state changes.

Your only directly callable model tool is programmatic_tool_call. Its code is sent directly to the
official Agent-Diff Python executor. Treat each PTC block as one semantically coherent phase. Put
mechanically foreseeable requests, loops, filtering, joins, and aggregation in the same block, then
return to the model for a new semantic decision or repair. Python variables do not persist between
blocks, while the isolated service state does persist. Print compact decision-relevant values and
IDs needed by the next turn; never print credentials or large responses. When all requested state
changes have been made and checked, answer concisely without another tool call."""

AGENT_DIFF_GRAPH_GUIDANCE = """Graph control uses GraphPTC's benchmark-neutral contract. Set action
to CONTINUE for a new dependency step, PATCH to correct a failed or unrealized block, and REPLAN
when changing the dependency path. Use task as the target and describe the observable expected
change. After every block, GRAPH_DELTA summarizes recorded API actions, artifacts, state effects,
failures, and the next dependency frontier. The graph never
contains the evaluator's expected assertions or hidden initial database state."""

AGENT_DIFF_USER_PROMPT = "Task: {question}"

AGENT_DIFF_DIRECT_PROMPT = """Complete the user's task against the current Agent-Diff service.
The session context gives the service and its only allowed official API base URL. Use http_request
for individual native HTTP calls. Authentication is intercepted by the official sandbox, so use a
placeholder bearer token when required. Discover uncertain endpoints and identifiers through API
responses, parse failures carefully, and verify state changes through the public API when useful.
Avoid unrelated state changes. When all requested changes are complete, answer concisely."""

AGENT_DIFF_DIRECT_USER_PROMPT = "Task: {task}"

AGENT_DIFF_DIRECT_TOOL_SPECS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "http_request",
            "description": "Send one HTTP request to the current Agent-Diff service API.",
            "parameters": {
                "type": "object",
                "properties": {
                    "method": {
                        "type": "string",
                        "enum": ["GET", "POST", "PUT", "PATCH", "DELETE"],
                    },
                    "url": {"type": "string", "minLength": 1},
                    "headers": {"type": "object"},
                    "params": {"type": "object"},
                    "json_body": {"type": "object"},
                    "data": {"oneOf": [{"type": "string"}, {"type": "object"}]},
                },
                "required": ["method", "url"],
                "additionalProperties": False,
            },
        },
    }
]

AGENT_DIFF_PTC_SPEC = {
    **PTC_TOOL_SPEC,
    "function": {
        **PTC_TOOL_SPEC["function"],
        "description": "Execute one coherent Python requests program in Agent-Diff's official executor.",
        "parameters": {
            **PTC_TOOL_SPEC["function"]["parameters"],
            "properties": {
                "code": {
                    "type": "string",
                    "description": "Exact Python source passed directly to the official PythonExecutorProxy.",
                }
            },
        },
    },
}


def _demo_call() -> dict[str, Any]:
    return {
        "role": "assistant",
        "content": "I will compute the deterministic intermediate result in one compact block.",
        "tool_calls": [
            {
                "id": "agent_diff_demo_1",
                "type": "function",
                "function": {
                    "name": "programmatic_tool_call",
                    "arguments": json.dumps(
                        {
                            "code": (
                                "records = [{'id':'a','active':True},{'id':'b','active':False},"
                                "{'id':'c','active':True}]\n"
                                "active_ids = [row['id'] for row in records if row['active']]\n"
                                "print({'active_ids': active_ids})"
                            ),
                            "action": "CONTINUE",
                            "target": "task",
                            "expected_change": "derive the IDs requiring the next API phase",
                        }
                    ),
                },
            }
        ],
    }


AGENT_DIFF_FEW_SHOT_MESSAGES: tuple[dict[str, Any], ...] = (
    {
        "role": "user",
        "content": (
            "PTC organization demonstration only: identify active IDs in these already available "
            "records; do not access any service."
        ),
    },
    _demo_call(),
    {
        "role": "tool",
        "tool_call_id": "agent_diff_demo_1",
        "content": (
            "{'active_ids': ['a', 'c']}\n\nGRAPH_DELTA "
            '{"declared_action":{"action":"CONTINUE","target":"task"},'
            '"action_verification":{"realized":true}}'
        ),
    },
    {"role": "assistant", "content": "The active IDs are a and c."},
)


def _agentdiff_prompt_bundle(
    variant: str,
    *,
    graph_adaptation_mode: str,
    documentation_condition: str,
) -> tuple[str, tuple[dict[str, Any], ...]]:
    if variant == "agent-diff-direct-tools-v1":
        if documentation_condition != "no-docs":
            raise ValueError(
                "the frozen Agent-Diff comparison uses documentation_condition='no-docs'"
            )
        if graph_adaptation_mode != "off":
            raise ValueError(
                "agent-diff-direct-tools-v1 requires graph_adaptation_mode='off'"
            )
        return AGENT_DIFF_DIRECT_PROMPT, ()
    if variant != "agent-diff-ptc-fewshot":
        raise ValueError(f"unsupported Agent-Diff prompt variant: {variant!r}")
    if documentation_condition != "no-docs":
        raise ValueError("the frozen Agent-Diff comparison uses documentation_condition='no-docs'")
    if graph_adaptation_mode not in {"off", "generic"}:
        raise ValueError("Agent-Diff graph_adaptation_mode must be off or generic")
    prompt = AGENT_DIFF_BASE_PROMPT
    demonstrations = copy.deepcopy(AGENT_DIFF_FEW_SHOT_MESSAGES)
    if graph_adaptation_mode == "generic":
        prompt += "\n\n" + AGENT_DIFF_GRAPH_GUIDANCE
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


def _agentdiff_direct_functions(
    runtime: AgentDiffProgramRuntime,
    service: str,
) -> dict[str, Callable[..., Any]]:
    if service not in SERVICE_CONTEXT:
        raise ValueError(f"unsupported Agent-Diff service: {service!r}")
    allowed_base = SERVICE_CONTEXT[service][1]

    def http_request(
        method: str,
        url: str,
        headers: dict[str, Any] | None = None,
        params: dict[str, Any] | None = None,
        json_body: Any = None,
        data: str | dict[str, Any] | None = None,
    ) -> Any:
        normalized_method = str(method).upper()
        if normalized_method not in {"GET", "POST", "PUT", "PATCH", "DELETE"}:
            raise ValueError(f"unsupported HTTP method: {method!r}")
        _validate_agentdiff_url(str(url), allowed_base)
        payload = {
            "method": normalized_method,
            "url": str(url),
            "headers": headers or {},
            "params": params or {},
            "json_body": json_body,
            "data": data,
        }
        encoded = json.dumps(payload, ensure_ascii=True, separators=(",", ":"))
        code = "\n".join(
            (
                "import json, requests",
                f"_request = json.loads({encoded!r})",
                "_response = requests.request(",
                "    method=_request['method'],",
                "    url=_request['url'],",
                "    headers=_request['headers'],",
                "    params=_request['params'],",
                "    json=_request['json_body'],",
                "    data=_request['data'],",
                "    timeout=30,",
                ")",
                "try:",
                "    _body = _response.json()",
                "except Exception:",
                "    _body = _response.text",
                "print(json.dumps({'status_code': _response.status_code, 'body': _body}, ensure_ascii=False, default=repr))",
            )
        )
        result = runtime.execute(code)
        if result.timed_out:
            raise TimeoutError("Agent-Diff HTTP request timed out")
        if result.return_code != 0:
            raise RuntimeError(result.stderr or result.stdout or "Agent-Diff HTTP request failed")
        output = result.stdout.strip()
        try:
            return json.loads(output)
        except json.JSONDecodeError:
            return output

    return {"http_request": http_request}


def _validate_agentdiff_url(url: str, allowed_base: str) -> None:
    parsed = urllib.parse.urlsplit(url)
    allowed = urllib.parse.urlsplit(allowed_base)
    if parsed.scheme != allowed.scheme or parsed.netloc != allowed.netloc:
        raise ValueError(f"URL must use the current service base: {allowed_base}")
    allowed_path = allowed.path.rstrip("/")
    requested_path = parsed.path.rstrip("/")
    if requested_path != allowed_path and not requested_path.startswith(allowed_path + "/"):
        raise ValueError(f"URL must use the current service base: {allowed_base}")


def _agentdiff_ptc_spec(config: ExperimentConfig) -> dict[str, Any]:
    if config.runtime.graph_adaptation_mode == "off":
        return copy.deepcopy(AGENT_DIFF_PTC_SPEC)
    if config.runtime.graph_adaptation_mode != "generic":
        raise ValueError("Agent-Diff graph_adaptation_mode must be off or generic")
    return extend_ptc_spec_with_graph_control(
        AGENT_DIFF_PTC_SPEC,
        include_input_artifacts=False,
        target_description="Use task for this Agent-Diff episode.",
    )


class _EmptySearchTools:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []


@dataclass(frozen=True)
class AgentDiffRunSummary:
    selected: int
    processed: int
    passed: int
    pass_rate: float
    assertion_weighted_score: float
    satisfied_assertions: int
    total_assertions: int
    unexpected_side_effects: int
    execution_failure_tasks: int
    execution_failure_blocks: int
    incomplete_tasks: int
    task_timeouts: int
    evaluator_failures: int
    runner_failures: int
    cleanup_failures: int
    run_signature: str

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


class _ProgressLog:
    def __init__(self, path: Path) -> None:
        self.path = path
        self._lock = threading.Lock()
        path.parent.mkdir(parents=True, exist_ok=True)

    def __call__(self, record: Mapping[str, Any]) -> None:
        payload = {
            key: record.get(key)
            for key in ("task_id", "trial", "service", "status", "started_at", "finished_at")
            if key in record
        }
        with self._lock:
            with self.path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(payload, ensure_ascii=False, default=repr) + "\n")


def download_agent_diff(config: ExperimentConfig) -> Path:
    target = config.agent_diff.dataset_dir
    target.mkdir(parents=True, exist_ok=True)
    commit = config.agent_diff.official_commit
    if commit != AGENT_DIFF_OFFICIAL_COMMIT:
        raise ValueError("Agent-Diff official commit does not match the audited dataset constants")
    for name, expected_hash in AGENT_DIFF_DATASET_FILES.items():
        path = target / name
        if not path.exists() or _file_sha256(path) != expected_hash:
            url = (
                "https://raw.githubusercontent.com/agent-diff-bench/agent-diff/"
                f"{commit}/datasets/agent-diff-bench/{name}"
            )
            with urllib.request.urlopen(url, timeout=120) as response:
                payload = response.read()
            if hashlib.sha256(payload).hexdigest() != expected_hash:
                raise ValueError(f"Agent-Diff dataset checksum mismatch for {name}")
            path.write_bytes(payload)
        _verify_dataset_file(path, expected_hash)
    manifest = {
        "official_commit": commit,
        "files": AGENT_DIFF_DATASET_FILES,
        "counts": AGENT_DIFF_DATASET_COUNTS,
    }
    (target / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return target


def inspect_agent_diff(config: ExperimentConfig) -> dict[str, Any]:
    inspection = _worker_request(
        config.agent_diff.worker_command,
        {"type": "inspect", "official_commit": config.agent_diff.official_commit},
        timeout=120,
        env_names=(config.agent_diff.api_key_env, config.agent_diff.base_url_env),
    )
    tasks = _load_tasks(config)
    return {
        **{key: value for key, value in inspection.items() if key != "type"},
        "dataset_split": config.agent_diff.dataset_split,
        "dataset_count": len(tasks),
        "dataset_sha256": _dataset_hash(config),
        "services": dict(sorted(_count_by(tasks, "service").items())),
    }


def run_agent_diff_benchmark(
    config: ExperimentConfig,
    *,
    limit: int | None = None,
    task_ids: Sequence[str] = (),
    trials: Sequence[int] = (),
    restart: bool = False,
    progress: Callable[[Mapping[str, Any]], None] | None = None,
) -> AgentDiffRunSummary:
    app = config.agent_diff
    if app.official_commit != AGENT_DIFF_OFFICIAL_COMMIT:
        raise ValueError("Agent-Diff official commit differs from the frozen protocol")
    prompt, demonstrations = _agentdiff_prompt_bundle(
        app.prompt_variant,
        graph_adaptation_mode=config.runtime.graph_adaptation_mode,
        documentation_condition=app.documentation_condition,
    )
    tasks = _load_tasks(config)
    by_id = {str(task["test_id"]): task for task in tasks}
    selected_tasks = [by_id[value] for value in task_ids] if task_ids else tasks
    unknown = sorted(set(task_ids) - set(by_id))
    if unknown:
        raise ValueError(f"unknown Agent-Diff task IDs: {unknown}")
    if limit is not None:
        selected_tasks = selected_tasks[:limit]
    if not task_ids and limit is None and len(selected_tasks) != app.expected_tasks:
        raise ValueError(f"expected {app.expected_tasks} tasks, found {len(selected_tasks)}")
    selected_trials = list(trials) if trials else list(range(app.trials))
    if any(value < 0 or value >= app.trials for value in selected_trials):
        raise ValueError(f"trial must be in [0, {app.trials - 1}]")
    selected = [(str(task["test_id"]), trial) for task in selected_tasks for trial in selected_trials]
    inspection = inspect_agent_diff(config)
    signature_payload = _signature_payload(config, inspection, selected_tasks, selected_trials, prompt, demonstrations)
    signature = _sha256(signature_payload)

    for path in (app.results_path.parent, app.artifact_dir, app.graph_dir, app.progress_path.parent):
        path.mkdir(parents=True, exist_ok=True)
    if restart:
        for path in (app.results_path, app.progress_path):
            path.unlink(missing_ok=True)
        for task_id, trial in selected:
            shutil.rmtree(app.artifact_dir / task_id / f"trial-{trial}", ignore_errors=True)
            (app.graph_dir / f"{task_id}.trial-{trial}.json").unlink(missing_ok=True)
    existing = _read_jsonl(app.results_path)
    if any(item.get("run_signature") != signature for item in existing):
        raise ValueError("existing Agent-Diff results use another run signature")
    seen = {
        (str(item.get("task_id")), int(item.get("trial", -1)))
        for item in existing
        if item.get("status") in {"finished", "failed"}
    }
    pending = [(by_id[task_id], trial) for task_id, trial in selected if (task_id, trial) not in seen]
    write_lock = threading.Lock()
    progress_callback = progress or _ProgressLog(app.progress_path)

    def append(record: dict[str, Any]) -> None:
        with write_lock:
            with app.results_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, ensure_ascii=False, default=repr) + "\n")
        progress_callback(record)

    def run_one(task: dict[str, Any], trial: int) -> dict[str, Any]:
        task_id = str(task["test_id"])
        service = str(task["service"])
        runtime = AgentDiffProgramRuntime(
            worker_command=app.worker_command,
            task=task,
            trial=trial,
            official_commit=app.official_commit,
            timeout_seconds=config.runtime.code_timeout_seconds,
        )
        controller: GoalGraphAdaptation | None = None
        agent_result = None
        evaluation: dict[str, Any] | None = None
        evaluator_error: str | None = None
        record: dict[str, Any]
        append({"task_id": task_id, "trial": trial, "service": service, "status": "started", "run_signature": signature})
        try:
            metadata = runtime.metadata
            if config.runtime.graph_adaptation_mode == "generic":
                controller = GoalGraphAdaptation({}, {}, task=str(task["question"]), expose_graph_api=False)
                hooks = GraphAgentHooks.from_controller(controller)
                hook_kwargs = hooks.agent_kwargs()
                hook_kwargs["runtime_functions"] = ()
            else:
                hook_kwargs = {"runtime_functions": ()}
            model = OpenAIChatModel(config.model, config.require_api_key(config.model.api_key_env))
            if app.prompt_variant == "agent-diff-direct-tools-v1":
                agent = DirectToolAgent(
                    model=model,
                    runtime=config.runtime,
                    system_prompt=_service_prompt(prompt, service),
                    user_prompt_template=AGENT_DIFF_DIRECT_USER_PROMPT,
                    functions=_agentdiff_direct_functions(runtime, service),
                    tool_specs=AGENT_DIFF_DIRECT_TOOL_SPECS,
                )
            else:
                agent = OriginalPTCAgent(
                    model=model,
                    search_tools=_EmptySearchTools(),  # type: ignore[arg-type]
                    runtime=config.runtime,
                    system_prompt=_service_prompt(prompt, service),
                    user_prompt_template=AGENT_DIFF_USER_PROMPT,
                    ptc_tool_spec=_agentdiff_ptc_spec(config),
                    demonstration_messages=demonstrations,
                    program_runtime=runtime,
                    **hook_kwargs,
                )
            agent_result = agent.run(str(task["question"]))
            if controller is not None:
                controller.finish(answered=agent_result.status == "success")
            try:
                evaluation = runtime.evaluate()
            except Exception as exc:
                evaluator_error = f"{type(exc).__name__}: {exc}"
            artifact_path = app.artifact_dir / task_id / f"trial-{trial}" / "official.json"
            artifact_path.parent.mkdir(parents=True, exist_ok=True)
            artifact_path.write_text(
                json.dumps(
                    {
                        "task": task,
                        "runtime": metadata,
                        "agent": agent_result.to_dict(),
                        "official_evaluation": evaluation,
                        "official_result": runtime.official_result,
                        "official_end_run": runtime.official_end_run,
                        "official_diff": runtime.official_diff,
                    },
                    ensure_ascii=False,
                    indent=2,
                    default=repr,
                ),
                encoding="utf-8",
            )
            graph_path: Path | None = None
            if controller is not None:
                graph_path = app.graph_dir / f"{task_id}.trial-{trial}.json"
                graph_path.write_text(
                    json.dumps(controller.graph_artifact(), ensure_ascii=False, indent=2, default=repr),
                    encoding="utf-8",
                )
            record = {
                "task_id": task_id,
                "trial": trial,
                "service": service,
                "task_horizon": task.get("task_horizon"),
                "operation_type": task.get("operation_type"),
                "entity_scope": task.get("entity_scope"),
                "information_availability": task.get("information_availability"),
                "prompt_ambiguity": task.get("prompt_ambiguity"),
                "status": "finished",
                "run_signature": signature,
                "agent": agent_result.to_dict(),
                "execution_failures": _agent_execution_failures(agent_result),
                "official_evaluation": evaluation,
                "evaluator_error": evaluator_error,
                "artifact_path": str(artifact_path),
                "graph_path": str(graph_path) if graph_path is not None else None,
                "graph_telemetry": controller.telemetry() if controller is not None else None,
                "total_assertions": len(_answer(task).get("assertions", [])),
            }
        except Exception as exc:
            record = {
                "task_id": task_id,
                "trial": trial,
                "service": service,
                "status": "failed",
                "run_signature": signature,
                "error": f"{type(exc).__name__}: {exc}",
                "agent": agent_result.to_dict() if agent_result is not None else None,
                "execution_failures": _agent_execution_failures(agent_result) if agent_result else 0,
                "official_evaluation": evaluation,
                "evaluator_error": evaluator_error,
                "total_assertions": len(_answer(task).get("assertions", [])),
                "graph_telemetry": controller.telemetry() if controller is not None else None,
            }
        finally:
            try:
                runtime.close()
            except Exception as exc:
                record["status"] = "failed"
                record["close_error"] = f"{type(exc).__name__}: {exc}"
        final_runtime = runtime.telemetry()
        if not final_runtime.get("environment_deleted") or not final_runtime.get("termination_confirmed"):
            record["status"] = "failed"
            record.setdefault("close_error", final_runtime.get("close_error") or "environment deletion was not confirmed")
        record["runtime_final"] = final_runtime
        append(record)
        return record

    with ThreadPoolExecutor(max_workers=max(1, app.workers)) as executor:
        futures = [executor.submit(run_one, task, trial) for task, trial in pending]
        for future in as_completed(futures):
            future.result()

    terminal = {
        (str(item.get("task_id")), int(item.get("trial", -1))): item
        for item in _read_jsonl(app.results_path)
        if item.get("status") in {"finished", "failed"}
    }
    records = [terminal[key] for key in selected if key in terminal]
    summary = _summarize(selected, records, signature)
    report = _build_report(config, summary, signature_payload, records)
    app.report_path.parent.mkdir(parents=True, exist_ok=True)
    app.report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2, default=repr), encoding="utf-8")
    return summary


def evaluate_agent_diff_benchmark(config: ExperimentConfig) -> dict[str, Any]:
    records = [item for item in _read_jsonl(config.agent_diff.results_path) if item.get("status") in {"finished", "failed"}]
    if not records:
        raise ValueError("Agent-Diff results do not exist")
    signatures = {str(item.get("run_signature", "")) for item in records}
    if len(signatures) != 1:
        raise ValueError("Agent-Diff results contain multiple run signatures")
    selected = [(str(item["task_id"]), int(item["trial"])) for item in records]
    summary = _summarize(selected, records, next(iter(signatures)))
    report = _build_report(config, summary, None, records)
    config.agent_diff.report_path.parent.mkdir(parents=True, exist_ok=True)
    config.agent_diff.report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2, default=repr), encoding="utf-8")
    return report


def _summarize(
    selected: Sequence[tuple[str, int]], records: Sequence[dict[str, Any]], signature: str
) -> AgentDiffRunSummary:
    finished_evaluations = [
        item["official_evaluation"]
        for item in records
        if item.get("status") == "finished" and isinstance(item.get("official_evaluation"), Mapping)
    ]
    known_totals = [int(value.get("total_assertions", 0)) for value in finished_evaluations if int(value.get("total_assertions", 0)) > 0]
    default_total = round(sum(known_totals) / len(known_totals)) if known_totals else 1
    total_assertions = 0
    satisfied = 0
    passed = 0
    unexpected = 0
    for item in records:
        evaluation = item.get("official_evaluation") if item.get("status") == "finished" else None
        if isinstance(evaluation, Mapping):
            total = int(evaluation.get("total_assertions") or item.get("total_assertions") or default_total)
            total_assertions += total
            satisfied += int(evaluation.get("satisfied_assertions", 0))
            passed += int(bool(evaluation.get("passed")))
        else:
            total_assertions += int(item.get("total_assertions") or default_total)
    missing = max(0, len(selected) - len(records))
    total_assertions += missing * default_total
    denominator = len(selected)
    return AgentDiffRunSummary(
        selected=denominator,
        processed=len(records),
        passed=passed,
        pass_rate=passed / denominator if denominator else 0.0,
        assertion_weighted_score=satisfied / total_assertions if total_assertions else 0.0,
        satisfied_assertions=satisfied,
        total_assertions=total_assertions,
        unexpected_side_effects=unexpected,
        execution_failure_tasks=sum(bool(item.get("execution_failures")) for item in records),
        execution_failure_blocks=sum(int(item.get("execution_failures") or 0) for item in records),
        incomplete_tasks=sum(
            item.get("status") == "finished"
            and (item.get("agent") or {}).get("status") != "success"
            for item in records
        ),
        task_timeouts=sum((item.get("agent") or {}).get("finish_reason") == "task_timeout" for item in records),
        evaluator_failures=sum(bool(item.get("evaluator_error")) for item in records),
        runner_failures=sum(item.get("status") == "failed" for item in records) + missing,
        cleanup_failures=sum(bool(item.get("close_error")) for item in records),
        run_signature=signature,
    )


def _build_report(
    config: ExperimentConfig,
    summary: AgentDiffRunSummary,
    signature_payload: dict[str, Any] | None,
    records: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in records:
        groups[str(item.get("service", "unknown"))].append(item)
    by_service = {
        name: _summarize(
            [(str(item["task_id"]), int(item["trial"])) for item in values],
            values,
            summary.run_signature,
        ).to_dict()
        for name, values in sorted(groups.items())
    }
    return {
        "summary": summary.to_dict(),
        "by_service": by_service,
        "run_signature_payload": signature_payload,
        "resolved_config": _resolved_config(config),
        "resolved_config_sha256": _sha256(_resolved_config(config)),
        "records": list(records),
    }


def _service_prompt(base: str, service: str) -> str:
    if service not in SERVICE_CONTEXT:
        raise ValueError(f"unsupported Agent-Diff service: {service!r}")
    name, url, extra = SERVICE_CONTEXT[service]
    context = f"Current service: {name}\nOfficial API base URL: {url}"
    if extra:
        context += "\n" + extra
    return base + "\n\n" + context


def _agent_execution_failures(agent_result: Any) -> int:
    runtime_session = getattr(agent_result, "runtime_session", {})
    if runtime_session.get("mode") == "direct_tool_calling":
        return int(runtime_session.get("tool_errors", 0))
    return sum(not block.success for block in agent_result.blocks)


def _load_tasks(config: ExperimentConfig) -> list[dict[str, Any]]:
    split = config.agent_diff.dataset_split
    if split not in AGENT_DIFF_DATASET_COUNTS:
        raise ValueError("Agent-Diff dataset_split must be train, test, or all")
    filename = "all_numbered.jsonl" if split == "all" else f"{split}.jsonl"
    path = config.agent_diff.dataset_dir / filename
    expected_hash = AGENT_DIFF_DATASET_FILES[filename]
    _verify_dataset_file(path, expected_hash)
    tasks = _read_jsonl(path)
    if len(tasks) != AGENT_DIFF_DATASET_COUNTS[split]:
        raise ValueError(f"Agent-Diff {split} split has {len(tasks)} tasks")
    for task in tasks:
        task["info"] = json.loads(task["info"]) if isinstance(task.get("info"), str) else task["info"]
        task["answer"] = json.loads(task["answer"]) if isinstance(task.get("answer"), str) else task["answer"]
    return tasks


def _answer(task: Mapping[str, Any]) -> dict[str, Any]:
    value = task.get("answer", {})
    return json.loads(value) if isinstance(value, str) else dict(value)


def _verify_dataset_file(path: Path, expected_hash: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Agent-Diff dataset is missing: {path}; run download-agent-diff")
    if _file_sha256(path) != expected_hash:
        raise ValueError(f"Agent-Diff dataset checksum mismatch: {path}")


def _dataset_hash(config: ExperimentConfig) -> str:
    split = config.agent_diff.dataset_split
    filename = "all_numbered.jsonl" if split == "all" else f"{split}.jsonl"
    return _file_sha256(config.agent_diff.dataset_dir / filename)


def _count_by(tasks: Sequence[Mapping[str, Any]], field: str) -> dict[str, int]:
    values: dict[str, int] = defaultdict(int)
    for task in tasks:
        values[str(task.get(field, "unknown"))] += 1
    return dict(values)


def _worker_request(
    command: Sequence[str],
    payload: dict[str, Any],
    *,
    timeout: float,
    env_names: Sequence[str] = ("AGENT_DIFF_API_KEY", "AGENT_DIFF_BASE_URL"),
) -> dict[str, Any]:
    if not command:
        raise ValueError("[agent_diff].worker_command is required")
    env = dict(__import__("os").environ)
    shared = [f"{name}/u" for name in env_names if name]
    current = env.get("WSLENV", "")
    env["WSLENV"] = ":".join([value for value in (current, *shared) if value])
    completed = subprocess.run(
        tuple(command),
        input=json.dumps(payload, ensure_ascii=True) + "\n",
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=timeout,
        env=env,
        check=False,
    )
    lines = [line for line in completed.stdout.splitlines() if line.strip()]
    if completed.returncode != 0 or not lines:
        detail = (completed.stdout or completed.stderr)[-4000:]
        raise RuntimeError(f"Agent-Diff worker failed ({completed.returncode}): {detail}")
    response = json.loads(lines[-1])
    if response.get("type") == "error":
        raise RuntimeError(str(response.get("error")))
    return response


def _signature_payload(
    config: ExperimentConfig,
    inspection: Mapping[str, Any],
    tasks: Sequence[Mapping[str, Any]],
    trials: Sequence[int],
    prompt: str,
    demonstrations: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    model = dataclasses.asdict(config.model)
    runtime = dataclasses.asdict(config.runtime)
    app = dataclasses.asdict(config.agent_diff)
    for key in ("dataset_dir", "results_path", "report_path", "artifact_dir", "graph_dir", "progress_path"):
        app[key] = str(app[key])
    app["worker_command"] = list(app["worker_command"])
    return {
        "schema_version": 1,
        "benchmark": "agent_diff",
        "model": model,
        "runtime": runtime,
        "agent_diff": app,
        "prompt": {
            "system_prompt_sha256": _sha256(prompt),
            "demonstrations_sha256": _sha256(demonstrations),
            "tool_spec_sha256": _sha256(
                AGENT_DIFF_DIRECT_TOOL_SPECS
                if config.agent_diff.prompt_variant == "agent-diff-direct-tools-v1"
                else _agentdiff_ptc_spec(config)
            ),
        },
        "environment": dict(inspection),
        "task_ids": [str(task["test_id"]) for task in tasks],
        "trials": list(trials),
        "dataset_sha256": _dataset_hash(config),
        "graphptc_commit": _git_commit(),
        "graphptc_git_dirty": _git_dirty(),
        "graphptc_source_hash": _source_hash(),
    }


def _resolved_config(config: ExperimentConfig) -> dict[str, Any]:
    app = dataclasses.asdict(config.agent_diff)
    for key in ("dataset_dir", "results_path", "report_path", "artifact_dir", "graph_dir", "progress_path"):
        app[key] = str(app[key])
    app["worker_command"] = list(app["worker_command"])
    return {"model": dataclasses.asdict(config.model), "runtime": dataclasses.asdict(config.runtime), "agent_diff": app}


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _sha256(value: Any) -> str:
    payload = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False, sort_keys=True, default=repr)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _git_commit() -> str:
    return git_commit()


def _git_dirty() -> bool:
    return git_dirty()


def _source_hash() -> str:
    digest = hashlib.sha256()
    source_root = Path(__file__).resolve().parents[2]
    for path in sorted(source_root.rglob("*.py")):
        digest.update(path.relative_to(source_root).as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()
