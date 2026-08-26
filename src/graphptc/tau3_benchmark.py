from __future__ import annotations

import copy
import dataclasses
import hashlib
import json
import os
import re
import subprocess
import threading
import time
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .config import ExperimentConfig
from .graph_agent import extend_ptc_spec_with_graph_control
from .ptc import PTC_TOOL_SPEC

TAU3_OFFICIAL_COMMIT = "fc0055dc4e0a316c3f83133267fbd6faaa770992"
TAU3_OFFICIAL_VERSION = "1.0.1"
TAU3_TEXT_DOMAINS = ("airline", "retail", "telecom")


def _safe_task_key(task_id: str) -> str:
    prefix = re.sub(r"[^A-Za-z0-9_-]+", "-", task_id).strip("-_")[:80]
    digest = hashlib.sha256(task_id.encode("utf-8")).hexdigest()[:12]
    return f"{prefix or 'task'}-{digest}"


def _tau3_agent_name(graph_adaptation_mode: str) -> str:
    if graph_adaptation_mode == "generic":
        return "graphptc"
    if graph_adaptation_mode == "off":
        return "fewshot_ptc"
    raise ValueError("tau3 graph_adaptation_mode must be off or generic")

TAU3_BASE_PROMPT = """You are the service agent in an official tau3-bench text conversation. The
authoritative domain policy is supplied above this prompt and must be followed exactly. Available
environment functions and their schemas are also supplied at runtime. Do not invent tool names or
parameters, and do not claim an action succeeded unless the official environment returned success.

Your only directly callable model tool is programmatic_tool_call. Write one semantically coherent phase
as a Python program per PTC block. The block can call the supplied environment functions multiple times,
including sequentially and conditionally; each call is executed by the official environment and
the program resumes with its result. Put mechanically foreseeable calls, loops, filtering, joins,
and checks in one block. Print only compact decision-relevant results needed for your next semantic
decision. Python variables do not persist between PTC blocks, while official service state does.
Reply to the user in plain text when you need information or when the task is complete. A message
must contain either a user-facing reply or one programmatic_tool_call, never both."""

TAU3_GRAPH_GUIDANCE = """Graph control uses GraphPTC's benchmark-neutral contract. Set action to
CONTINUE for a new dependency step, PATCH to correct a failed or unrealized block, and REPLAN when
changing the dependency path. Use task as the target and describe the observable expected change.
After every block, GRAPH_DELTA summarizes official API calls, output artifacts, state effects,
failures, and the next dependency frontier. No graph inspection API is available. The graph never
contains hidden task assertions, database state, or task answers."""

TAU3_PTC_SPEC = {
    **PTC_TOOL_SPEC,
    "function": {
        **PTC_TOOL_SPEC["function"],
        "description": (
            "Execute one coherent Python program whose environment API calls are routed through "
            "the official tau3-bench orchestrator."
        ),
        "parameters": {
            **PTC_TOOL_SPEC["function"]["parameters"],
            "properties": {
                "code": {
                    "type": "string",
                    "description": "Python source; official environment functions are globals.",
                }
            },
        },
    },
}

_FEWSHOT: tuple[dict[str, Any], ...] = (
    {
        "role": "user",
        "content": "PTC organization example only: inspect two known records and print the open ID.",
    },
    {
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {
                "id": "tau3_demo_1",
                "type": "function",
                "function": {
                    "name": "programmatic_tool_call",
                    "arguments": json.dumps(
                        {
                            "code": (
                                "rows = [lookup_record(record_id=value) for value in ['a', 'b']]\n"
                                "print([row['id'] for row in rows if row['status'] == 'open'])"
                            ),
                            "action": "CONTINUE",
                            "target": "task",
                            "expected_change": "identify the open record needed by the next step",
                        }
                    ),
                },
            }
        ],
    },
    {
        "role": "tool",
        "tool_call_id": "tau3_demo_1",
        "content": (
            "['b']\n\nGRAPH_DELTA "
            '{"declared_action":{"action":"CONTINUE","target":"task"},'
            '"action_verification":{"realized":true}}'
        ),
    },
    {"role": "assistant", "content": "The open record is b."},
)


def _tau3_prompt_bundle(
    variant: str, *, graph_adaptation_mode: str
) -> tuple[str, tuple[dict[str, Any], ...]]:
    if variant != "tau3-ptc-fewshot":
        raise ValueError(f"unsupported tau3 prompt variant: {variant!r}")
    if graph_adaptation_mode not in {"off", "generic"}:
        raise ValueError("tau3 graph_adaptation_mode must be off or generic")
    prompt = TAU3_BASE_PROMPT
    demonstrations = copy.deepcopy(_FEWSHOT)
    if graph_adaptation_mode == "generic":
        prompt += "\n\n" + TAU3_GRAPH_GUIDANCE
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


def _tau3_ptc_spec(config: ExperimentConfig) -> dict[str, Any]:
    if config.runtime.graph_inspection_enabled:
        raise ValueError("tau3 evaluation does not expose graph inspection")
    if config.runtime.graph_adaptation_mode == "off":
        return copy.deepcopy(TAU3_PTC_SPEC)
    if config.runtime.graph_adaptation_mode != "generic":
        raise ValueError("tau3 graph_adaptation_mode must be off or generic")
    return extend_ptc_spec_with_graph_control(
        TAU3_PTC_SPEC,
        include_input_artifacts=False,
        include_inspection=False,
        target_description="Use task for this tau3 episode.",
    )


@dataclass(frozen=True)
class Tau3RunSummary:
    selected: int
    processed: int
    passed: int
    pass_hat_1: float
    mean_reward: float
    execution_failure_tasks: int
    execution_failure_blocks: int
    incomplete_tasks: int
    evaluator_failures: int
    runner_failures: int
    runner_retry_tasks: int
    runner_retry_attempts: int
    by_domain: dict[str, dict[str, Any]]
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
            for key in ("domain", "task_id", "trial", "status", "started_at", "finished_at")
            if key in record
        }
        with self._lock, self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, ensure_ascii=False, default=repr) + "\n")


def inspect_tau3(config: ExperimentConfig) -> dict[str, Any]:
    return _worker_request(
        config.tau3.worker_command,
        {
            "type": "inspect",
            "root": config.tau3.root,
            "domains": list(config.tau3.domains),
            "task_split_name": config.tau3.task_split_name,
        },
        timeout=300,
    )


def validate_tau3_alignment(config: ExperimentConfig, inspection: Mapping[str, Any]) -> None:
    app = config.tau3
    if app.official_commit != TAU3_OFFICIAL_COMMIT:
        raise ValueError("configured tau3 commit differs from the frozen official release")
    if inspection.get("official_commit") != app.official_commit:
        raise ValueError("installed tau3 commit differs from the frozen official release")
    if inspection.get("package_version") != TAU3_OFFICIAL_VERSION:
        raise ValueError("installed tau3 package version is not 1.0.1")
    if not inspection.get("data_verified"):
        raise ValueError("official tau3 data verification did not pass")
    if tuple(app.domains) != TAU3_TEXT_DOMAINS:
        raise ValueError("formal text evaluation requires airline, retail, and telecom")
    domains = inspection.get("domains") or {}
    for domain in app.domains:
        if not (domains.get(domain) or {}).get("task_ids"):
            raise ValueError(f"official base split is empty for {domain}")
    defaults = inspection.get("official_defaults") or {}
    expected = {
        "max_steps": app.max_steps,
        "max_errors": app.max_errors,
        "seed": app.seed,
        "max_concurrency": app.workers,
        "agent_temperature": config.model.temperature,
        "user_temperature": 0.0,
        "enforce_communication_protocol": app.enforce_communication_protocol,
        "max_retries": app.task_max_retries,
        "retry_delay": app.retry_delay_seconds,
    }
    for key, value in expected.items():
        if defaults.get(key) != value:
            raise ValueError(f"official {key} mismatch: expected {value}, got {defaults.get(key)}")
    if app.task_split_name != "base" or app.trials != 4 or app.workers != 3:
        raise ValueError("formal tau3 config must use base split, 4 trials, and concurrency 3")


def run_tau3_benchmark(
    config: ExperimentConfig,
    *,
    limit: int | None = None,
    domains: Sequence[str] = (),
    task_ids: Sequence[str] = (),
    trials: Sequence[int] = (),
    restart: bool = False,
    progress: Callable[[Mapping[str, Any]], None] | None = None,
) -> Tau3RunSummary:
    inspection = inspect_tau3(config)
    validate_tau3_alignment(config, inspection)
    app = config.tau3
    prompt, demonstrations = _tau3_prompt_bundle(
        app.prompt_variant, graph_adaptation_mode=config.runtime.graph_adaptation_mode
    )
    chosen_domains = tuple(domains) if domains else app.domains
    unknown_domains = sorted(set(chosen_domains) - set(app.domains))
    if unknown_domains:
        raise ValueError(f"unknown tau3 domains: {unknown_domains}")
    available = {
        domain: [str(value) for value in inspection["domains"][domain]["task_ids"]]
        for domain in chosen_domains
    }
    selected_tasks = [
        (domain, task_id)
        for domain, ids in available.items()
        for task_id in ids
        if not task_ids or task_id in task_ids
    ]
    if task_ids:
        known = {task_id for _, task_id in selected_tasks}
        unknown = sorted(set(task_ids) - known)
        if unknown:
            raise ValueError(f"unknown tau3 task IDs for selected domains: {unknown}")
    if limit is not None:
        selected_tasks = selected_tasks[:limit]
    selected_trials = list(trials) if trials else list(range(app.trials))
    if any(value < 0 or value >= app.trials for value in selected_trials):
        raise ValueError(f"trial must be in [0, {app.trials - 1}]")
    selected = [(*task, trial) for task in selected_tasks for trial in selected_trials]
    signature = _hash(
        {
            "official": inspection,
            "config": _public_config(config),
            "prompt": prompt,
            "demonstrations": demonstrations,
            "selected": selected,
            "ptc_spec": _tau3_ptc_spec(config),
        }
    )
    for path in (app.results_path.parent, app.artifact_dir, app.graph_dir, app.progress_path.parent):
        path.mkdir(parents=True, exist_ok=True)
    if restart:
        for path in (app.results_path, app.progress_path):
            path.unlink(missing_ok=True)
    existing = _read_jsonl(app.results_path)
    if any(item.get("run_signature") != signature for item in existing):
        raise ValueError("existing tau3 results use another run signature")
    seen = {
        (str(item.get("domain")), str(item.get("task_id")), int(item.get("trial", -1)))
        for item in existing
        if item.get("status") in {"finished", "failed"}
    }
    pending = [item for item in selected if item not in seen]
    write_lock = threading.Lock()
    progress_callback = progress or _ProgressLog(app.progress_path)

    def append(record: dict[str, Any]) -> None:
        with write_lock, app.results_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False, default=repr) + "\n")
        progress_callback(record)

    def run_one(domain: str, task_id: str, trial: int) -> dict[str, Any]:
        append({"domain": domain, "task_id": task_id, "trial": trial, "status": "started", "run_signature": signature})
        task_key = _safe_task_key(task_id)
        request = {
            "type": "run",
            "domain": domain,
            "task_id": task_id,
            "trial": trial,
            "seed": app.seed + trial,
            "task_split_name": app.task_split_name,
            "max_steps": app.max_steps,
            "max_errors": app.max_errors,
            "enforce_communication_protocol": app.enforce_communication_protocol,
            "timeout": config.runtime.task_timeout_seconds,
            "system_prompt": prompt,
            "demonstration_messages": demonstrations,
            "ptc_tool_spec": _tau3_ptc_spec(config),
            "graph_adaptation_mode": config.runtime.graph_adaptation_mode,
            "agent_name": _tau3_agent_name(config.runtime.graph_adaptation_mode),
            "runtime": dataclasses.asdict(config.runtime),
            "agent_model": dataclasses.asdict(config.model),
            "user_model": app.user_model,
            "user_base_url": app.user_base_url,
            "official_path": _as_wsl_path(
                app.artifact_dir / domain / task_key / f"trial-{trial}.json"
            ),
            "agent_path": _as_wsl_path(
                app.artifact_dir / domain / task_key / f"trial-{trial}.agent.json"
            ),
            "graph_path": _as_wsl_path(
                app.graph_dir / domain / f"{task_key}.trial-{trial}.json"
            ),
        }
        try:
            response, retry_errors = _worker_request_with_retry(
                app.worker_command,
                request,
                timeout=config.runtime.task_timeout_seconds + 120,
                env_names=(config.model.api_key_env, app.user_api_key_env),
                max_retries=app.task_max_retries,
                retry_delay=app.retry_delay_seconds,
            )
            record = {**response, "domain": domain, "task_id": task_id, "trial": trial, "run_signature": signature}
            record.setdefault("status", "finished")
            record["runner_retry_count"] = len(retry_errors)
            record["runner_retry_errors"] = retry_errors
        except Exception as exc:  # noqa: BLE001 - task boundaries record infrastructure failures
            retry_errors = list(getattr(exc, "_tau3_retry_errors", ()))
            record = {
                "domain": domain,
                "task_id": task_id,
                "trial": trial,
                "status": "failed",
                "runner_error": f"{type(exc).__name__}: {exc}",
                "runner_retry_count": len(retry_errors),
                "runner_retry_errors": retry_errors,
                "run_signature": signature,
            }
        append(record)
        return record

    if app.workers <= 1:
        for item in pending:
            run_one(*item)
    else:
        with ThreadPoolExecutor(max_workers=app.workers) as executor:
            futures = [executor.submit(run_one, *item) for item in pending]
            for future in as_completed(futures):
                future.result()
    final_records = _read_jsonl(app.results_path)
    summary = _summarize(selected, final_records, signature)
    _write_report(config, summary, final_records, inspection=inspection)
    return summary


def evaluate_tau3_benchmark(config: ExperimentConfig) -> dict[str, Any]:
    records = _read_jsonl(config.tau3.results_path)
    finished = [item for item in records if item.get("status") in {"finished", "failed"}]
    selected = [
        (str(item["domain"]), str(item["task_id"]), int(item["trial"]))
        for item in finished
    ]
    signatures = {str(item.get("run_signature")) for item in finished}
    if len(signatures) != 1:
        raise ValueError("tau3 result file must contain exactly one run signature")
    summary_object = _summarize(selected, finished, signatures.pop())
    return _write_report(
        config, summary_object, finished, inspection=inspect_tau3(config)
    )


def _write_report(
    config: ExperimentConfig,
    summary: Tau3RunSummary,
    records: Sequence[Mapping[str, Any]],
    *,
    inspection: Mapping[str, Any],
) -> dict[str, Any]:
    official_results = _aggregate_official_results(config, records)
    prompt, demonstrations = _tau3_prompt_bundle(
        config.tau3.prompt_variant,
        graph_adaptation_mode=config.runtime.graph_adaptation_mode,
    )
    payload = {
        **summary.to_dict(),
        "official_results": official_results,
        "official_inspection": inspection,
        "config": _public_config(config),
        "prompt_sha256": _hash(prompt),
        "demonstrations_sha256": _hash(demonstrations),
        "ptc_spec_sha256": _hash(_tau3_ptc_spec(config)),
    }
    config.tau3.report_path.parent.mkdir(parents=True, exist_ok=True)
    config.tau3.report_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return payload


def _aggregate_official_results(
    config: ExperimentConfig, records: Sequence[Mapping[str, Any]]
) -> dict[str, dict[str, Any]]:
    app = config.tau3
    finished = [item for item in records if item.get("status") == "finished"]
    output: dict[str, dict[str, Any]] = {}
    for domain in app.domains:
        domain_records = [item for item in finished if item.get("domain") == domain]
        if not domain_records:
            continue
        response = _worker_request(
            app.worker_command,
            {
                "type": "aggregate",
                "domain": domain,
                "task_split_name": app.task_split_name,
                "task_ids": [str(item["task_id"]) for item in domain_records],
                "official_paths": [str(item["official_path"]) for item in domain_records],
                "output_path": _as_wsl_path(
                    app.artifact_dir / "official-results" / domain / "results.json"
                ),
                "agent_model": config.model.model,
                "agent_name": _tau3_agent_name(
                    config.runtime.graph_adaptation_mode
                ),
                "user_model": app.user_model,
                "user_base_url": app.user_base_url,
                "num_trials": app.trials,
                "max_steps": app.max_steps,
                "max_errors": app.max_errors,
                "enforce_communication_protocol": app.enforce_communication_protocol,
                "max_concurrency": app.workers,
                "seed": app.seed,
                "max_retries": app.task_max_retries,
                "retry_delay": app.retry_delay_seconds,
            },
            timeout=300,
        )
        output[domain] = {
            "path": response["output_path"],
            "tasks": response["tasks"],
            "simulations": response["simulations"],
        }
    return output


def _summarize(
    selected: Sequence[tuple[str, str, int]], records: Sequence[Mapping[str, Any]], signature: str
) -> Tau3RunSummary:
    final = {
        (str(item.get("domain")), str(item.get("task_id")), int(item.get("trial", -1))): item
        for item in records
        if item.get("status") in {"finished", "failed"}
    }
    ordered = [final.get(key, {}) for key in selected]
    rewards = [float(item.get("reward") or 0.0) for item in ordered]
    by_domain: dict[str, dict[str, Any]] = {}
    for domain in sorted({key[0] for key in selected}):
        domain_rewards = [
            float(final.get(key, {}).get("reward") or 0.0) for key in selected if key[0] == domain
        ]
        by_domain[domain] = {
            "count": len(domain_rewards),
            "passed": sum(value == 1.0 for value in domain_rewards),
            "pass_hat_1": sum(value == 1.0 for value in domain_rewards) / len(domain_rewards) if domain_rewards else 0.0,
            "mean_reward": sum(domain_rewards) / len(domain_rewards) if domain_rewards else 0.0,
        }
    return Tau3RunSummary(
        selected=len(selected),
        processed=sum(bool(item) for item in ordered),
        passed=sum(value == 1.0 for value in rewards),
        pass_hat_1=sum(value == 1.0 for value in rewards) / len(selected) if selected else 0.0,
        mean_reward=sum(rewards) / len(selected) if selected else 0.0,
        execution_failure_tasks=sum(int(item.get("execution_failures", 0)) > 0 for item in ordered),
        execution_failure_blocks=sum(int(item.get("execution_failures", 0)) for item in ordered),
        incomplete_tasks=sum(bool(item.get("incomplete")) for item in ordered),
        evaluator_failures=sum(bool(item.get("evaluator_failed")) for item in ordered),
        runner_failures=sum(item.get("status") == "failed" for item in ordered),
        runner_retry_tasks=sum(int(item.get("runner_retry_count", 0)) > 0 for item in ordered),
        runner_retry_attempts=sum(int(item.get("runner_retry_count", 0)) for item in ordered),
        by_domain=by_domain,
        run_signature=signature,
    )


def _worker_request(
    command: Sequence[str], request: Mapping[str, Any], *, timeout: float, env_names: Sequence[str] = ()
) -> dict[str, Any]:
    if not command:
        raise ValueError("tau3.worker_command is required")
    environment = os.environ.copy()
    for name in env_names:
        value = os.getenv(name)
        if value:
            environment[name] = value
    shared = [f"{name}/u" for name in env_names if name]
    current = environment.get("WSLENV", "")
    environment["WSLENV"] = ":".join(
        value for value in (current, *shared) if value
    )
    completed = subprocess.run(
        list(command),
        input=json.dumps(request, ensure_ascii=True) + "\n",
        text=True,
        encoding="utf-8",
        errors="replace",
        capture_output=True,
        timeout=timeout,
        env=environment,
        check=False,
    )
    lines = [line for line in (completed.stdout or "").splitlines() if line.strip()]
    if completed.returncode != 0:
        if lines:
            try:
                error_payload = json.loads(lines[-1])
                if error_payload.get("error"):
                    raise RuntimeError(str(error_payload["error"]))
            except json.JSONDecodeError:
                pass
        raise RuntimeError((completed.stderr or "").strip() or (completed.stdout or "").strip())
    if not lines:
        raise RuntimeError("tau3 worker returned no JSON response")
    return json.loads(lines[-1])


def _worker_request_with_retry(
    command: Sequence[str],
    request: Mapping[str, Any],
    *,
    timeout: float,
    env_names: Sequence[str] = (),
    max_retries: int,
    retry_delay: float,
) -> tuple[dict[str, Any], list[str]]:
    """Apply the official per-task retry count without emitting console progress."""
    errors: list[str] = []
    for attempt in range(max_retries + 1):
        try:
            return (
                _worker_request(command, request, timeout=timeout, env_names=env_names),
                errors,
            )
        except Exception as exc:
            errors.append(f"{type(exc).__name__}: {exc}")
            if attempt >= max_retries:
                exc._tau3_retry_errors = tuple(errors[:-1])
                raise
            if retry_delay:
                time.sleep(retry_delay)
    raise AssertionError("unreachable")


def _public_config(config: ExperimentConfig) -> dict[str, Any]:
    return {
        "model": dataclasses.asdict(config.model),
        "runtime": dataclasses.asdict(config.runtime),
        "tau3": {
            key: value
            for key, value in dataclasses.asdict(config.tau3).items()
            if key not in {"worker_command", "results_path", "report_path", "artifact_dir", "graph_dir", "progress_path"}
        },
    }


def _hash(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, ensure_ascii=False, default=str, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _as_wsl_path(path: Path) -> str:
    absolute = path.resolve()
    drive = absolute.drive.rstrip(":").lower()
    rest = absolute.as_posix().split(":", 1)[-1].lstrip("/")
    return f"/mnt/{drive}/{rest}"
