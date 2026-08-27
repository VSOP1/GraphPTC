from __future__ import annotations

import copy
import dataclasses
import hashlib
import json
import subprocess
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from .config import ExperimentConfig
from .goal_adaptation import GoalGraphAdaptation
from .graph_agent import GraphAgentHooks, extend_ptc_spec_with_graph_control
from .mcpmark_runtime import MCPMarkProgramRuntime, create_mcp_client
from .model import OpenAIChatModel
from .ptc import OriginalPTCAgent, PTC_TOOL_SPEC
from .tool_effects import ToolEffectContract


MCPMARK_OFFICIAL_COMMIT = "cd45b7f57923b9b3985467f5139927575f83141c"
MCPMARK_SERVICES = (
    "filesystem",
    "notion",
    "github",
    "postgres",
    "playwright",
    "playwright_webarena",
)

MCPMARK_PTC_BASE_PROMPT = """You are an autonomous MCPMark agent. Your only directly callable model
tool is programmatic_tool_call. Its Python source runs directly in one persistent namespace for the
current task; variables and safe computation imports persist across blocks and reset before the next
task. The MCP tools listed below are Python globals. Call them with keyword arguments matching their
live schemas. Their return value is the MCP SDK result as a Python dictionary.

Use only these wrappers to inspect or change benchmark state. Do not access files, environment
variables, the shell, or the network by other means. Never invent tool names, argument names, IDs,
or successful effects. Inspect compactly, make only requested changes, and verify uncertain state
through MCP when appropriate. Put mechanically foreseeable calls, loops, filtering, joins, and
aggregation in one coherent PTC block, then return to the model for a new semantic decision or a
repair. Only printed stdout is visible to the next model turn, so print compact decision-relevant
values rather than raw collections or secrets. When the task is complete, reply concisely without
another tool call.

{tool_manifest}"""

MCPMARK_USER_PROMPT = """Complete this MCPMark task autonomously:

<task>{question}</task>"""

MCPMARK_PTC_SPEC = {
    **PTC_TOOL_SPEC,
    "function": {
        **PTC_TOOL_SPEC["function"],
        "description": (
            "Execute one coherent Python program directly in this task's persistent "
            "namespace. The listed MCP wrappers are globals."
        ),
        "parameters": {
            **PTC_TOOL_SPEC["function"]["parameters"],
            "properties": {
                "code": {
                    "type": "string",
                    "description": (
                        "Exact Python source for the persistent namespace; call only the "
                        "listed dynamic MCP wrappers for external effects."
                    ),
                }
            },
        },
    },
}


def _demo_messages() -> tuple[dict[str, Any], ...]:
    arguments: dict[str, Any] = {
        "code": (
            "records = [{'id':'a','active':True},{'id':'b','active':False},"
            "{'id':'c','active':True}]\n"
            "active_ids = [row['id'] for row in records if row['active']]\n"
            "print({'active_ids': active_ids})"
        )
    }
    return (
        {
            "role": "user",
            "content": (
                "PTC organization demonstration only: identify active IDs in these already "
                "available records; do not call an MCP tool."
            ),
        },
        {
            "role": "assistant",
            "content": "I will compute the deterministic intermediate result in one block.",
            "tool_calls": [
                {
                    "id": "mcpmark_demo_1",
                    "type": "function",
                    "function": {
                        "name": "programmatic_tool_call",
                        "arguments": json.dumps(arguments),
                    },
                }
            ],
        },
        {
            "role": "tool",
            "tool_call_id": "mcpmark_demo_1",
            "content": "{'active_ids': ['a', 'c']}",
        },
        {"role": "assistant", "content": "The active IDs are a and c."},
    )


class _EmptySearchTools:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []


class OfficialWorker:
    def __init__(self, command: Sequence[str]) -> None:
        if not command:
            raise ValueError("[mcpmark].official_worker_command is required")
        self._process = subprocess.Popen(
            tuple(command),
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        self._stderr: list[str] = []

        def read_stderr() -> None:
            assert self._process.stderr is not None
            self._stderr.extend(self._process.stderr)

        threading.Thread(target=read_stderr, daemon=True).start()

    def request(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        if self._process.poll() is not None:
            raise RuntimeError(f"official worker exited: {''.join(self._stderr[-20:])}")
        assert self._process.stdin is not None
        assert self._process.stdout is not None
        self._process.stdin.write(json.dumps(dict(payload), ensure_ascii=True) + "\n")
        self._process.stdin.flush()
        line = self._process.stdout.readline()
        if not line:
            raise RuntimeError(f"official worker returned no response: {''.join(self._stderr[-20:])}")
        response = json.loads(line)
        if response.get("type") == "error":
            raise RuntimeError(
                f"official worker {response.get('error_type')}: {response.get('error')}"
            )
        return response

    def close(self) -> None:
        if self._process.poll() is not None:
            return
        try:
            self.request({"type": "close"})
            self._process.wait(timeout=10)
        except Exception:
            self._process.kill()
            self._process.wait(timeout=10)


@dataclass(frozen=True)
class MCPMarkRunSummary:
    selected: int
    processed: int
    passed: int
    setup_failures: int
    execution_failures: int
    evaluator_failures: int
    verifier_failures: int
    cleanup_failures: int
    mcp_calls: int
    input_tokens: int
    output_tokens: int
    run_signature: str

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


def inspect_mcpmark(config: ExperimentConfig) -> dict[str, Any]:
    mcpmark = config.mcpmark
    worker = OfficialWorker(mcpmark.official_worker_command)
    try:
        inspection = worker.request(
            {
                "type": "inspect",
                "root": mcpmark.root,
                "env_path": str(mcpmark.env_path),
                "task_suite": mcpmark.task_suite,
                "services": list(MCPMARK_SERVICES),
            }
        )
    finally:
        worker.close()
    _validate_inspection(config, inspection)
    tasks = _flatten_tasks(inspection)
    manifest = {
        "schema_version": 1,
        "benchmark": "MCPMark Verified",
        "official_commit": inspection["official_commit"],
        "official_tree": inspection["official_tree"],
        "official_worktree_line_endings_differ": inspection.get(
            "worktree_line_endings_differ", False
        ),
        "task_suite": mcpmark.task_suite,
        "expected_tasks": mcpmark.expected_tasks,
        "tasks": tasks,
        "tasks_sha256": _sha256(tasks),
        "environment": {
            "python": inspection.get("python"),
            "pixi_lock_sha256": inspection.get("pixi_lock_sha256"),
            "packages": inspection.get("packages"),
        },
    }
    mcpmark.task_manifest_path.parent.mkdir(parents=True, exist_ok=True)
    mcpmark.task_manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return manifest


def run_mcpmark_benchmark(
    config: ExperimentConfig,
    *,
    task_ids: Sequence[str] = (),
    limit: int | None = None,
    progress: Callable[[Mapping[str, Any]], None] | None = None,
) -> MCPMarkRunSummary:
    _validate_config(config)
    manifest = _load_manifest(config)
    selected = list(manifest["tasks"])
    requested_task_ids = tuple(task_ids) or config.mcpmark.task_ids
    if requested_task_ids:
        wanted = set(requested_task_ids)
        if len(wanted) != len(requested_task_ids):
            raise ValueError("MCPMark task IDs must be unique")
        selected = [task for task in selected if _task_id(task) in wanted]
        missing = sorted(wanted - {_task_id(task) for task in selected})
        if missing:
            raise ValueError(f"unknown MCPMark task IDs: {missing}")
    if limit is not None:
        selected = selected[:limit]
    signature_payload = _signature_payload(config, manifest, selected)
    signature = _sha256(signature_payload)
    existing = _terminal_records(config.mcpmark.results_path)
    mismatched = {
        record.get("run_signature")
        for record in existing.values()
        if record.get("run_signature") != signature
    }
    if mismatched:
        raise ValueError("existing MCPMark results use another run signature")
    pending = [task for task in selected if _task_id(task) not in existing]
    for task in pending:
        record = _run_one(config, task, signature, progress=progress)
        _append_jsonl(config.mcpmark.results_path, record)
    terminal = _terminal_records(config.mcpmark.results_path)
    records = [terminal[_task_id(task)] for task in selected if _task_id(task) in terminal]
    summary = _summarize(selected, records, signature)
    report = _build_report(config, manifest, signature_payload, summary, records)
    config.mcpmark.report_path.parent.mkdir(parents=True, exist_ok=True)
    config.mcpmark.report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, default=repr), encoding="utf-8"
    )
    return summary


def evaluate_mcpmark_benchmark(config: ExperimentConfig) -> dict[str, Any]:
    report_path = config.mcpmark.report_path
    if not report_path.exists():
        raise ValueError("MCPMark report does not exist")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    manifest = _load_manifest(config)
    selected = report.get("selected_tasks") or []
    payload = _signature_payload(config, manifest, selected)
    if payload != report.get("run_signature_payload"):
        raise ValueError("MCPMark report signature payload does not match current inputs")
    if _sha256(payload) != report.get("summary", {}).get("run_signature"):
        raise ValueError("MCPMark report signature is invalid")
    return report


def compare_mcpmark_benchmarks(
    graph_config: ExperimentConfig,
    baseline_config: ExperimentConfig,
    output_path: Path,
) -> dict[str, Any]:
    _validate_arm_pair(graph_config, baseline_config)
    graph = evaluate_mcpmark_benchmark(graph_config)
    baseline = evaluate_mcpmark_benchmark(baseline_config)
    graph_records = {record["task_id"]: record for record in graph["tasks"]}
    baseline_records = {record["task_id"]: record for record in baseline["tasks"]}
    if graph_records.keys() != baseline_records.keys():
        raise ValueError("MCPMark arm reports do not contain the same task IDs")
    pairs = [
        (graph_records[task_id], baseline_records[task_id])
        for task_id in sorted(graph_records)
    ]
    report = {
        "schema_version": 1,
        "benchmark": "MCPMark Verified",
        "graph_run_signature": graph["summary"]["run_signature"],
        "baseline_run_signature": baseline["summary"]["run_signature"],
        "overall": _paired_metrics(pairs),
        "per_service": {
            service: _paired_metrics(
                [pair for pair in pairs if pair[0]["task"]["report_service"] == service]
            )
            for service in ("filesystem", "notion", "github", "postgres", "playwright")
        },
        "graph_delta_mechanism": {
            "tasks_with_delta": sum(
                int((record.get("graph_delta_sequence") or {}).get("graph_deltas", 0)) > 0
                for record in graph_records.values()
            ),
            "graph_deltas": sum(
                int((record.get("graph_delta_sequence") or {}).get("graph_deltas", 0))
                for record in graph_records.values()
            ),
            "deltas_preceding_later_action": sum(
                int(
                    (record.get("graph_delta_sequence") or {}).get(
                        "deltas_preceding_later_action", 0
                    )
                )
                for record in graph_records.values()
            ),
            "causal_influence_established": False,
            "note": "Temporal exposure and action alignment do not establish counterfactual influence.",
        },
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return report


def _run_one(
    config: ExperimentConfig,
    task: Mapping[str, Any],
    signature: str,
    *,
    progress: Callable[[Mapping[str, Any]], None] | None,
) -> dict[str, Any]:
    mcpmark = config.mcpmark
    task_id = _task_id(task)
    _progress(mcpmark.progress_path, {"task_id": task_id, "status": "started"}, progress)
    worker = OfficialWorker(mcpmark.official_worker_command)
    runtime: MCPMarkProgramRuntime | None = None
    controller: GoalGraphAdaptation | None = None
    cleanup: dict[str, Any] | None = None
    setup: dict[str, Any] | None = None
    verification: dict[str, Any] | None = None
    agent_result = None
    evaluator_attempted = False
    messages: list[dict[str, Any]] = []
    server: dict[str, Any] | None = None
    started = time.time()
    error: str | None = None
    phase = "setup"
    phase_errors: dict[str, str] = {}
    status = "failed"
    try:
        setup = worker.request(
            {
                "type": "initialize",
                "root": mcpmark.root,
                "env_path": str(mcpmark.env_path),
                "task_suite": mcpmark.task_suite,
                "service": task["official_service"],
                "task_key": task["task_key"],
            }
        )
        if not setup.get("setup_success"):
            raise RuntimeError("official setup returned false")
        phase = "execution"
        client, server = create_mcp_client(
            str(task["official_service"]),
            setup["service_config"],
            timeout_seconds=config.runtime.code_timeout_seconds,
            commands={
                "npx": mcpmark.npx_command,
                "pipx": mcpmark.pipx_command,
                "docker": mcpmark.docker_command,
            },
            npm_cache_dir=mcpmark.npm_cache_dir,
            npm_dependency_cutoff=mcpmark.npm_dependency_cutoff,
            postgres_pip_constraints=str(mcpmark.postgres_pip_constraints),
        )
        tools = client.list_tools()
        runtime = MCPMarkProgramRuntime(client, tools)
        system_prompt, demonstrations = _prompt_bundle(config, runtime.tool_manifest)
        latest_checkpoint: dict[str, Any] = {}

        def checkpoint(value: dict[str, Any]) -> None:
            latest_checkpoint.clear()
            latest_checkpoint.update(copy.deepcopy(value))

        if config.runtime.graph_adaptation_mode == "generic":
            contracts = _contracts(runtime.tool_manifest)
            functions = {function.__name__: function for function in runtime.functions}
            controller = GoalGraphAdaptation(
                functions,
                contracts,
                task=str(setup["instruction"]),
                expose_graph_api=False,
                host_inspection_enabled=False,
            )
            hooks = GraphAgentHooks.from_controller(controller).agent_kwargs()
        else:
            hooks = {"runtime_functions": runtime.functions}
        model = OpenAIChatModel(
            config.model, config.require_api_key(config.model.api_key_env)
        )
        agent = OriginalPTCAgent(
            model=model,
            search_tools=_EmptySearchTools(),  # type: ignore[arg-type]
            runtime=config.runtime,
            system_prompt=system_prompt,
            user_prompt_template=MCPMARK_USER_PROMPT,
            ptc_tool_spec=_ptc_spec(config),
            demonstration_messages=demonstrations,
            program_runtime=runtime,
            checkpoint_callback=checkpoint,
            **hooks,
        )
        agent_result = agent.run(str(setup["instruction"]))
        messages = _final_messages(
            latest_checkpoint.get("messages", []),
            task=str(setup["instruction"]),
            answer=agent_result.answer,
        )
        sdk_messages = _to_sdk_messages(messages)
        artifact = mcpmark.artifact_dir / _artifact_name(task)
        artifact.mkdir(parents=True, exist_ok=True)
        messages_path = artifact / "messages.json"
        messages_path.write_text(
            json.dumps(sdk_messages, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        (artifact / "execution.json").write_text(
            json.dumps(
                {
                    "agent": agent_result.to_dict(),
                    "runtime": runtime.telemetry(),
                    "tool_manifest": runtime.tool_manifest,
                    "server": server,
                },
                ensure_ascii=False,
                indent=2,
                default=repr,
            ),
            encoding="utf-8",
        )
        phase = "evaluator"
        evaluator_attempted = True
        verification = worker.request(
            {
                "type": "verify",
                "messages_path": str(messages_path),
                "agent_success": agent_result.status == "success",
                "agent_error": agent_result.error,
                "token_usage": dataclasses.asdict(agent_result.usage),
                "turn_count": agent_result.model_requests,
            }
        )
        status = "finished"
    except BaseException as exc:
        error = f"{type(exc).__name__}: {exc}"
        phase_errors[phase] = error
    finally:
        if runtime is not None:
            try:
                runtime.close()
            except BaseException as exc:
                error = error or f"runtime close {type(exc).__name__}: {exc}"
        try:
            cleanup = worker.request({"type": "cleanup"})
            if not cleanup.get("success"):
                error = error or "official cleanup returned false"
                phase_errors["cleanup"] = "official cleanup returned false"
        except BaseException as exc:
            cleanup = {"type": "cleanup", "success": False, "error": str(exc)}
            error = error or f"cleanup {type(exc).__name__}: {exc}"
            phase_errors["cleanup"] = f"{type(exc).__name__}: {exc}"
        worker.close()
    if controller is not None:
        try:
            controller.finish(
                answered=agent_result is not None and agent_result.status == "success"
            )
            graph_path = mcpmark.graph_dir / f"{_artifact_name(task)}.json"
            graph_path.parent.mkdir(parents=True, exist_ok=True)
            graph_path.write_text(
                json.dumps(
                    controller.graph_artifact(),
                    ensure_ascii=False,
                    indent=2,
                    default=repr,
                ),
                encoding="utf-8",
            )
        except BaseException as exc:
            artifact_error = f"{type(exc).__name__}: {exc}"
            error = error or f"graph artifact {artifact_error}"
            phase_errors["artifacts"] = artifact_error
    record = {
        "task_id": task_id,
        "status": status if error is None else "failed",
        "run_signature": signature,
        "task": dict(task),
        "setup": _without_service_config(setup),
        "agent": agent_result.to_dict() if agent_result is not None else None,
        "evaluator_attempted": evaluator_attempted,
        "verification": verification,
        "cleanup": cleanup,
        "server": server,
        "graph_telemetry": controller.telemetry() if controller is not None else None,
        "graph_delta_sequence": _graph_delta_sequence(messages),
        "error": error,
        "phase_errors": phase_errors,
        "duration_seconds": time.time() - started,
    }
    artifact = mcpmark.artifact_dir / _artifact_name(task)
    artifact.mkdir(parents=True, exist_ok=True)
    for name, value in (
        ("setup.json", _without_service_config(setup)),
        ("verification.json", verification),
        ("cleanup.json", cleanup),
    ):
        if value is not None:
            (artifact / name).write_text(
                json.dumps(value, ensure_ascii=False, indent=2, default=repr),
                encoding="utf-8",
            )
    (artifact / "meta.json").write_text(
        json.dumps(record, ensure_ascii=False, indent=2, default=repr), encoding="utf-8"
    )
    _progress(
        mcpmark.progress_path,
        {"task_id": task_id, "status": record["status"], "passed": _passed(record)},
        progress,
    )
    return record


def _prompt_bundle(
    config: ExperimentConfig, tool_manifest: Sequence[Mapping[str, Any]]
) -> tuple[str, tuple[dict[str, Any], ...]]:
    if config.mcpmark.prompt_variant != "mcpmark-ptc-fewshot":
        raise ValueError(f"unsupported MCPMark prompt variant: {config.mcpmark.prompt_variant}")
    rendered = "Available MCP wrappers:\n" + "\n".join(
        f"- {tool['wrapper']} -> {tool['mcp_tool']}: {tool.get('description', '')}\n"
        f"  schema={json.dumps(tool.get('input_schema', {}), ensure_ascii=False, sort_keys=True)}"
        for tool in tool_manifest
    )
    prompt = MCPMARK_PTC_BASE_PROMPT.format(tool_manifest=rendered)
    return prompt, _demo_messages()


def _ptc_spec(config: ExperimentConfig) -> dict[str, Any]:
    if config.runtime.graph_inspection_enabled:
        raise ValueError("MCPMark does not expose graph inspection")
    if config.runtime.graph_adaptation_mode == "off":
        return copy.deepcopy(MCPMARK_PTC_SPEC)
    if config.runtime.graph_adaptation_mode != "generic":
        raise ValueError("MCPMark graph_adaptation_mode must be off or generic")
    return extend_ptc_spec_with_graph_control(
        MCPMARK_PTC_SPEC,
        include_input_artifacts=False,
        include_inspection=False,
        target_description="Use task for this MCPMark episode.",
    )


def _contracts(
    manifest: Sequence[Mapping[str, Any]],
) -> dict[str, ToolEffectContract]:
    contracts: dict[str, ToolEffectContract] = {}
    for tool in manifest:
        annotations = tool.get("annotations") or {}
        effect = "read" if annotations.get("readOnlyHint") is True else "write"
        name = str(tool["wrapper"])
        contracts[name] = ToolEffectContract(name=name, effect=effect)
    return contracts


def _validate_config(config: ExperimentConfig) -> None:
    if config.mcpmark.official_commit != MCPMARK_OFFICIAL_COMMIT:
        raise ValueError("MCPMark official commit is not the frozen Verified commit")
    if config.mcpmark.workers != 1:
        raise ValueError("MCPMark must remain serial until isolation is verified")
    if config.mcpmark.k != 1:
        raise ValueError("MCPMark evaluation requires k=1")
    if config.model.model != "mimo-v2.5":
        raise ValueError("MCPMark model must be mimo-v2.5")
    if config.model.temperature != 0:
        raise ValueError("MCPMark temperature must be zero")
    if config.model.thinking != "disabled":
        raise ValueError("MCPMark thinking must be disabled")
    if config.model.max_completion_tokens != 32768:
        raise ValueError("MCPMark max completion tokens must be 32768")
    if config.model.max_retries != 0:
        raise ValueError("MCPMark model retries must be disabled")
    if config.runtime.max_turns != 100:
        raise ValueError("MCPMark requires MAX_TURNS=100")
    if config.runtime.max_ptc_blocks != 99:
        raise ValueError("MCPMark reserves turn 100 for finalization")
    if config.runtime.finalization_max_tokens != 32768:
        raise ValueError("MCPMark finalization tokens must be 32768")
    if config.runtime.task_timeout_seconds != 3600:
        raise ValueError("MCPMark requires a 3600 second task timeout")
    if config.runtime.max_compactions != 0:
        raise ValueError("MCPMark compaction must be disabled")
    if config.runtime.compaction_trigger_input_tokens is not None:
        raise ValueError("MCPMark compaction trigger must be disabled")
    if config.runtime.graph_inspection_enabled:
        raise ValueError("MCPMark graph inspection must be disabled")
    if config.runtime.graph_adaptation_mode not in {"off", "generic"}:
        raise ValueError("MCPMark graph adaptation must be off or generic")
    if not config.mcpmark.postgres_pip_constraints.is_file():
        raise ValueError("MCPMark PostgreSQL MCP constraints file is missing")
    if not config.mcpmark.platform_provenance_path.is_file():
        raise ValueError("MCPMark platform provenance artifact is missing")


def _validate_inspection(config: ExperimentConfig, inspection: Mapping[str, Any]) -> None:
    _validate_config(config)
    if inspection.get("official_commit") != config.mcpmark.official_commit:
        raise ValueError("inspected MCPMark checkout is not at the frozen commit")
    if inspection.get("official_dirty"):
        raise ValueError("inspected MCPMark checkout is dirty")
    count = sum(int(value["count"]) for value in inspection["services"].values())
    if count != config.mcpmark.expected_tasks:
        raise ValueError(f"expected {config.mcpmark.expected_tasks} tasks, found {count}")


def _flatten_tasks(inspection: Mapping[str, Any]) -> list[dict[str, Any]]:
    tasks = []
    for service in MCPMARK_SERVICES:
        for task in inspection["services"][service]["tasks"]:
            tasks.append(
                {
                    **task,
                    "official_service": service,
                    "report_service": "playwright" if service == "playwright_webarena" else service,
                }
            )
    return tasks


def _load_manifest(config: ExperimentConfig) -> dict[str, Any]:
    path = config.mcpmark.task_manifest_path
    if not path.exists():
        raise ValueError("MCPMark task manifest does not exist; run inspect-mcpmark first")
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("official_commit") != config.mcpmark.official_commit:
        raise ValueError("MCPMark manifest commit mismatch")
    if manifest.get("task_suite") != config.mcpmark.task_suite:
        raise ValueError("MCPMark manifest suite mismatch")
    if len(manifest.get("tasks", [])) != config.mcpmark.expected_tasks:
        raise ValueError("MCPMark manifest task count mismatch")
    if _sha256(manifest["tasks"]) != manifest.get("tasks_sha256"):
        raise ValueError("MCPMark manifest hash mismatch")
    environment = manifest.get("environment") or {}
    if not environment.get("pixi_lock_sha256") or not environment.get("packages"):
        raise ValueError("MCPMark manifest lacks the locked official environment; inspect again")
    return manifest


def _signature_payload(
    config: ExperimentConfig,
    manifest: Mapping[str, Any],
    selected: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    behavior = {
        "model": dataclasses.asdict(config.model),
        "runtime": dataclasses.asdict(config.runtime),
        "prompt_variant": config.mcpmark.prompt_variant,
        "k": config.mcpmark.k,
        "task_suite": config.mcpmark.task_suite,
        "server_commands": {
            "npx": config.mcpmark.npx_command,
            "npm_cache_dir": config.mcpmark.npm_cache_dir,
            "npm_dependency_cutoff": config.mcpmark.npm_dependency_cutoff,
            "pipx": config.mcpmark.pipx_command,
            "docker": config.mcpmark.docker_command,
            "postgres_pip_constraints": str(
                config.mcpmark.postgres_pip_constraints
            ),
            "postgres_pip_constraints_sha256": hashlib.sha256(
                config.mcpmark.postgres_pip_constraints.read_bytes()
            ).hexdigest(),
            "platform_provenance_path": str(
                config.mcpmark.platform_provenance_path
            ),
            "platform_provenance_sha256": hashlib.sha256(
                config.mcpmark.platform_provenance_path.read_bytes()
            ).hexdigest(),
        },
    }
    return {
        "schema_version": 1,
        "benchmark": "MCPMark Verified",
        "official_commit": config.mcpmark.official_commit,
        "official_tree": manifest["official_tree"],
        "official_environment": dict(manifest["environment"]),
        "official_environment_sha256": _sha256(manifest["environment"]),
        "manifest_sha256": manifest["tasks_sha256"],
        "selected_tasks": [dict(task) for task in selected],
        "behavior": behavior,
        "behavior_sha256": _sha256(behavior),
        "graphptc_commit": _git("rev-parse", "HEAD"),
        "graphptc_dirty": bool(_git("status", "--porcelain", "--untracked-files=all")),
        "graphptc_source_sha256": _source_hash(),
        "official_mcpmark_agent_difference": {
            "agent_class": {
                "graphptc": "OriginalPTCAgent",
                "official": "MCPMarkAgent",
            },
            "temperature": {"graphptc": config.model.temperature, "official": 1.0},
            "thinking_mode": {"graphptc": config.model.thinking, "official": "on"},
            "max_completion_tokens": {
                "graphptc": config.model.max_completion_tokens,
                "official": 32768,
                "official_parameter_name": "max_tokens",
            },
            "enforcer_mode": {
                "graphptc": "not used by the direct OpenAI-compatible SDK adapter",
                "official": "on",
            },
            "tool_surface": {
                "graphptc": "one programmatic_tool_call over dynamic MCP wrappers",
                "official": "direct MCP function calls",
            },
            "matched_controls": {
                "max_turns": 100,
                "task_timeout_seconds": 3600,
                "compaction": "disabled",
                "k": 1,
            },
        },
    }


def _build_report(
    config: ExperimentConfig,
    manifest: Mapping[str, Any],
    signature_payload: Mapping[str, Any],
    summary: MCPMarkRunSummary,
    records: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    per_service: dict[str, Any] = {}
    for service in ("filesystem", "notion", "github", "postgres", "playwright"):
        subset = [r for r in records if r["task"]["report_service"] == service]
        per_service[service] = {
            "tasks": len(subset),
            "passed": sum(_passed(record) for record in subset),
            "pass_at_1": (sum(_passed(record) for record in subset) / len(subset) if subset else 0),
            "setup_failures": sum(_setup_failed(record) for record in subset),
            "execution_failures": sum(_execution_failed(record) for record in subset),
            "evaluator_failures": sum(_evaluator_failed(record) for record in subset),
            "verifier_failures": sum(_verifier_failed(record) for record in subset),
            "cleanup_failures": sum(
                not bool((record.get("cleanup") or {}).get("success")) for record in subset
            ),
            "mcp_calls": sum(
                int(((record.get("agent") or {}).get("runtime_session") or {}).get("mcp_calls", 0))
                for record in subset
            ),
            "input_tokens": sum(
                int(((record.get("agent") or {}).get("usage") or {}).get("input_tokens", 0))
                for record in subset
            ),
            "output_tokens": sum(
                int(((record.get("agent") or {}).get("usage") or {}).get("output_tokens", 0))
                for record in subset
            ),
        }
    return {
        "summary": summary.to_dict(),
        "per_service": per_service,
        "run_signature_payload": dict(signature_payload),
        "selected_tasks": [dict(task) for task in signature_payload["selected_tasks"]],
        "manifest_sha256": manifest["tasks_sha256"],
        "resolved_config": {
            "model": dataclasses.asdict(config.model),
            "runtime": dataclasses.asdict(config.runtime),
            "mcpmark": _jsonable_config(config),
        },
        "tasks": list(records),
    }


def _summarize(
    selected: Sequence[Mapping[str, Any]],
    records: Sequence[Mapping[str, Any]],
    signature: str,
) -> MCPMarkRunSummary:
    return MCPMarkRunSummary(
        selected=len(selected),
        processed=len(records),
        passed=sum(_passed(record) for record in records),
        setup_failures=sum(_setup_failed(record) for record in records),
        execution_failures=sum(_execution_failed(record) for record in records),
        evaluator_failures=sum(_evaluator_failed(record) for record in records),
        verifier_failures=sum(_verifier_failed(record) for record in records),
        cleanup_failures=sum(not bool((record.get("cleanup") or {}).get("success")) for record in records),
        mcp_calls=sum(int(((record.get("agent") or {}).get("runtime_session") or {}).get("mcp_calls", 0)) for record in records),
        input_tokens=sum(int(((record.get("agent") or {}).get("usage") or {}).get("input_tokens", 0)) for record in records),
        output_tokens=sum(int(((record.get("agent") or {}).get("usage") or {}).get("output_tokens", 0)) for record in records),
        run_signature=signature,
    )


def _final_messages(
    checkpoint_messages: Sequence[Mapping[str, Any]], *, task: str, answer: str
) -> list[dict[str, Any]]:
    messages = [copy.deepcopy(dict(message)) for message in checkpoint_messages]
    if not messages:
        messages.append({"role": "user", "content": MCPMARK_USER_PROMPT.format(question=task)})
    if answer and not (
        messages
        and messages[-1].get("role") == "assistant"
        and messages[-1].get("content") == answer
    ):
        messages.append({"role": "assistant", "content": answer})
    return messages


def _to_sdk_messages(messages: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    sdk: list[dict[str, Any]] = []
    for message in messages:
        role = message.get("role")
        if role == "user":
            sdk.append({"role": "user", "content": message.get("content", "")})
        elif role == "assistant":
            content = message.get("content")
            if content:
                sdk.append(
                    {
                        "id": "__fake_id__",
                        "content": [{"annotations": [], "text": content, "type": "output_text"}],
                        "role": "assistant",
                        "status": "completed",
                        "type": "message",
                    }
                )
            for call in message.get("tool_calls", ()):
                function = call.get("function", {})
                sdk.append(
                    {
                        "arguments": function.get("arguments", "{}"),
                        "call_id": call.get("id", ""),
                        "name": function.get("name", ""),
                        "type": "function_call",
                        "id": "__fake_id__",
                    }
                )
        elif role == "tool":
            sdk.append(
                {
                    "call_id": message.get("tool_call_id", ""),
                    "output": json.dumps(
                        {
                            "type": "text",
                            "text": message.get("content", ""),
                            "annotations": None,
                            "meta": None,
                        }
                    ),
                    "type": "function_call_output",
                }
            )
    return sdk


def _graph_delta_sequence(messages: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    delta_positions = [
        index
        for index, message in enumerate(messages)
        if message.get("role") == "tool" and "GRAPH_DELTA " in str(message.get("content", ""))
    ]
    later_actions = 0
    for position in delta_positions:
        if any(
            message.get("role") == "assistant" and message.get("tool_calls")
            for message in messages[position + 1 :]
        ):
            later_actions += 1
    return {
        "graph_deltas": len(delta_positions),
        "deltas_preceding_later_action": later_actions,
        "temporal_exposure_verified": bool(delta_positions) and later_actions > 0,
        "causal_influence_established": False,
        "causal_note": "Temporal order and action alignment do not identify counterfactual influence.",
    }


def _without_service_config(setup: Mapping[str, Any] | None) -> dict[str, Any] | None:
    if setup is None:
        return None
    return {key: value for key, value in setup.items() if key != "service_config"}


def _passed(record: Mapping[str, Any]) -> bool:
    return bool(((record.get("verification") or {}).get("result") or {}).get("success"))


def _setup_failed(record: Mapping[str, Any]) -> bool:
    return not bool((record.get("setup") or {}).get("setup_success"))


def _execution_failed(record: Mapping[str, Any]) -> bool:
    if _setup_failed(record):
        return False
    agent = record.get("agent") or {}
    return not agent or agent.get("status") != "success"


def _evaluator_failed(record: Mapping[str, Any]) -> bool:
    if _setup_failed(record):
        return False
    return bool(record.get("evaluator_attempted")) and record.get("verification") is None


def _verifier_failed(record: Mapping[str, Any]) -> bool:
    if _setup_failed(record) or _execution_failed(record) or _evaluator_failed(record):
        return False
    return not _passed(record)


def _paired_metrics(
    pairs: Sequence[tuple[Mapping[str, Any], Mapping[str, Any]]],
) -> dict[str, Any]:
    graph_passes = sum(_passed(graph) for graph, _ in pairs)
    baseline_passes = sum(_passed(baseline) for _, baseline in pairs)
    return {
        "tasks": len(pairs),
        "graph_passed": graph_passes,
        "baseline_passed": baseline_passes,
        "graph_pass_at_1": graph_passes / len(pairs) if pairs else 0,
        "baseline_pass_at_1": baseline_passes / len(pairs) if pairs else 0,
        "graph_wins": sum(_passed(graph) and not _passed(base) for graph, base in pairs),
        "graph_losses": sum(not _passed(graph) and _passed(base) for graph, base in pairs),
        "ties": sum(_passed(graph) == _passed(base) for graph, base in pairs),
    }


def _validate_arm_pair(
    graph_config: ExperimentConfig, baseline_config: ExperimentConfig
) -> None:
    if graph_config.model != baseline_config.model:
        raise ValueError("MCPMark arms must use identical model configuration")
    graph_runtime = dataclasses.asdict(graph_config.runtime)
    baseline_runtime = dataclasses.asdict(baseline_config.runtime)
    graph_runtime["graph_adaptation_mode"] = "off"
    if graph_runtime != baseline_runtime:
        raise ValueError("MCPMark arms differ beyond graph_adaptation_mode")
    if graph_config.runtime.graph_adaptation_mode != "generic":
        raise ValueError("GraphPTC comparison arm must use generic adaptation")
    if baseline_config.runtime.graph_adaptation_mode != "off":
        raise ValueError("baseline comparison arm must disable adaptation")
    for field in (
        "root",
        "official_commit",
        "official_worker_command",
        "npx_command",
        "npm_cache_dir",
        "npm_dependency_cutoff",
        "pipx_command",
        "docker_command",
        "postgres_pip_constraints",
        "platform_provenance_path",
        "env_path",
        "task_suite",
        "expected_tasks",
        "task_manifest_path",
        "task_ids",
        "workers",
        "k",
        "prompt_variant",
    ):
        if getattr(graph_config.mcpmark, field) != getattr(baseline_config.mcpmark, field):
            raise ValueError(f"MCPMark arms differ in {field}")


def _task_id(task: Mapping[str, Any]) -> str:
    return f"{task['official_service']}:{task['task_key']}"


def _artifact_name(task: Mapping[str, Any]) -> str:
    return _task_id(task).replace(":", "__").replace("/", "__")


def _terminal_records(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    records: dict[str, dict[str, Any]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            record = json.loads(line)
            task_id = str(record["task_id"])
            if task_id in records:
                raise ValueError(f"duplicate terminal MCPMark task record: {task_id}")
            records[task_id] = record
    return records


def _append_jsonl(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(dict(value), ensure_ascii=False, default=repr) + "\n")


def _progress(
    path: Path,
    value: Mapping[str, Any],
    callback: Callable[[Mapping[str, Any]], None] | None,
) -> None:
    _append_jsonl(path, {"timestamp": time.time(), **dict(value)})
    if callback is not None:
        callback(value)


def _jsonable_config(config: ExperimentConfig) -> dict[str, Any]:
    value = dataclasses.asdict(config.mcpmark)
    for key, item in list(value.items()):
        if isinstance(item, Path):
            value[key] = str(item)
        elif isinstance(item, tuple):
            value[key] = list(item)
    return value


def _sha256(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=repr)
    return hashlib.sha256(payload.encode()).hexdigest()


def _source_hash() -> str:
    digest = hashlib.sha256()
    root = Path(__file__).resolve().parent
    for path in sorted(root.rglob("*.py")):
        digest.update(path.relative_to(root).as_posix().encode())
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _git(*args: str) -> str:
    return subprocess.run(("git", *args), capture_output=True, text=True, check=True).stdout.strip()
