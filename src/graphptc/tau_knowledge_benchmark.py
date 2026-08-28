from __future__ import annotations

import dataclasses
import json
import threading
from collections.abc import Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import UTC, datetime
from itertools import pairwise
from pathlib import Path
from typing import Any

from .config import ExperimentConfig
from .tau3_benchmark import (
    TAU3_OFFICIAL_COMMIT,
    TAU3_OFFICIAL_VERSION,
    _hash,
    _public_config,
    _read_jsonl,
    _safe_task_key,
    _tau3_agent_name,
    _tau3_prompt_bundle,
    _tau3_ptc_spec,
    _worker_request,
    _worker_request_with_retry,
)

DEFAULT_PROTOCOL_PATH = Path("configs/tau_knowledge/protocol.json")


@dataclass(frozen=True)
class TauKnowledgeRunSummary:
    selected: int
    processed: int
    passed: int
    pass_hat_1: float
    mean_official_reward: float
    retrieval_calls: int
    tool_calls: int
    dynamic_tool_calls: int
    model_turns: int
    ptc_blocks: int
    input_tokens: int
    output_tokens: int
    cached_input_tokens: int
    duration_seconds: float
    execution_failure_tasks: int
    execution_failure_blocks: int
    incomplete_tasks: int
    evaluator_failures: int
    runner_failures: int
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
            for key in (
                "domain",
                "task_id",
                "trial",
                "status",
                "started_at",
                "finished_at",
            )
            if key in record
        }
        with self._lock, self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, ensure_ascii=False) + "\n")


def load_tau_knowledge_protocol(
    path: str | Path = DEFAULT_PROTOCOL_PATH,
) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    required = {
        "official_version",
        "official_commit",
        "source_repository",
        "required_runtime_files",
        "domain",
        "task_split_name",
        "expected_tasks",
        "trial_indices",
        "retrieval_config",
        "retrieval_config_kwargs",
        "bm25_index",
        "knowledge_base",
        "task_git_manifest_sha256",
        "retrieval_probe_queries",
        "smoke_task_ids",
    }
    missing = sorted(required - payload.keys())
    if missing:
        raise ValueError(f"tau-Knowledge protocol is missing fields: {missing}")
    return payload


def inspect_tau_knowledge(
    config: ExperimentConfig,
    protocol: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    protocol = dict(protocol or load_tau_knowledge_protocol())
    return _worker_request(
        config.tau3.worker_command,
        {
            "type": "inspect",
            "root": config.tau3.root,
            "official_commit": protocol["official_commit"],
            "source_repository": protocol["source_repository"],
            "required_runtime_files": protocol["required_runtime_files"],
            "task_split_name": protocol["task_split_name"],
            "retrieval_config": protocol["retrieval_config"],
            "retrieval_config_kwargs": protocol["retrieval_config_kwargs"],
            "retrieval_probe_queries": protocol["retrieval_probe_queries"],
        },
        timeout=600,
    )


def validate_tau_knowledge_alignment(
    config: ExperimentConfig,
    inspection: Mapping[str, Any],
    protocol: Mapping[str, Any],
) -> None:
    app = config.tau3
    if protocol["official_commit"] != TAU3_OFFICIAL_COMMIT:
        raise ValueError(
            "protocol commit differs from the frozen official v1.0.1 commit"
        )
    if protocol["official_version"] != TAU3_OFFICIAL_VERSION:
        raise ValueError("protocol version must be tau2-bench 1.0.1")
    if app.official_commit != protocol["official_commit"]:
        raise ValueError("configured commit differs from the tau-Knowledge protocol")
    if inspection.get("official_commit") != protocol["official_commit"]:
        raise ValueError("installed tau2-bench commit differs from the frozen protocol")
    provenance = inspection.get("source_provenance") or {}
    if provenance.get("commit") != protocol["official_commit"]:
        raise ValueError("tau2 source provenance commit changed")
    if provenance.get("transport") != "git":
        raise ValueError("tau2 source must be an exact git checkout")
    if provenance.get("url") != protocol["source_repository"]["url"]:
        raise ValueError("tau2 source repository URL changed")
    if provenance.get("tag") != protocol["source_repository"]["tag"]:
        raise ValueError("tau2 source tag changed")
    if inspection.get("required_runtime_files") != protocol["required_runtime_files"]:
        raise ValueError("official tau2 runtime file hashes changed")
    if inspection.get("package_version") != protocol["official_version"]:
        raise ValueError("installed tau2 package is not the frozen official version")
    if not inspection.get("data_verified"):
        raise ValueError("official tau2 check-data did not pass")
    if tuple(app.domains) != (protocol["domain"],):
        raise ValueError("tau-Knowledge evaluation must only use banking_knowledge")
    if app.task_split_name != protocol["task_split_name"]:
        raise ValueError("tau-Knowledge evaluation must use the complete base split")
    if app.trials != 1 or protocol["trial_indices"] != [0]:
        raise ValueError("tau-Knowledge matched evaluation is frozen to trial 0 only")
    if app.task_max_retries != 0:
        raise ValueError("tau-Knowledge selected tasks must not be retried")
    if app.workers not in {1, 3}:
        raise ValueError(
            "tau-Knowledge uses concurrency 1 for smoke or 3 for full evaluation"
        )
    if config.runtime.graph_inspection_enabled:
        raise ValueError("tau-Knowledge must not expose graph inspection")
    if config.runtime.max_stdout_chars != 8_000:
        raise ValueError("tau-Knowledge max_stdout_chars must be 8000")
    if config.runtime.graph_adaptation_mode not in {"generic", "off"}:
        raise ValueError("tau-Knowledge graph mode must be generic or off")
    if config.model.temperature != 0.0:
        raise ValueError("tau-Knowledge agent temperature must be 0")
    defaults = inspection.get("official_defaults") or {}
    expected_defaults = {
        "max_steps": app.max_steps,
        "max_errors": app.max_errors,
        "seed": app.seed,
        "agent_temperature": config.model.temperature,
        "user_temperature": 0.0,
    }
    for key, value in expected_defaults.items():
        if defaults.get(key) != value:
            raise ValueError(
                f"official {key} mismatch: expected {value}, got {defaults.get(key)}"
            )
    if int(inspection.get("task_count", -1)) != int(protocol["expected_tasks"]):
        raise ValueError("installed banking_knowledge base task count changed")
    if len(set(inspection.get("task_ids") or [])) != int(protocol["expected_tasks"]):
        raise ValueError(
            "installed banking_knowledge task IDs are not unique and complete"
        )
    if (inspection.get("task_files") or {}).get("git_manifest_sha256") != protocol[
        "task_git_manifest_sha256"
    ]:
        raise ValueError("banking_knowledge task manifest hash changed")
    knowledge = protocol["knowledge_base"]
    installed_documents = inspection.get("knowledge_documents") or {}
    if installed_documents.get("count") != knowledge["document_count"]:
        raise ValueError("banking_knowledge document count changed")
    if (
        installed_documents.get("git_manifest_sha256")
        != knowledge["document_git_manifest_sha256"]
    ):
        raise ValueError("banking_knowledge document manifest hash changed")
    installed_prompts = inspection.get("knowledge_prompts") or {}
    if installed_prompts.get("count") != knowledge["prompt_count"]:
        raise ValueError("banking_knowledge prompt count changed")
    if (
        installed_prompts.get("git_manifest_sha256")
        != knowledge["prompt_git_manifest_sha256"]
    ):
        raise ValueError("banking_knowledge prompt manifest hash changed")
    retrieval = inspection.get("retrieval") or {}
    if retrieval.get("config") != "bm25":
        raise ValueError("tau-Knowledge retrieval_config must be bm25")
    if retrieval.get("config_kwargs") != protocol["retrieval_config_kwargs"]:
        raise ValueError("tau-Knowledge BM25 kwargs changed")
    if not retrieval.get("offline_bm25_only"):
        raise ValueError(
            "retrieval inspection found embedding, reranker, grep, or shell access"
        )
    if not retrieval.get("arms_identical"):
        raise ValueError("GraphPTC and Fewshot PTC BM25 probes differ")
    probes = (
        retrieval.get("graphptc_probe") or {},
        retrieval.get("fewshot_ptc_probe") or {},
    )
    if any(probe.get("hidden_names_exposed") for probe in probes):
        raise ValueError("discoverable tool names leaked into the initial tool surface")
    variant = retrieval.get("variant") or {}
    pipeline = variant.get("kb_search") or {}
    if pipeline != {
        "type": "bm25",
        "embedder_type": None,
        "embedder_model": None,
        "top_k": 10,
        "reranker": False,
        "reranker_min_score": 5,
    }:
        raise ValueError("official BM25 pipeline specification changed")


def validate_tau_knowledge_arm_pair(
    graph_config: ExperimentConfig, baseline_config: ExperimentConfig
) -> None:
    if graph_config.model != baseline_config.model:
        raise ValueError("tau-Knowledge arms must use identical model configuration")
    graph_runtime = dataclasses.asdict(graph_config.runtime)
    baseline_runtime = dataclasses.asdict(baseline_config.runtime)
    graph_runtime["graph_adaptation_mode"] = "off"
    if graph_runtime != baseline_runtime:
        raise ValueError("tau-Knowledge arms differ beyond graph_adaptation_mode")
    if graph_config.runtime.graph_adaptation_mode != "generic":
        raise ValueError("GraphPTC arm must use generic graph adaptation")
    if baseline_config.runtime.graph_adaptation_mode != "off":
        raise ValueError("Fewshot PTC arm must disable graph adaptation")
    ignored = {
        "results_path",
        "report_path",
        "artifact_dir",
        "graph_dir",
        "progress_path",
    }
    graph_app = dataclasses.asdict(graph_config.tau3)
    baseline_app = dataclasses.asdict(baseline_config.tau3)
    for field in ignored:
        graph_app.pop(field)
        baseline_app.pop(field)
    if graph_app != baseline_app:
        raise ValueError("tau-Knowledge arms differ beyond output paths")


def run_tau_knowledge_benchmark(
    config: ExperimentConfig,
    *,
    protocol: Mapping[str, Any] | None = None,
    task_ids: Sequence[str] = (),
    restart: bool = False,
) -> TauKnowledgeRunSummary:
    protocol = dict(protocol or load_tau_knowledge_protocol())
    inspection = inspect_tau_knowledge(config, protocol)
    validate_tau_knowledge_alignment(config, inspection, protocol)
    app = config.tau3
    available = [str(value) for value in inspection["task_ids"]]
    selected_ids = list(task_ids) if task_ids else available
    unknown = sorted(set(selected_ids) - set(available))
    if unknown:
        raise ValueError(f"unknown banking_knowledge task IDs: {unknown}")
    if len(selected_ids) != len(set(selected_ids)):
        raise ValueError("selected tau-Knowledge task IDs must be unique")
    selected = [(protocol["domain"], task_id, 0) for task_id in selected_ids]
    prompt, demonstrations = _tau3_prompt_bundle(
        app.prompt_variant,
        graph_adaptation_mode=config.runtime.graph_adaptation_mode,
    )
    signature_payload = {
        "protocol": protocol,
        "official_inspection": inspection,
        "config": _public_config(config),
        "prompt": prompt,
        "demonstrations": demonstrations,
        "ptc_spec": _tau3_ptc_spec(config),
        "selected": selected,
    }
    signature = _hash(signature_payload)
    for path in (
        app.results_path.parent,
        app.artifact_dir,
        app.graph_dir,
        app.progress_path.parent,
    ):
        path.mkdir(parents=True, exist_ok=True)
    if restart:
        for path in (app.results_path, app.progress_path, app.report_path):
            path.unlink(missing_ok=True)
    existing = _read_jsonl(app.results_path)
    if any(item.get("run_signature") != signature for item in existing):
        raise ValueError("existing tau-Knowledge results use another run signature")
    seen = {
        (str(item.get("domain")), str(item.get("task_id")), int(item.get("trial", -1)))
        for item in existing
        if item.get("status") in {"finished", "failed"}
    }
    pending = [item for item in selected if item not in seen]
    write_lock = threading.Lock()
    progress = _ProgressLog(app.progress_path)

    def append(record: dict[str, Any]) -> None:
        with write_lock, app.results_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False, default=repr) + "\n")
        progress(record)

    def run_one(domain: str, task_id: str, trial: int) -> dict[str, Any]:
        started_at = datetime.now(UTC).isoformat()
        append(
            {
                "domain": domain,
                "task_id": task_id,
                "trial": trial,
                "status": "started",
                "started_at": started_at,
                "run_signature": signature,
            }
        )
        task_key = _safe_task_key(task_id)
        official_path = (
            app.artifact_dir / domain / task_key / "trial-0.json"
        ).resolve()
        agent_path = (
            app.artifact_dir / domain / task_key / "trial-0.agent.json"
        ).resolve()
        graph_path = (app.graph_dir / domain / f"{task_key}.trial-0.json").resolve()
        request = {
            "type": "run",
            "domain": domain,
            "task_id": task_id,
            "trial": trial,
            "seed": app.seed,
            "task_split_name": app.task_split_name,
            "task_ids": [task_id],
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
            "retrieval_config": protocol["retrieval_config"],
            "retrieval_config_kwargs": protocol["retrieval_config_kwargs"],
            "official_path": str(official_path),
            "agent_path": str(agent_path),
            "graph_path": str(graph_path),
        }
        try:
            response, retry_errors = _worker_request_with_retry(
                app.worker_command,
                request,
                timeout=config.runtime.task_timeout_seconds + 120,
                env_names=(config.model.api_key_env, app.user_api_key_env),
                max_retries=0,
                retry_delay=0,
            )
            record = {
                **response,
                "domain": domain,
                "task_id": task_id,
                "trial": trial,
                "started_at": started_at,
                "finished_at": datetime.now(UTC).isoformat(),
                "run_signature": signature,
                "runner_retry_count": len(retry_errors),
                "runner_retry_errors": retry_errors,
            }
            record.setdefault("status", "finished")
        except Exception as exc:  # noqa: BLE001 - preserve one failed trial
            record = {
                "domain": domain,
                "task_id": task_id,
                "trial": trial,
                "status": "failed",
                "started_at": started_at,
                "finished_at": datetime.now(UTC).isoformat(),
                "runner_error": f"{type(exc).__name__}: {exc}",
                "runner_retry_count": 0,
                "runner_retry_errors": [],
                "run_signature": signature,
            }
        append(record)
        return record

    if app.workers == 1:
        for item in pending:
            run_one(*item)
    else:
        with ThreadPoolExecutor(max_workers=app.workers) as executor:
            futures = [executor.submit(run_one, *item) for item in pending]
            for future in as_completed(futures):
                future.result()
    terminal = _terminal_records(_read_jsonl(app.results_path), signature)
    summary = _summarize(selected, terminal, signature)
    _write_report(
        config,
        protocol,
        inspection,
        signature_payload,
        selected,
        terminal,
        summary,
    )
    return summary


def evaluate_tau_knowledge_benchmark(
    config: ExperimentConfig,
    *,
    protocol: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    protocol = dict(protocol or load_tau_knowledge_protocol())
    report = json.loads(config.tau3.report_path.read_text(encoding="utf-8"))
    inspection = inspect_tau_knowledge(config, protocol)
    validate_tau_knowledge_alignment(config, inspection, protocol)
    if report.get("protocol_sha256") != _hash(protocol):
        raise ValueError("saved tau-Knowledge report uses another protocol")
    if report.get("official_inspection") != inspection:
        raise ValueError(
            "saved tau-Knowledge report differs from current official installation"
        )
    return report


def compare_tau_knowledge_benchmarks(
    graph_config: ExperimentConfig,
    baseline_config: ExperimentConfig,
    *,
    output_path: str | Path,
    protocol: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    protocol = dict(protocol or load_tau_knowledge_protocol())
    validate_tau_knowledge_arm_pair(graph_config, baseline_config)
    graph_report = evaluate_tau_knowledge_benchmark(graph_config, protocol=protocol)
    baseline_report = evaluate_tau_knowledge_benchmark(
        baseline_config, protocol=protocol
    )
    graph = {
        _record_key(item): _with_partial_artifact_metrics(graph_config, item)
        for item in graph_report["tasks"]
    }
    baseline = {
        _record_key(item): _with_partial_artifact_metrics(baseline_config, item)
        for item in baseline_report["tasks"]
    }
    if graph.keys() != baseline.keys():
        raise ValueError(
            "tau-Knowledge arm reports do not contain identical tasks and trials"
        )
    pairs = [(graph[key], baseline[key]) for key in sorted(graph)]
    for graph_record, baseline_record in pairs:
        graph_surface = graph_record.get("tool_surface") or {}
        baseline_surface = baseline_record.get("tool_surface") or {}
        if graph_surface.get("visible_tool_schema_sha256") != baseline_surface.get(
            "visible_tool_schema_sha256"
        ):
            raise ValueError(
                "tau-Knowledge arms received different official tool schemas"
            )
        if graph_surface.get("hidden_names_exposed") or baseline_surface.get(
            "hidden_names_exposed"
        ):
            raise ValueError(
                "discoverable tool names leaked before knowledge retrieval"
            )
    payload = {
        "schema_version": 1,
        "benchmark": "tau-Knowledge",
        "trial_indices": [0],
        "tasks": len(pairs),
        "protocol_sha256": _hash(protocol),
        "graph_run_signature": graph_report["summary"]["run_signature"],
        "baseline_run_signature": baseline_report["summary"]["run_signature"],
        "paired": _paired_summary(pairs),
        "graphptc": _arm_metrics(graph.values()),
        "fewshot_ptc": _arm_metrics(baseline.values()),
        "operational_deltas": _operational_deltas(graph.values(), baseline.values()),
        "graph_delta_mechanism": _graph_delta_mechanism(graph.values()),
        "task_pairs": [
            _task_pair_record(key, graph[key], baseline[key]) for key in sorted(graph)
        ],
    }
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return payload


def _write_report(
    config: ExperimentConfig,
    protocol: Mapping[str, Any],
    inspection: Mapping[str, Any],
    signature_payload: Mapping[str, Any],
    selected: Sequence[tuple[str, str, int]],
    records: Sequence[Mapping[str, Any]],
    summary: TauKnowledgeRunSummary,
) -> dict[str, Any]:
    app = config.tau3
    finished = [item for item in records if item.get("status") == "finished"]
    official_results: dict[str, Any] | None = None
    if finished:
        response = _worker_request(
            app.worker_command,
            {
                "type": "aggregate",
                "task_split_name": app.task_split_name,
                "task_ids": [str(item["task_id"]) for item in finished],
                "official_paths": [str(item["official_path"]) for item in finished],
                "output_path": str(
                    (app.artifact_dir / "official-results" / "results.json").resolve()
                ),
                "agent_model": config.model.model,
                "agent_name": _tau3_agent_name(config.runtime.graph_adaptation_mode),
                "user_model": app.user_model,
                "user_base_url": app.user_base_url,
                "retrieval_config": protocol["retrieval_config"],
                "retrieval_config_kwargs": protocol["retrieval_config_kwargs"],
                "max_steps": app.max_steps,
                "max_errors": app.max_errors,
                "max_concurrency": app.workers,
                "seed": app.seed,
                "enforce_communication_protocol": app.enforce_communication_protocol,
            },
            timeout=600,
        )
        official_results = response
    prompt, demonstrations = _tau3_prompt_bundle(
        app.prompt_variant,
        graph_adaptation_mode=config.runtime.graph_adaptation_mode,
    )
    payload = {
        "schema_version": 1,
        "benchmark": "tau-Knowledge",
        "development_or_official": "official tau2 evaluator on a custom PTC agent",
        "summary": summary.to_dict(),
        "protocol": dict(protocol),
        "protocol_sha256": _hash(protocol),
        "official_inspection": dict(inspection),
        "official_results": official_results,
        "resolved_config": _public_config(config),
        "run_signature_payload_sha256": _hash(signature_payload),
        "selected": [list(item) for item in selected],
        "prompt_sha256": _hash(prompt),
        "demonstrations_sha256": _hash(demonstrations),
        "ptc_spec_sha256": _hash(_tau3_ptc_spec(config)),
        "tasks": list(records),
    }
    app.report_path.parent.mkdir(parents=True, exist_ok=True)
    app.report_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return payload


def _terminal_records(
    records: Sequence[Mapping[str, Any]], signature: str
) -> list[dict[str, Any]]:
    terminal: dict[tuple[str, str, int], dict[str, Any]] = {}
    for item in records:
        if item.get("run_signature") != signature:
            continue
        if item.get("status") in {"finished", "failed"}:
            terminal[_record_key(item)] = dict(item)
    return list(terminal.values())


def _record_key(item: Mapping[str, Any]) -> tuple[str, str, int]:
    return (
        str(item.get("domain")),
        str(item.get("task_id")),
        int(item.get("trial", -1)),
    )


def _with_partial_artifact_metrics(
    config: ExperimentConfig, record: Mapping[str, Any]
) -> dict[str, Any]:
    enriched = dict(record)
    if enriched.get("status") != "failed" or enriched.get("runtime_metrics"):
        return enriched
    domain, task_id, trial = _record_key(enriched)
    task_key = _safe_task_key(task_id)
    agent_path = config.tau3.artifact_dir / domain / task_key / f"trial-{trial}.agent.json"
    if not agent_path.exists():
        return enriched
    artifact = json.loads(agent_path.read_text(encoding="utf-8"))
    blocks = artifact.get("blocks") or []
    calls = [
        call
        for block in blocks
        for call in ((block.get("runtime_trace") or {}).get("external_actions") or [])
    ]
    telemetry = artifact.get("telemetry") or {}
    usage = telemetry.get("usage") or {}
    try:
        duration_seconds = max(
            (
                datetime.fromisoformat(str(enriched["finished_at"]))
                - datetime.fromisoformat(str(enriched["started_at"]))
            ).total_seconds(),
            0.0,
        )
    except (KeyError, TypeError, ValueError):
        duration_seconds = 0.0
    graph_path = config.tau3.graph_dir / domain / f"{task_key}.trial-{trial}.json"
    enriched.update(
        {
            "partial_artifact_metrics": True,
            "agent_path": str(agent_path.resolve()),
            "graph_path": str(graph_path.resolve()) if graph_path.exists() else None,
            "telemetry": telemetry,
            "execution_failures": int(telemetry.get("execution_failures", 0)),
            "runtime_metrics": {
                "model_turns": int(telemetry.get("model_requests", 0)),
                "ptc_blocks": len(blocks),
                "tool_calls": len(calls),
                "retrieval_calls": sum(call.get("name") == "KB_search" for call in calls),
                "unlock_calls": sum(
                    call.get("name") == "unlock_discoverable_agent_tool"
                    for call in calls
                ),
                "dynamic_tool_calls": sum(
                    call.get("name") == "call_discoverable_agent_tool"
                    for call in calls
                ),
                "state_change_calls": sum(
                    call.get("state_changed") is True for call in calls
                ),
                "input_tokens": int(usage.get("input_tokens", 0)),
                "output_tokens": int(usage.get("output_tokens", 0)),
                "cached_input_tokens": int(usage.get("cached_input_tokens", 0)),
                "duration_seconds": duration_seconds,
            },
        }
    )
    return enriched


def _summarize(
    selected: Sequence[tuple[str, str, int]],
    records: Sequence[Mapping[str, Any]],
    signature: str,
) -> TauKnowledgeRunSummary:
    terminal = {_record_key(item): item for item in records}
    ordered = [terminal.get(key, {}) for key in selected]
    rewards = [float(item.get("reward") or 0) for item in ordered]
    metrics = [item.get("runtime_metrics") or {} for item in ordered]
    return TauKnowledgeRunSummary(
        selected=len(selected),
        processed=sum(bool(item) for item in ordered),
        passed=sum(value == 1.0 for value in rewards),
        pass_hat_1=sum(value == 1.0 for value in rewards) / len(selected)
        if selected
        else 0.0,
        mean_official_reward=sum(rewards) / len(selected) if selected else 0.0,
        retrieval_calls=sum(int(item.get("retrieval_calls", 0)) for item in metrics),
        tool_calls=sum(int(item.get("tool_calls", 0)) for item in metrics),
        dynamic_tool_calls=sum(
            int(item.get("dynamic_tool_calls", 0)) for item in metrics
        ),
        model_turns=sum(int(item.get("model_turns", 0)) for item in metrics),
        ptc_blocks=sum(int(item.get("ptc_blocks", 0)) for item in metrics),
        input_tokens=sum(int(item.get("input_tokens", 0)) for item in metrics),
        output_tokens=sum(int(item.get("output_tokens", 0)) for item in metrics),
        cached_input_tokens=sum(
            int(item.get("cached_input_tokens", 0)) for item in metrics
        ),
        duration_seconds=sum(
            float(item.get("duration_seconds", 0)) for item in metrics
        ),
        execution_failure_tasks=sum(
            int(item.get("execution_failures", 0)) > 0 for item in ordered
        ),
        execution_failure_blocks=sum(
            int(item.get("execution_failures", 0)) for item in ordered
        ),
        incomplete_tasks=sum(bool(item.get("incomplete")) for item in ordered),
        evaluator_failures=sum(bool(item.get("evaluator_failed")) for item in ordered),
        runner_failures=sum(item.get("status") == "failed" for item in ordered),
        run_signature=signature,
    )


def _paired_summary(
    pairs: Sequence[tuple[Mapping[str, Any], Mapping[str, Any]]],
) -> dict[str, Any]:
    deltas: list[float] = []
    for graph, baseline in pairs:
        graph_reward = _official_reward(graph)
        baseline_reward = _official_reward(baseline)
        if graph_reward is None or baseline_reward is None:
            continue
        deltas.append(graph_reward - baseline_reward)
    return {
        "pairs": len(pairs),
        "evaluable_pairs": len(deltas),
        "unevaluated_pairs": len(pairs) - len(deltas),
        "graphptc_wins": sum(value > 0 for value in deltas),
        "graphptc_losses": sum(value < 0 for value in deltas),
        "ties": sum(value == 0 for value in deltas),
        "mean_reward_delta": sum(deltas) / len(deltas) if deltas else None,
    }


def _official_reward(record: Mapping[str, Any]) -> float | None:
    if record.get("status") != "finished":
        return None
    if record.get("evaluator_failed"):
        return None
    reward = record.get("reward")
    return None if reward is None else float(reward)


def _task_pair_record(
    key: tuple[str, str, int],
    graph: Mapping[str, Any],
    baseline: Mapping[str, Any],
) -> dict[str, Any]:
    graph_reward = _official_reward(graph)
    baseline_reward = _official_reward(baseline)
    evaluable = graph_reward is not None and baseline_reward is not None
    return {
        "task_id": key[1],
        "trial": key[2],
        "evaluable": evaluable,
        "graphptc_status": graph.get("status"),
        "fewshot_ptc_status": baseline.get("status"),
        "graphptc_reward": graph_reward,
        "fewshot_ptc_reward": baseline_reward,
        "reward_delta": graph_reward - baseline_reward if evaluable else None,
        "graphptc_evaluator_failed": bool(graph.get("evaluator_failed")),
        "fewshot_ptc_evaluator_failed": bool(baseline.get("evaluator_failed")),
        "graphptc_runner_error": graph.get("runner_error"),
        "fewshot_ptc_runner_error": baseline.get("runner_error"),
    }


def _arm_metrics(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    rows = list(records)
    rewards = [float(item.get("reward") or 0) for item in rows]
    runtime = [item.get("runtime_metrics") or {} for item in rows]
    return {
        "tasks": len(rows),
        "passed": sum(value == 1.0 for value in rewards),
        "pass_hat_1": sum(value == 1.0 for value in rewards) / len(rows)
        if rows
        else 0.0,
        "mean_official_reward": sum(rewards) / len(rows) if rows else 0.0,
        **{
            key: sum(float(item.get(key, 0)) for item in runtime)
            for key in (
                "retrieval_calls",
                "tool_calls",
                "unlock_calls",
                "dynamic_tool_calls",
                "state_change_calls",
                "model_turns",
                "ptc_blocks",
                "input_tokens",
                "output_tokens",
                "cached_input_tokens",
                "duration_seconds",
            )
        },
        "execution_failure_tasks": sum(
            int(item.get("execution_failures", 0)) > 0 for item in rows
        ),
        "execution_failure_blocks": sum(
            int(item.get("execution_failures", 0)) for item in rows
        ),
        "incomplete_tasks": sum(bool(item.get("incomplete")) for item in rows),
        "evaluator_failures": sum(bool(item.get("evaluator_failed")) for item in rows),
        "runner_failures": sum(item.get("status") == "failed" for item in rows),
        "partial_artifact_tasks": sum(
            bool(item.get("partial_artifact_metrics")) for item in rows
        ),
    }


def _operational_deltas(
    graph: Sequence[Mapping[str, Any]], baseline: Sequence[Mapping[str, Any]]
) -> dict[str, float]:
    graph_metrics = _arm_metrics(graph)
    baseline_metrics = _arm_metrics(baseline)
    return {
        key: float(graph_metrics[key]) - float(baseline_metrics[key])
        for key in (
            "retrieval_calls",
            "tool_calls",
            "unlock_calls",
            "dynamic_tool_calls",
            "state_change_calls",
            "model_turns",
            "ptc_blocks",
            "input_tokens",
            "output_tokens",
            "cached_input_tokens",
            "duration_seconds",
            "execution_failure_tasks",
            "execution_failure_blocks",
            "incomplete_tasks",
            "evaluator_failures",
            "runner_failures",
            "partial_artifact_tasks",
        )
    }


def _graph_delta_mechanism(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    rows = list(records)
    histories = [
        (((item.get("telemetry") or {}).get("graph") or {}).get("action_history") or [])
        for item in rows
    ]
    later_actions = sum(max(len(history) - 1, 0) for history in histories)
    reactive = 0
    for history in histories:
        for previous, current in pairwise(history):
            previous_failed = previous.get("realized") is False
            current_action = str(current.get("action", "")).upper()
            if previous_failed and current_action in {"PATCH", "REPLAN"}:
                reactive += 1
    return {
        "tasks_with_graph_delta": sum(bool(history) for history in histories),
        "graph_deltas": sum(len(history) for history in histories),
        "deltas_preceding_later_action": later_actions,
        "unrealized_delta_followed_by_patch_or_replan": reactive,
        "causal_influence_established": False,
        "note": (
            "GRAPH_DELTA exposure and temporal follow-up are observable, but separate stochastic "
            "trajectories do not establish that the delta counterfactually changed the next action."
        ),
    }
