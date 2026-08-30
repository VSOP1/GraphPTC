from __future__ import annotations

import os
try:
    import tomllib
except ModuleNotFoundError:  # Python 3.10 in the isolated ToolSandbox environment.
    import tomli as tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ModelConfig:
    model: str
    base_url: str | None = None
    api_key_env: str = "OPENAI_API_KEY"
    max_completion_tokens: int = 32_000
    thinking: str | None = None
    timeout_seconds: float = 600.0
    # A negative value retries retryable transport failures until the request deadline.
    max_retries: int = 8
    retry_backoff_seconds: float | None = None
    retry_all_errors: bool = False
    temperature: float | None = None
    top_p: float | None = None


@dataclass(frozen=True)
class SearchConfig:
    api_key_env: str = "TAVILY_API_KEY"
    search_depth: str = "advanced"
    max_results: int = 10
    max_tool_calls: int = 1_000
    timeout_seconds: float = 60.0


@dataclass(frozen=True)
class BrowseCompPlusConfig:
    source_browsecomp_path: Path = Path("data/browse_comp_test_set.csv")
    corpus_dir: Path = Path("data/browsecomp_plus/corpus_parquet")
    index_path: Path = Path("data/browsecomp_plus/corpus.sqlite3")
    qrels_gold_path: Path = Path("data/browsecomp_plus/qrel_golds.txt")
    qrels_evidence_path: Path = Path("data/browsecomp_plus/qrel_evidence.txt")
    retriever_url: str = "http://127.0.0.1:8765"
    retriever_timeout_seconds: float = 60.0
    top_k: int = 5
    snippet_max_tokens: int = 512
    snippet_max_chars: int = 2_048
    max_tool_calls: int = 1_000
    expected_examples: int = 830
    prompt_variant: str = "original-ptc-v1"


@dataclass(frozen=True)
class AppWorldConfig:
    root: str = ""
    dataset_name: str = "dev"
    experiment_name: str = "graphptc-dev"
    worker_command: tuple[str, ...] = ()
    results_path: Path = Path("runs/appworld/graphptc-dev/results.jsonl")
    report_path: Path = Path("runs/appworld/graphptc-dev/report.json")
    graph_dir: Path = Path("runs/appworld/graphptc-dev/graphs")
    workers: int = 1
    expected_tasks: int = 56
    prompt_variant: str = "appworld-general"


@dataclass(frozen=True)
class AlfWorldConfig:
    data_root: str = "/home/agent/.cache/alfworld"
    official_config_path: str = ""
    split: str = "eval_in_distribution"
    worker_command: tuple[str, ...] = ()
    results_path: Path = Path("runs/alfworld/graphptc/results.jsonl")
    report_path: Path = Path("runs/alfworld/graphptc/report.json")
    graph_dir: Path = Path("runs/alfworld/graphptc/graphs")
    workers: int = 3
    expected_tasks: int = 140
    seed: int = 42
    max_steps: int = 50
    prompt_variant: str = "alfworld-ptc-fewshot"
    official_version: str = "0.4.2"


@dataclass(frozen=True)
class ToolSandboxConfig:
    root: str = "/home/agent/graphptc-toolsandbox"
    worker_command: tuple[str, ...] = ()
    results_path: Path = Path("runs/toolsandbox/graphptc/results.jsonl")
    report_path: Path = Path("runs/toolsandbox/graphptc/report.json")
    artifact_dir: Path = Path("runs/toolsandbox/graphptc/artifacts")
    graph_dir: Path = Path("runs/toolsandbox/graphptc/graphs")
    workers: int = 4
    expected_scenarios: int = 1_032
    tool_backend: str = "default"
    prompt_variant: str = "toolsandbox-ptc-fewshot"


@dataclass(frozen=True)
class AgentDiffConfig:
    root: str = "/home/agent/graphptc-agent-diff"
    worker_command: tuple[str, ...] = ()
    dataset_dir: Path = Path("data/agent_diff")
    dataset_split: str = "all"
    results_path: Path = Path("runs/agent_diff/graphptc/results.jsonl")
    report_path: Path = Path("runs/agent_diff/graphptc/report.json")
    artifact_dir: Path = Path("runs/agent_diff/graphptc/artifacts")
    graph_dir: Path = Path("runs/agent_diff/graphptc/graphs")
    progress_path: Path = Path("runs/agent_diff/graphptc/progress.jsonl")
    workers: int = 8
    expected_tasks: int = 224
    trials: int = 3
    prompt_variant: str = "agent-diff-ptc-fewshot"
    documentation_condition: str = "no-docs"
    official_commit: str = "3bb9c40707df23d89e5dbc0e40c424ba38c69ff8"
    api_key_env: str = "AGENT_DIFF_API_KEY"
    base_url_env: str = "AGENT_DIFF_BASE_URL"


@dataclass(frozen=True)
class Tau3Config:
    root: str = "/home/agent/graphptc-tau3-bench"
    worker_command: tuple[str, ...] = ()
    results_path: Path = Path("runs/tau3/graphptc/results.jsonl")
    report_path: Path = Path("runs/tau3/graphptc/report.json")
    artifact_dir: Path = Path("runs/tau3/graphptc/artifacts")
    graph_dir: Path = Path("runs/tau3/graphptc/graphs")
    progress_path: Path = Path("runs/tau3/graphptc/progress.jsonl")
    domains: tuple[str, ...] = ("airline", "retail", "telecom")
    task_split_name: str = "base"
    trials: int = 4
    workers: int = 3
    max_steps: int = 200
    max_errors: int = 10
    seed: int = 300
    enforce_communication_protocol: bool = False
    task_max_retries: int = 3
    retry_delay_seconds: float = 1.0
    prompt_variant: str = "tau3-ptc-fewshot"
    official_commit: str = "fc0055dc4e0a316c3f83133267fbd6faaa770992"
    user_model: str = "openai/mimo-v2.5"
    user_api_key_env: str = "MIMO_API_KEY"
    user_base_url: str = "https://api.xiaomimimo.com/v1"


@dataclass(frozen=True)
class MCPMarkConfig:
    root: str = "/mnt/d/MCPMark"
    official_commit: str = "cd45b7f57923b9b3985467f5139927575f83141c"
    official_worker_command: tuple[str, ...] = ()
    npx_command: str = "npx"
    npm_cache_dir: str = ""
    npm_dependency_cutoff: str = ""
    pipx_command: str = "pipx"
    docker_command: str = "docker"
    postgres_pip_constraints: Path = Path(
        "data/mcpmark/postgres-mcp-0.3.0-constraints.txt"
    )
    platform_provenance_path: Path = Path(
        "runs/mcpmark/platform-verification/meta.json"
    )
    env_path: Path = Path(".mcp_env")
    task_manifest_path: Path = Path("data/mcpmark/standard-127.json")
    results_path: Path = Path("runs/mcpmark/results.jsonl")
    report_path: Path = Path("runs/mcpmark/report.json")
    artifact_dir: Path = Path("runs/mcpmark/artifacts")
    graph_dir: Path = Path("runs/mcpmark/graphs")
    progress_path: Path = Path("runs/mcpmark/progress.jsonl")
    task_suite: str = "standard"
    expected_tasks: int = 127
    task_ids: tuple[str, ...] = ()
    workers: int = 1
    k: int = 1
    prompt_variant: str = "mcpmark-ptc-fewshot"


@dataclass(frozen=True)
class APIFlowConfig:
    root: str = "D:/APIFlow-Bench-v1.0/APIFlow-Bench-1.0"
    official_worker_command: tuple[str, ...] = ()
    bank_path: Path = Path("D:/APIFlow-Bench-v1.0/APIFlow-Bench-1.0/tasks/v1.0")
    bank_sha256: str = "abc3a823386b7f755e326017191a2e42596bf884ed8e60c44ac1d6e1cc0b615e"
    task_manifest_path: Path = Path("data/apiflow/v1.0-467.json")
    results_path: Path = Path("runs/apiflow/results.jsonl")
    report_path: Path = Path("runs/apiflow/report.json")
    artifact_dir: Path = Path("runs/apiflow/artifacts")
    graph_dir: Path = Path("runs/apiflow/graphs")
    progress_path: Path = Path("runs/apiflow/progress.jsonl")
    expected_tasks: int = 467
    epochs: int = 1
    workers: int = 1
    prompt_variant: str = "apiflow-ptc-fewshot"


@dataclass(frozen=True)
class ToolHopConfig:
    root: str = "D:/ToolHopSource"
    dataset_path: Path = Path("D:/ToolHopSource/data/ToolHop.json")
    official_worker_command: tuple[str, ...] = ()
    official_commit: str = "b439d7279af359fda46e8117ae4f0245b75f5c6b"
    data_sha256: str = "0a51f71a44b7025645e452123af3caf2e348301922af91778e268db0188a7fab"
    task_manifest_path: Path = Path("data/toolhop/mandatory-995.json")
    results_path: Path = Path("runs/toolhop/results.jsonl")
    report_path: Path = Path("runs/toolhop/report.json")
    artifact_dir: Path = Path("runs/toolhop/artifacts")
    graph_dir: Path = Path("runs/toolhop/graphs")
    progress_path: Path = Path("runs/toolhop/progress.jsonl")
    scenario: str = "Mandatory"
    expected_tasks: int = 995
    epochs: int = 1
    workers: int = 1
    prompt_variant: str = "toolhop-ptc-fewshot"


@dataclass(frozen=True)
class FanOutQAConfig:
    split: str = "dev"
    setting: str = "openbook"
    wikipedia_type: str = "kiwix"
    kiwix_base: str = "http://localhost:8888"
    kiwix_zimname: str = "wikipedia_en_all_nopic_2023-09"
    search_results: int = 10
    expected_tasks: int = 310
    workers: int = 4
    wiki_cache_dir: Path = Path("runs/fanoutqa/graphptc-dev/wiki-cache")
    results_path: Path = Path("runs/fanoutqa/graphptc-dev/results.jsonl")
    submission_path: Path = Path("runs/fanoutqa/graphptc-dev/submission.jsonl")
    grades_path: Path = Path("runs/fanoutqa/graphptc-dev/grades.jsonl")
    report_path: Path = Path("runs/fanoutqa/graphptc-dev/report.json")
    artifact_dir: Path = Path("runs/fanoutqa/graphptc-dev/artifacts")
    graph_dir: Path = Path("runs/fanoutqa/graphptc-dev/graphs")
    prompt_variant: str = "fanoutqa-ptc-fewshot"


@dataclass(frozen=True)
class DeepPlanningConfig:
    root: str = "D:/GraphPTC-DeepPlanning"
    python_command: str = "D:/GraphPTC-DeepPlanning/.venv/python.exe"
    official_commit: str = "31a4d36d123688581a9e9744427272b33ce940e0"
    data_revision: str = "213876cce679f993a476d01042e13d111c0e3648"
    results_dir: Path = Path("runs/deepplanning/graphptc")
    progress_path: Path = Path("runs/deepplanning/graphptc/progress.jsonl")
    workers: int = 1
    run_count: int = 4
    expected_travel_tasks_per_language: int = 120
    expected_shopping_tasks: tuple[int, int, int] = (50, 50, 20)
    max_model_calls: int = 400
    conversion_model: str = "qwen-plus"
    prompt_variant: str = "deepplanning-ptc-fewshot"


@dataclass(frozen=True)
class RuntimeConfig:
    max_turns: int = 100
    max_ptc_blocks: int = 100
    task_timeout_seconds: float = 3_600.0
    code_timeout_seconds: float = 300.0
    max_stdout_chars: int = 60_000
    finalization_max_tokens: int = 4_096
    compaction_trigger_input_tokens: int | None = None
    compaction_max_tokens: int = 4_096
    max_compactions: int = 8
    max_total_output_tokens: int | None = None
    graph_adaptation_mode: str = "off"
    graph_inspection_enabled: bool = False


@dataclass(frozen=True)
class BenchmarkConfig:
    dataset_path: Path = Path("data/DSQA-full.csv")
    responses_path: Path = Path("runs/deepsearchqa/responses.jsonl")
    grades_path: Path = Path("runs/deepsearchqa/grades.jsonl")
    report_path: Path = Path("runs/deepsearchqa/report.json")
    workers: int = 1


@dataclass(frozen=True)
class GraderConfig:
    provider: str = "openai_compatible"
    model: str = "mimo-v2.5"
    base_url: str | None = None
    api_key_env: str = "MIMO_API_KEY"
    workers: int = 5
    max_retries: int = 2
    max_completion_tokens: int = 8_000
    thinking: str | None = "disabled"
    timeout_seconds: float = 120.0


@dataclass(frozen=True)
class ExperimentConfig:
    model: ModelConfig
    search: SearchConfig
    runtime: RuntimeConfig
    benchmark: BenchmarkConfig
    grader: GraderConfig
    browsecomp_plus: BrowseCompPlusConfig
    appworld: AppWorldConfig
    alfworld: AlfWorldConfig
    toolsandbox: ToolSandboxConfig
    agent_diff: AgentDiffConfig
    tau3: Tau3Config
    mcpmark: MCPMarkConfig
    apiflow: APIFlowConfig
    toolhop: ToolHopConfig
    fanoutqa: FanOutQAConfig
    deepplanning: DeepPlanningConfig

    @classmethod
    def from_toml(cls, path: str | Path) -> "ExperimentConfig":
        config_path = Path(path)
        with config_path.open("rb") as handle:
            raw = tomllib.load(handle)

        base = _repository_root(config_path.resolve())
        benchmark = dict(raw.get("benchmark", {}))
        for key in ("dataset_path", "responses_path", "grades_path", "report_path"):
            value = benchmark.get(key)
            if value is not None:
                candidate = Path(value)
                benchmark[key] = candidate if candidate.is_absolute() else base / candidate

        browsecomp_plus = dict(raw.get("browsecomp_plus", {}))
        for key in (
            "source_browsecomp_path",
            "corpus_dir",
            "index_path",
            "qrels_gold_path",
            "qrels_evidence_path",
        ):
            value = browsecomp_plus.get(key)
            if value is not None:
                candidate = Path(value)
                browsecomp_plus[key] = (
                    candidate if candidate.is_absolute() else base / candidate
                )

        appworld = dict(raw.get("appworld", {}))
        for key in ("results_path", "report_path", "graph_dir"):
            value = appworld.get(key)
            if value is not None:
                candidate = Path(value)
                appworld[key] = candidate if candidate.is_absolute() else base / candidate
        if "worker_command" in appworld:
            appworld["worker_command"] = tuple(appworld["worker_command"])

        alfworld = dict(raw.get("alfworld", {}))
        for key in ("results_path", "report_path", "graph_dir"):
            value = alfworld.get(key)
            if value is not None:
                candidate = Path(value)
                alfworld[key] = candidate if candidate.is_absolute() else base / candidate
        if "worker_command" in alfworld:
            alfworld["worker_command"] = tuple(alfworld["worker_command"])

        toolsandbox = dict(raw.get("toolsandbox", {}))
        for key in ("results_path", "report_path", "artifact_dir", "graph_dir"):
            value = toolsandbox.get(key)
            if value is not None:
                candidate = Path(value)
                toolsandbox[key] = candidate if candidate.is_absolute() else base / candidate
        if "worker_command" in toolsandbox:
            toolsandbox["worker_command"] = tuple(toolsandbox["worker_command"])

        agent_diff = dict(raw.get("agent_diff", {}))
        for key in (
            "dataset_dir",
            "results_path",
            "report_path",
            "artifact_dir",
            "graph_dir",
            "progress_path",
        ):
            value = agent_diff.get(key)
            if value is not None:
                candidate = Path(value)
                agent_diff[key] = candidate if candidate.is_absolute() else base / candidate
        if "worker_command" in agent_diff:
            agent_diff["worker_command"] = tuple(agent_diff["worker_command"])

        tau3 = dict(raw.get("tau3", {}))
        for key in (
            "results_path",
            "report_path",
            "artifact_dir",
            "graph_dir",
            "progress_path",
        ):
            value = tau3.get(key)
            if value is not None:
                candidate = Path(value)
                tau3[key] = candidate if candidate.is_absolute() else base / candidate
        for key in ("worker_command", "domains"):
            if key in tau3:
                tau3[key] = tuple(tau3[key])

        mcpmark = dict(raw.get("mcpmark", {}))
        for key in (
            "env_path",
            "task_manifest_path",
            "results_path",
            "report_path",
            "artifact_dir",
            "graph_dir",
            "progress_path",
            "postgres_pip_constraints",
            "platform_provenance_path",
        ):
            value = mcpmark.get(key)
            if value is not None:
                candidate = Path(value)
                mcpmark[key] = candidate if candidate.is_absolute() else base / candidate
        for key in ("official_worker_command", "task_ids"):
            if key in mcpmark:
                mcpmark[key] = tuple(mcpmark[key])

        apiflow = dict(raw.get("apiflow", {}))
        for key in (
            "bank_path",
            "task_manifest_path",
            "results_path",
            "report_path",
            "artifact_dir",
            "graph_dir",
            "progress_path",
        ):
            value = apiflow.get(key)
            if value is not None:
                candidate = Path(value)
                apiflow[key] = candidate if candidate.is_absolute() else base / candidate
        if "official_worker_command" in apiflow:
            apiflow["official_worker_command"] = tuple(apiflow["official_worker_command"])

        toolhop = dict(raw.get("toolhop", {}))
        for key in (
            "dataset_path",
            "task_manifest_path",
            "results_path",
            "report_path",
            "artifact_dir",
            "graph_dir",
            "progress_path",
        ):
            value = toolhop.get(key)
            if value is not None:
                candidate = Path(value)
                toolhop[key] = candidate if candidate.is_absolute() else base / candidate
        if "official_worker_command" in toolhop:
            toolhop["official_worker_command"] = tuple(toolhop["official_worker_command"])

        fanoutqa = dict(raw.get("fanoutqa", {}))
        for key in (
            "wiki_cache_dir",
            "results_path",
            "submission_path",
            "grades_path",
            "report_path",
            "artifact_dir",
            "graph_dir",
        ):
            value = fanoutqa.get(key)
            if value is not None:
                candidate = Path(value)
                fanoutqa[key] = candidate if candidate.is_absolute() else base / candidate

        deepplanning = dict(raw.get("deepplanning", {}))
        for key in ("results_dir", "progress_path"):
            value = deepplanning.get(key)
            if value is not None:
                candidate = Path(value)
                deepplanning[key] = candidate if candidate.is_absolute() else base / candidate
        if "expected_shopping_tasks" in deepplanning:
            deepplanning["expected_shopping_tasks"] = tuple(
                deepplanning["expected_shopping_tasks"]
            )

        return cls(
            model=_build(ModelConfig, raw.get("model", {})),
            search=_build(SearchConfig, raw.get("search", {})),
            runtime=_build(RuntimeConfig, raw.get("runtime", {})),
            benchmark=_build(BenchmarkConfig, benchmark),
            grader=_build(GraderConfig, raw.get("grader", {})),
            browsecomp_plus=_build(BrowseCompPlusConfig, browsecomp_plus),
            appworld=_build(AppWorldConfig, appworld),
            alfworld=_build(AlfWorldConfig, alfworld),
            toolsandbox=_build(ToolSandboxConfig, toolsandbox),
            agent_diff=_build(AgentDiffConfig, agent_diff),
            tau3=_build(Tau3Config, tau3),
            mcpmark=_build(MCPMarkConfig, mcpmark),
            apiflow=_build(APIFlowConfig, apiflow),
            toolhop=_build(ToolHopConfig, toolhop),
            fanoutqa=_build(FanOutQAConfig, fanoutqa),
            deepplanning=_build(DeepPlanningConfig, deepplanning),
        )

    def require_api_key(self, env_name: str) -> str:
        value = os.getenv(env_name)
        if not value:
            raise ConfigError(f"Missing API key environment variable: {env_name}")
        return value


class ConfigError(ValueError):
    pass


def _build(cls: type[Any], values: dict[str, Any]) -> Any:
    try:
        return cls(**values)
    except TypeError as exc:
        raise ConfigError(f"Invalid [{cls.__name__.removesuffix('Config').lower()}] config: {exc}") from exc


def _repository_root(config_path: Path) -> Path:
    for parent in config_path.parents:
        if (parent / "pyproject.toml").exists():
            return parent
    return config_path.parent.parent
