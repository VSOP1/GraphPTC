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
    max_retries: int = 8
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
    toolsandbox: ToolSandboxConfig
    agent_diff: AgentDiffConfig

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

        return cls(
            model=_build(ModelConfig, raw.get("model", {})),
            search=_build(SearchConfig, raw.get("search", {})),
            runtime=_build(RuntimeConfig, raw.get("runtime", {})),
            benchmark=_build(BenchmarkConfig, benchmark),
            grader=_build(GraderConfig, raw.get("grader", {})),
            browsecomp_plus=_build(BrowseCompPlusConfig, browsecomp_plus),
            appworld=_build(AppWorldConfig, appworld),
            toolsandbox=_build(ToolSandboxConfig, toolsandbox),
            agent_diff=_build(AgentDiffConfig, agent_diff),
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
