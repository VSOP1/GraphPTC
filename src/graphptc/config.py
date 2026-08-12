from __future__ import annotations

import os
import tomllib
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
    reuse_exact_results: bool = False
    graph_progress_mode: str = "off"
    graph_adaptation_mode: str = "off"
    graph_answer_review: bool = False


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

    @classmethod
    def from_toml(cls, path: str | Path) -> "ExperimentConfig":
        config_path = Path(path)
        with config_path.open("rb") as handle:
            raw = tomllib.load(handle)

        base = config_path.resolve().parent.parent
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

        return cls(
            model=_build(ModelConfig, raw.get("model", {})),
            search=_build(SearchConfig, raw.get("search", {})),
            runtime=_build(RuntimeConfig, raw.get("runtime", {})),
            benchmark=_build(BenchmarkConfig, benchmark),
            grader=_build(GraderConfig, raw.get("grader", {})),
            browsecomp_plus=_build(BrowseCompPlusConfig, browsecomp_plus),
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
