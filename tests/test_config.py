from __future__ import annotations

from pathlib import Path

import pytest

from graphptc.config import ConfigError, ExperimentConfig


def test_example_config_paths_resolve_from_repository() -> None:
    config = ExperimentConfig.from_toml("configs/deepsearchqa.example.toml")

    assert config.model.model == "mimo-v2.5"
    assert config.model.base_url == "https://api.xiaomimimo.com/v1"
    assert config.benchmark.dataset_path == Path.cwd() / "data/DSQA-full.csv"
    assert config.runtime.max_turns == 100
    assert config.runtime.max_ptc_blocks == 100
    assert config.runtime.task_timeout_seconds == 3_600
    assert config.search.max_tool_calls == 1_000
    assert config.model.max_retries == 8
    assert config.grader.model == "mimo-v2.5"
    assert config.grader.api_key_env == "MIMO_API_KEY"


def test_missing_key_is_explicit(monkeypatch: pytest.MonkeyPatch) -> None:
    config = ExperimentConfig.from_toml("configs/deepsearchqa.example.toml")
    monkeypatch.delenv("NOT_SET_FOR_TEST", raising=False)

    with pytest.raises(ConfigError, match="NOT_SET_FOR_TEST"):
        config.require_api_key("NOT_SET_FOR_TEST")


def test_browsecomp_config_is_isolated_from_deepsearchqa() -> None:
    config = ExperimentConfig.from_toml("configs/browsecomp.example.toml")

    assert config.benchmark.dataset_path == Path.cwd() / "data/browse_comp_test_set.csv"
    assert config.benchmark.responses_path == (
        Path.cwd() / "runs/browsecomp/official-style/responses.jsonl"
    )
    assert config.grader.max_completion_tokens == 8


def test_browsecomp_plus_config_uses_fixed_local_retriever() -> None:
    config = ExperimentConfig.from_toml("configs/browsecomp_plus.example.toml")

    assert config.benchmark.dataset_path == (
        Path.cwd() / "data/browsecomp_plus/questions.jsonl"
    )
    assert config.browsecomp_plus.index_path == (
        Path.cwd() / "data/browsecomp_plus/corpus.sqlite3"
    )
    assert config.browsecomp_plus.retriever_url == "http://127.0.0.1:8765"
    assert config.browsecomp_plus.top_k == 5
    assert config.browsecomp_plus.snippet_max_tokens == 512
    assert config.browsecomp_plus.expected_examples == 830
    assert config.browsecomp_plus.prompt_variant == "original-ptc-v1"
    assert config.runtime.max_turns == 30
    assert config.grader.max_completion_tokens == 1024


def test_original_ptc_full_config_uses_frozen_runtime_budget() -> None:
    config = ExperimentConfig.from_toml("configs/browsecomp_plus.example.toml")

    assert config.runtime.compaction_trigger_input_tokens is None
    assert config.runtime.compaction_max_tokens == 2048
    assert config.runtime.max_total_output_tokens == 61_440


def test_graphptc_stage1_config_has_separate_outputs() -> None:
    config = ExperimentConfig.from_toml(
        "configs/graphptc_browsecomp_plus.example.toml"
    )

    assert config.benchmark.responses_path == (
        Path.cwd() / "runs/browsecomp_plus/graphptc-stage1/responses.jsonl"
    )
    assert config.benchmark.workers == 1
