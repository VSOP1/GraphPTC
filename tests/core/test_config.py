from __future__ import annotations

from pathlib import Path

import pytest

from graphptc.config import ExperimentConfig


@pytest.mark.parametrize(
    "path",
    [
        "configs/browsecomp_plus/browsecomp_plus.direct-tools-full.toml",
        "configs/appworld/appworld.direct-tools-test-normal.toml",
        "configs/appworld/appworld.direct-tools-test-challenge.toml",
        "configs/toolsandbox/direct-tools.toml",
        "configs/agent_diff/direct-tools.toml",
        "configs/fanoutqa/direct-tools-dev.toml",
        "configs/frames/direct-tools-test.toml",
    ],
)
def test_formal_direct_tool_configs_disable_graph_control(path: str) -> None:
    config = ExperimentConfig.from_toml(path)

    assert config.runtime.graph_adaptation_mode == "off"


@pytest.mark.parametrize(
    ("path", "section"),
    [
        ("configs/appworld/appworld.graphptc-test-normal.toml", "appworld"),
        ("configs/toolsandbox/graphptc.toml", "toolsandbox"),
        ("configs/agent_diff/graphptc.toml", "agent_diff"),
    ],
)
def test_isolated_workers_resolve_from_repository(path: str, section: str) -> None:
    config = ExperimentConfig.from_toml(path)
    benchmark = getattr(config, section)
    command = tuple(benchmark.worker_command)

    assert command
    assert all("wsl.exe" not in item and "/mnt/d/" not in item for item in command)
    assert all("{repo}" not in item for item in command)
    assert Path(command[0]).is_absolute()
    assert Path(command[1]).is_absolute()


def test_toolsandbox_has_an_independent_frozen_user_model() -> None:
    config = ExperimentConfig.from_toml("configs/toolsandbox/graphptc.toml")

    assert config.user_model is not config.model
    assert config.user_model.model == "mimo-v2.5"
    assert config.user_model.api_key_env == "MIMO_API_KEY"


def test_browsecomp_plus_config_uses_fixed_local_retriever() -> None:
    config = ExperimentConfig.from_toml(
        "configs/browsecomp_plus/browsecomp_plus.graphptc-full.toml"
    )

    assert config.benchmark.dataset_path == (
        Path.cwd() / "data/browsecomp_plus/questions.jsonl"
    )
    assert config.browsecomp_plus.index_path == (
        Path.cwd() / "data/browsecomp_plus/corpus.sqlite3"
    )
    assert config.browsecomp_plus.source_browsecomp_path == (
        Path.cwd() / "data/browsecomp_plus/browse_comp_test_set.csv"
    )
    assert config.browsecomp_plus.retriever_url == "http://127.0.0.1:8765"
    assert config.browsecomp_plus.top_k == 5
    assert config.browsecomp_plus.snippet_max_tokens == 512
    assert config.browsecomp_plus.expected_examples == 830
    assert config.browsecomp_plus.prompt_variant == "fewshot-ptc-v1"
    assert config.runtime.max_turns == 30
    assert config.grader.max_completion_tokens == 1024


def test_browsecomp_plus_full_pair_runs_one_matched_830_question_dataset() -> None:
    graph = ExperimentConfig.from_toml(
        "configs/browsecomp_plus/browsecomp_plus.graphptc-full.toml"
    )
    baseline = ExperimentConfig.from_toml(
        "configs/browsecomp_plus/browsecomp_plus.fewshot-ptc-full.toml"
    )

    expected_dataset = Path.cwd() / "data/browsecomp_plus/questions.jsonl"
    assert graph.benchmark.dataset_path == baseline.benchmark.dataset_path == expected_dataset
    assert graph.browsecomp_plus.expected_examples == baseline.browsecomp_plus.expected_examples == 830
    assert graph.browsecomp_plus == baseline.browsecomp_plus
    assert graph.model == baseline.model
    assert graph.grader == baseline.grader
    assert graph.runtime.graph_adaptation_mode == "generic"
    assert baseline.runtime.graph_adaptation_mode == "off"
    assert graph.benchmark.responses_path != baseline.benchmark.responses_path


def test_original_ptc_full_config_uses_frozen_runtime_budget() -> None:
    config = ExperimentConfig.from_toml(
        "configs/browsecomp_plus/browsecomp_plus.graphptc-full.toml"
    )

    assert config.runtime.compaction_trigger_input_tokens is None
    assert config.runtime.compaction_max_tokens == 2048
    assert config.runtime.max_total_output_tokens == 61_440
