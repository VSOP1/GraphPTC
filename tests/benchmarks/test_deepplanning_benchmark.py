from __future__ import annotations

import math
from types import SimpleNamespace

import pytest

from graphptc.config import ExperimentConfig
from graphptc.deepplanning_benchmark import (
    PTC_GUIDANCE,
    compare_deepplanning_configs,
    probe_deepplanning_api,
)


def test_deepplanning_arms_are_matched_except_graph_mode() -> None:
    graph = ExperimentConfig.from_toml("configs/deepplanning/graphptc.toml")
    baseline = ExperimentConfig.from_toml("configs/deepplanning/fewshot-ptc.toml")
    assert compare_deepplanning_configs(graph, baseline)["matched"] is True
    assert graph.runtime.max_stdout_chars == baseline.runtime.max_stdout_chars == 8000
    assert graph.runtime.max_turns == baseline.runtime.max_turns == 400
    assert math.isinf(graph.runtime.task_timeout_seconds)
    assert math.isinf(baseline.runtime.task_timeout_seconds)
    assert graph.model.max_retries == baseline.model.max_retries == 29
    assert graph.model.retry_backoff_seconds == baseline.model.retry_backoff_seconds == 1.5
    assert graph.model.retry_all_errors is baseline.model.retry_all_errors is True
    assert graph.runtime.graph_inspection_enabled is baseline.runtime.graph_inspection_enabled is False


def test_prompt_states_persistent_runtime_and_final_output_contract() -> None:
    assert "persist between blocks" in PTC_GUIDANCE
    assert "without another programmatic_tool_call" in PTC_GUIDANCE
    assert "<plan>...</plan>" in PTC_GUIDANCE
    assert "official cart" in PTC_GUIDANCE


def test_api_probe_uses_raw_requests_and_selects_highest_clean_level(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    class Completions:
        def create(self, **kwargs):
            return SimpleNamespace(
                choices=[SimpleNamespace(message=SimpleNamespace(content="OK", tool_calls=[]))],
                usage=SimpleNamespace(prompt_tokens=3, completion_tokens=1),
            )

    class Client:
        def __init__(self, **kwargs):
            assert kwargs["max_retries"] == 0
            self.chat = SimpleNamespace(completions=Completions())

    monkeypatch.setenv("MIMO_API_KEY", "test-key")
    monkeypatch.setattr("graphptc.deepplanning_benchmark.OpenAI", Client)
    config = ExperimentConfig.from_toml("configs/deepplanning/graphptc.toml")
    output = tmp_path / "probe.json"

    report = probe_deepplanning_api(
        config, concurrencies=(2, 4), waves=2, output=output
    )

    assert report["highest_stable_total_concurrency"] == 4
    assert report["recommended_workers_per_arm"] == 2
    assert report["transport_retries"] == 0
    assert output.exists()
