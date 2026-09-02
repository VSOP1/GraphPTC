from __future__ import annotations

from dataclasses import replace
from pathlib import Path

from graphptc.benchmarks.toolsandbox import benchmark as toolsandbox_benchmark
from graphptc.benchmarks.toolsandbox.benchmark import (
    TOOL_SANDBOX_DIRECT_PROMPT,
    TOOL_SANDBOX_PTC_BASE_PROMPT,
    _summarize,
    _toolsandbox_prompt_bundle,
    _toolsandbox_ptc_spec,
)
from graphptc.config import ExperimentConfig


def test_toolsandbox_prompt_preserves_official_and_ptc_semantics() -> None:
    graph_prompt, graph_demos = _toolsandbox_prompt_bundle(
        "toolsandbox-ptc-fewshot", graph_adaptation_mode="generic"
    )
    baseline_prompt, baseline_demos = _toolsandbox_prompt_bundle(
        "toolsandbox-ptc-fewshot", graph_adaptation_mode="off"
    )

    for prompt in (graph_prompt, baseline_prompt):
        assert "Ask the user for clarification" in prompt
        assert "ToolSandbox's\npersistent Python shell" in prompt
        assert "one semantically coherent phase" in prompt
        assert "Only printed stdout" in prompt
    assert "GRAPH_DELTA" in graph_prompt
    assert "GRAPH_DELTA" not in baseline_prompt
    assert len(graph_demos) == len(baseline_demos) > 0
    assert TOOL_SANDBOX_PTC_BASE_PROMPT in graph_prompt


def test_toolsandbox_arms_only_change_graph_control() -> None:
    config = ExperimentConfig.from_toml("configs/toolsandbox/graphptc.toml")
    graph = replace(
        config,
        runtime=replace(config.runtime, graph_adaptation_mode="generic"),
    )
    baseline = replace(
        config,
        runtime=replace(config.runtime, graph_adaptation_mode="off"),
    )

    graph_spec = _toolsandbox_ptc_spec(graph)
    baseline_spec = _toolsandbox_ptc_spec(baseline)
    assert set(graph_spec["function"]["parameters"]["properties"]) == {
        "code",
        "action",
        "target",
        "expected_change",
    }
    assert set(baseline_spec["function"]["parameters"]["properties"]) == {"code"}


def test_toolsandbox_direct_prompt_requires_graph_off() -> None:
    prompt, demonstrations = _toolsandbox_prompt_bundle(
        "toolsandbox-direct-tools-v1", graph_adaptation_mode="off"
    )

    assert prompt == TOOL_SANDBOX_DIRECT_PROMPT
    assert "native tools" in prompt
    assert "programmatic_tool_call" not in prompt
    assert demonstrations == ()


def test_toolsandbox_direct_request_selects_native_worker_mode(
    tmp_path: Path, monkeypatch
) -> None:
    config = ExperimentConfig.from_toml("configs/toolsandbox/graphptc.toml")
    config = replace(
        config,
        user_model=replace(config.user_model, model="frozen-user-simulator"),
        runtime=replace(config.runtime, graph_adaptation_mode="off"),
        toolsandbox=replace(
            config.toolsandbox,
            prompt_variant="toolsandbox-direct-tools-v1",
            workers=1,
            results_path=tmp_path / "results.jsonl",
            report_path=tmp_path / "report.json",
            artifact_dir=tmp_path / "artifacts",
            graph_dir=tmp_path / "graphs",
        ),
    )
    requests = []
    monkeypatch.setattr(
        toolsandbox_benchmark,
        "inspect_toolsandbox",
        lambda _: {
            "scenario_names": ["one"],
            "scenario_categories": {"one": []},
            "git_commit": "commit",
        },
    )

    def fake_worker(command, request):
        requests.append(request)
        return {
            "similarity": 1.0,
            "milestone_similarity": 1.0,
            "minefield_similarity": 0.0,
            "execution_failures": 0,
        }

    monkeypatch.setattr(toolsandbox_benchmark, "_worker_request", fake_worker)

    toolsandbox_benchmark.run_toolsandbox_benchmark(
        config, scenario_names=["one"], restart=True
    )

    assert requests[0]["agent_mode"] == "direct_tools"
    assert requests[0]["ptc_tool_spec"] is None
    assert requests[0]["demonstration_messages"] == ()
    assert requests[0]["agent_model"] == config.model.__dict__
    assert requests[0]["user_model"] == config.user_model.__dict__
    assert requests[0]["agent_model"] != requests[0]["user_model"]


def test_runner_failure_remains_in_official_score_denominator() -> None:
    summary = _summarize(
        ["success", "failure"],
        [
            {
                "scenario_name": "success",
                "status": "finished",
                "similarity": 1.0,
                "milestone_similarity": 0.8,
                "minefield_similarity": 0.0,
                "execution_failures": 0,
            },
            {"scenario_name": "failure", "status": "failed"},
        ],
    )

    assert summary.mean_similarity == 0.5
    assert summary.mean_milestone_similarity == 0.4
    assert summary.runner_failures == 1
