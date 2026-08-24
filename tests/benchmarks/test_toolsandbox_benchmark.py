from __future__ import annotations

from dataclasses import replace

from graphptc.config import ExperimentConfig
from graphptc.toolsandbox_benchmark import (
    TOOL_SANDBOX_PTC_BASE_PROMPT,
    _summarize,
    _toolsandbox_prompt_bundle,
    _toolsandbox_ptc_spec,
)


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
    config = ExperimentConfig.from_toml("configs/toolsandbox/graphptc-smoke.toml")
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
