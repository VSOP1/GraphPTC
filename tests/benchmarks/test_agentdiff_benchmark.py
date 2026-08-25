from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

from graphptc.agentdiff_benchmark import (
    AGENT_DIFF_BASE_PROMPT,
    _agentdiff_prompt_bundle,
    _agentdiff_ptc_spec,
    _summarize,
)
from graphptc.config import ExperimentConfig


def test_prompt_preserves_official_no_docs_and_shared_ptc_semantics() -> None:
    graph_prompt, graph_demos = _agentdiff_prompt_bundle(
        "agent-diff-ptc-fewshot",
        graph_adaptation_mode="generic",
        documentation_condition="no-docs",
    )
    baseline_prompt, baseline_demos = _agentdiff_prompt_bundle(
        "agent-diff-ptc-fewshot",
        graph_adaptation_mode="off",
        documentation_condition="no-docs",
    )

    for prompt in (graph_prompt, baseline_prompt):
        assert "official API base URL" in prompt
        assert "Authentication is intercepted" in prompt
        assert "one semantically coherent phase" in prompt
        assert "Python variables do not persist" in prompt
        assert "API documentation" not in prompt
    assert AGENT_DIFF_BASE_PROMPT in graph_prompt
    assert "GRAPH_DELTA" in graph_prompt
    assert "GRAPH_DELTA" not in baseline_prompt
    assert len(graph_demos) == len(baseline_demos) > 0


def test_two_arms_only_change_graph_control_and_outputs() -> None:
    config = ExperimentConfig.from_toml("configs/agent_diff/graphptc-smoke.toml")
    graph = replace(config, runtime=replace(config.runtime, graph_adaptation_mode="generic"))
    baseline = ExperimentConfig.from_toml("configs/agent_diff/fewshot-ptc-smoke.toml")

    assert set(_agentdiff_ptc_spec(graph)["function"]["parameters"]["properties"]) == {
        "code",
        "action",
        "target",
        "expected_change",
    }
    assert set(_agentdiff_ptc_spec(baseline)["function"]["parameters"]["properties"]) == {
        "code"
    }
    assert graph.agent_diff.workers > 1
    assert graph.agent_diff.progress_path != baseline.agent_diff.progress_path


def test_official_failures_remain_in_score_denominator() -> None:
    summary = _summarize(
        [("one", 0), ("two", 0)],
        [
            {
                "task_id": "one",
                "trial": 0,
                "status": "finished",
                "official_evaluation": {
                    "passed": True,
                    "score": 1.0,
                    "satisfied_assertions": 2,
                    "total_assertions": 2,
                    "clean": True,
                },
                "execution_failures": 0,
            },
            {"task_id": "two", "trial": 0, "status": "failed"},
        ],
        "signature",
    )

    assert summary.processed == 2
    assert summary.pass_rate == 0.5
    assert summary.assertion_weighted_score == 0.5
    assert summary.runner_failures == 1


def test_progress_is_written_to_log_not_stdout(tmp_path: Path, capsys) -> None:
    from graphptc.agentdiff_benchmark import _ProgressLog

    path = tmp_path / "progress.jsonl"
    progress = _ProgressLog(path)
    progress({"task_id": "one", "trial": 0, "status": "finished"})

    assert capsys.readouterr().out == ""
    assert json.loads(path.read_text(encoding="utf-8"))["status"] == "finished"
