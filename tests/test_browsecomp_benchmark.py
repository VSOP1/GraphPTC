from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

import graphptc.browsecomp_benchmark as browsecomp_benchmark
from graphptc.browsecomp import BrowseCompExample
from graphptc.config import ExperimentConfig
from graphptc.ptc import AgentResult


class FakeAgent:
    def __init__(self, **kwargs: Any) -> None:
        pass

    def run(self, task: str) -> AgentResult:
        return AgentResult(
            answer=f"notes\n<result>answer for {task}</result>",
            status="success",
            model_requests=1,
        )


def _config(tmp_path: Path) -> ExperimentConfig:
    config = ExperimentConfig.from_toml("configs/browsecomp.example.toml")
    return replace(
        config,
        benchmark=replace(
            config.benchmark,
            dataset_path=tmp_path / "dataset.csv",
            responses_path=tmp_path / "responses.jsonl",
            grades_path=tmp_path / "grades.jsonl",
            report_path=tmp_path / "report.json",
        ),
    )


def test_browsecomp_runner_does_not_persist_ground_truth(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _config(tmp_path)
    example = BrowseCompExample("0", "question", "secret answer", "Other")
    monkeypatch.setenv("MIMO_API_KEY", "model-key")
    monkeypatch.setenv("TAVILY_API_KEY", "search-key")
    monkeypatch.setattr(
        browsecomp_benchmark, "load_browsecomp", lambda *args, **kwargs: [example]
    )
    monkeypatch.setattr(
        browsecomp_benchmark, "OpenAIChatModel", lambda *args, **kwargs: object()
    )
    monkeypatch.setattr(
        browsecomp_benchmark, "TavilySearchTools", lambda *args, **kwargs: object()
    )
    monkeypatch.setattr(browsecomp_benchmark, "OriginalPTCAgent", FakeAgent)

    summary = browsecomp_benchmark.run_browsecomp_benchmark(config, limit=1)
    record = json.loads(config.benchmark.responses_path.read_text(encoding="utf-8"))

    assert summary.succeeded == 1
    assert record["prediction"] == "answer for question"
    assert record["problem_topic"] == "Other"
    assert "answer" not in record
    assert "canary" not in record
    assert "problem" not in record


def test_browsecomp_evaluation_counts_missing_predictions_as_zero(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _config(tmp_path)
    examples = [
        BrowseCompExample("0", "question-0", "truth", "Other"),
        BrowseCompExample("1", "question-1", "truth", "Other"),
    ]
    signature = browsecomp_benchmark._browsecomp_run_signature(config)
    config.benchmark.responses_path.write_text(
        json.dumps(
            {
                "example_id": "0",
                "prediction": "truth",
                "status": "success",
                "run_signature": signature,
            }
        )
        + "\n",
        encoding="utf-8",
    )

    class FakeJudge:
        def judge(self, prompt: str) -> str:
            return "A"

    monkeypatch.setattr(
        browsecomp_benchmark, "load_browsecomp", lambda *args, **kwargs: examples
    )
    monkeypatch.setattr(
        browsecomp_benchmark, "_create_browsecomp_judge", lambda config: FakeJudge()
    )

    result = browsecomp_benchmark.evaluate_browsecomp_benchmark(config)

    assert result.summary.total_examples == 2
    assert result.summary.correct == 1
    assert result.summary.empty_model_responses == 1
    assert result.summary.accuracy == 0.5
