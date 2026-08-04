from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

import graphptc.benchmark as benchmark
from graphptc.config import ExperimentConfig
from graphptc.deepsearchqa import DeepSearchQAExample
from graphptc.ptc import AgentResult


class FakeAgent:
    def __init__(self, **kwargs: Any) -> None:
        pass

    def run(self, task: str) -> AgentResult:
        return AgentResult(
            answer=f"research notes\n<result>answer for {task}</result>",
            status="success",
            model_requests=2,
            ptc_blocks=1,
        )


def test_generation_summary_reports_ptc_efficiency() -> None:
    records = [
        {
            "status": "success",
            "agent": {
                "duration_ms": 50,
                "model_requests": 2,
                "blocks": [
                    {"success": True, "duration_ms": 5, "runtime_calls": 3},
                    {"success": False, "duration_ms": 2, "runtime_calls": 0},
                ],
                "requests": [
                    {"duration_ms": 20, "context_chars": 100},
                    {"duration_ms": 25, "context_chars": 240},
                ],
                "search_calls": [
                    {"operation": "search", "query": "Alpha", "duration_ms": 1},
                    {"operation": "search", "query": "alpha", "duration_ms": 1},
                    {"operation": "fetch", "docid": "1", "duration_ms": 1},
                ],
                "usage": {},
            },
        }
    ]

    summary = benchmark._summarize_generation(records)

    assert summary["runtime_calls"] == 3
    assert summary["multi_call_ptc_blocks"] == 1
    assert summary["zero_call_ptc_blocks"] == 1
    assert summary["mean_runtime_calls_per_ptc_block"] == 1.5
    assert summary["repeated_exact_search_queries"] == 1
    assert summary["search_calls"] == 2
    assert summary["fetch_calls"] == 1
    assert summary["tool_calls"] == 3
    assert summary["model_request_duration_ms"] == 45
    assert summary["max_context_chars"] == 240


def test_benchmark_writes_enriched_records_and_resumes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = ExperimentConfig.from_toml("configs/deepsearchqa.example.toml")
    benchmark_config = replace(
        config.benchmark,
        dataset_path=tmp_path / "dataset.csv",
        responses_path=tmp_path / "responses.jsonl",
        grades_path=tmp_path / "grades.jsonl",
        report_path=tmp_path / "report.json",
    )
    config = replace(config, benchmark=benchmark_config)
    examples = [
        DeepSearchQAExample(
            example_id=str(index),
            problem=f"question {index}",
            problem_category="Other",
            answer=f"answer {index}",
            answer_type="Single Answer",
        )
        for index in range(2)
    ]

    monkeypatch.setenv("MIMO_API_KEY", "model-key")
    monkeypatch.setenv("TAVILY_API_KEY", "search-key")
    monkeypatch.setattr(benchmark, "load_deepsearchqa", lambda *args, **kwargs: examples)
    monkeypatch.setattr(benchmark, "OpenAIChatModel", lambda *args, **kwargs: object())
    monkeypatch.setattr(benchmark, "TavilySearchTools", lambda *args, **kwargs: object())
    monkeypatch.setattr(benchmark, "OriginalPTCAgent", FakeAgent)

    first = benchmark.run_benchmark(config, limit=1)

    assert first.completed == 1
    assert first.succeeded == 1
    record = json.loads(benchmark_config.responses_path.read_text(encoding="utf-8"))
    assert record["example_id"] == "0"
    assert record["prediction"] == "answer for question 0"
    assert record["agent"]["answer"].startswith("research notes")
    assert record["agent"]["ptc_blocks"] == 1

    record["status"] = "failed"
    record["prediction"] = ""
    benchmark_config.responses_path.write_text(
        json.dumps(record) + "\n", encoding="utf-8"
    )
    monkeypatch.delenv("MIMO_API_KEY")
    with pytest.raises(ValueError, match="MIMO_API_KEY"):
        benchmark.run_benchmark(config, limit=1)
    assert json.loads(
        benchmark_config.responses_path.read_text(encoding="utf-8")
    )["status"] == "failed"

    monkeypatch.setenv("MIMO_API_KEY", "model-key")
    retried = benchmark.run_benchmark(config, limit=1)
    assert retried.completed == 1
    assert retried.succeeded == 1
    assert (
        len(benchmark_config.responses_path.read_text(encoding="utf-8").splitlines())
        == 1
    )

    monkeypatch.delenv("MIMO_API_KEY")
    monkeypatch.delenv("TAVILY_API_KEY")
    resumed = benchmark.run_benchmark(config, limit=1)

    assert resumed.completed == 0
    assert resumed.skipped_existing == 1

    changed_config = replace(
        config,
        runtime=replace(config.runtime, max_ptc_blocks=99),
    )
    with pytest.raises(ValueError, match="another or unknown run configuration"):
        benchmark.run_benchmark(changed_config, limit=1)


def test_benchmark_retries_successful_agent_output_without_result_tag(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = ExperimentConfig.from_toml("configs/deepsearchqa.example.toml")
    config = replace(
        config,
        benchmark=replace(
            config.benchmark,
            dataset_path=tmp_path / "dataset.csv",
            responses_path=tmp_path / "responses.jsonl",
            grades_path=tmp_path / "grades.jsonl",
            report_path=tmp_path / "report.json",
        ),
    )
    example = DeepSearchQAExample("0", "question", "Other", "answer", "Single Answer")

    class UntaggedAgent(FakeAgent):
        def run(self, task: str) -> AgentResult:
            return AgentResult(answer="untagged answer", status="success")

    monkeypatch.setenv("MIMO_API_KEY", "model-key")
    monkeypatch.setenv("TAVILY_API_KEY", "search-key")
    monkeypatch.setattr(benchmark, "load_deepsearchqa", lambda *args, **kwargs: [example])
    monkeypatch.setattr(benchmark, "OpenAIChatModel", lambda *args, **kwargs: object())
    monkeypatch.setattr(benchmark, "TavilySearchTools", lambda *args, **kwargs: object())
    monkeypatch.setattr(benchmark, "OriginalPTCAgent", UntaggedAgent)

    result = benchmark.run_benchmark(config, limit=1)
    record = json.loads(config.benchmark.responses_path.read_text(encoding="utf-8"))

    assert result.failed == 1
    assert record["status"] == "failed"
    assert record["prediction"] == ""
    assert "<result>" in record["error"]


def test_evaluation_reuses_valid_grades_and_retries_invalid_ones(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = ExperimentConfig.from_toml("configs/deepsearchqa.example.toml")
    config = replace(
        config,
        benchmark=replace(
            config.benchmark,
            dataset_path=tmp_path / "dataset.csv",
            responses_path=tmp_path / "responses.jsonl",
            grades_path=tmp_path / "grades.jsonl",
            report_path=tmp_path / "report.json",
        ),
    )
    examples = [
        DeepSearchQAExample(
            example_id=str(index),
            problem=f"question {index}",
            problem_category="Other",
            answer=f"answer {index}",
            answer_type="Single Answer",
        )
        for index in range(2)
    ]
    signature = benchmark._run_signature(config)
    response_records = [
        {
            "example_id": example.example_id,
            "prediction": example.answer,
            "status": "success",
            "run_signature": signature,
        }
        for example in examples
    ]
    config.benchmark.responses_path.write_text(
        "".join(json.dumps(record) + "\n" for record in response_records),
        encoding="utf-8",
    )
    monkeypatch.setattr(benchmark, "load_deepsearchqa", lambda *args, **kwargs: examples)
    monkeypatch.setenv("GOOGLE_API_KEY", "judge-key")

    class CountingJudge:
        calls = 0

        def __init__(self, **kwargs: Any) -> None:
            pass

        def judge(self, prompt: str) -> str:
            type(self).calls += 1
            return json.dumps(
                {
                    "Answer Correctness": {
                        "Explanation": "correct",
                        "Correctness Details": {"answer": True},
                        "Excessive Answers": [],
                    }
                }
            )

    monkeypatch.setattr(
        benchmark, "_create_judge", lambda config: CountingJudge()
    )
    first = benchmark.evaluate_benchmark(config)
    assert first.summary.valid_examples == 2
    assert CountingJudge.calls == 2

    monkeypatch.delenv("GOOGLE_API_KEY")
    monkeypatch.setattr(
        benchmark,
        "_create_judge",
        lambda config: pytest.fail("cached evaluation created a judge"),
    )
    cached = benchmark.evaluate_benchmark(config)
    assert cached.summary.valid_examples == 2

    grade_records = [
        json.loads(line)
        for line in config.benchmark.grades_path.read_text(
            encoding="utf-8"
        ).splitlines()
    ]
    grade_records[0]["status"] = "invalid_auto_rater_response"
    config.benchmark.grades_path.write_text(
        "".join(json.dumps(record) + "\n" for record in grade_records),
        encoding="utf-8",
    )
    CountingJudge.calls = 0
    monkeypatch.setenv("GOOGLE_API_KEY", "judge-key")
    monkeypatch.setattr(
        benchmark, "_create_judge", lambda config: CountingJudge()
    )
    retried = benchmark.evaluate_benchmark(config)

    assert retried.summary.valid_examples == 2
    assert CountingJudge.calls == 1
