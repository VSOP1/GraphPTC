from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

import graphptc.graph_browsecomp_plus_benchmark as graph_benchmark
from graphptc.browsecomp_plus import BrowseCompPlusExample
from graphptc.config import ExperimentConfig
from graphptc.graph_agent import GraphPTCResult
from graphptc.observability import ExecutionEvent
from graphptc.ptc import AgentResult


class FakeAgent:
    calls = 0

    def __init__(self, **kwargs: Any) -> None:
        pass

    def run(self, task: str) -> GraphPTCResult:
        type(self).calls += 1
        event = ExecutionEvent(
            event_id="episode_test:event:1",
            sequence=1,
            kind="episode.finished",
            occurred_at="2026-07-31T00:00:00+00:00",
            episode_id="episode_test",
            status="success",
        )
        return GraphPTCResult(
            episode_id="episode_test",
            agent=AgentResult(
                answer=f"<result>answer for {task}</result>",
                status="success",
                model_requests=1,
            ),
            events=(event,),
        )


class FakeSearchTools:
    def search_local(self, *, query: str) -> list[dict[str, Any]]:
        return []

    def search_local_batch(
        self, *, queries: list[str]
    ) -> dict[str, list[dict[str, Any]]]:
        return {query: [] for query in queries}


def _config(tmp_path: Path) -> ExperimentConfig:
    config = ExperimentConfig.from_toml(
        "configs/graphptc_browsecomp_plus.example.toml"
    )
    return replace(
        config,
        benchmark=replace(
            config.benchmark,
            dataset_path=tmp_path / "questions.jsonl",
            responses_path=tmp_path / "responses.jsonl",
            grades_path=tmp_path / "grades.jsonl",
            report_path=tmp_path / "report.json",
        ),
        browsecomp_plus=replace(
            config.browsecomp_plus,
            index_path=tmp_path / "corpus.sqlite3",
        ),
    )


def test_graph_runner_persists_provenance_without_ground_truth_and_resumes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _config(tmp_path)
    example = BrowseCompPlusExample("769", "question", "secret answer")
    FakeAgent.calls = 0
    monkeypatch.setenv("MIMO_API_KEY", "model-key")
    monkeypatch.setattr(
        graph_benchmark, "load_browsecomp_plus", lambda *args, **kwargs: [example]
    )
    monkeypatch.setattr(graph_benchmark, "index_document_count", lambda path: 100_195)
    monkeypatch.setattr(
        graph_benchmark, "OpenAIChatModel", lambda *args, **kwargs: object()
    )
    monkeypatch.setattr(
        graph_benchmark,
        "SQLiteCorpusSearchTools",
        lambda *args, **kwargs: FakeSearchTools(),
    )
    monkeypatch.setattr(graph_benchmark, "GraphPTCAgent", FakeAgent)

    first = graph_benchmark.run_graphptc_browsecomp_plus_benchmark(
        config, limit=1, resume=False
    )
    record = json.loads(config.benchmark.responses_path.read_text(encoding="utf-8"))
    resumed = graph_benchmark.run_graphptc_browsecomp_plus_benchmark(config, limit=1)

    assert first.succeeded == 1
    assert resumed.skipped_existing == 1
    assert FakeAgent.calls == 1
    assert record["prediction"] == "answer for question"
    assert record["graphptc"] == {
        "stage": 1,
        "episode_id": "episode_test",
        "event_count": 1,
        "events_path": str(tmp_path / "events.jsonl"),
    }
    assert "answer" not in record
    assert "question" not in record
