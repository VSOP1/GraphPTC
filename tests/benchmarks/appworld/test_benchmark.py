from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from graphptc.benchmarks.appworld import benchmark as appworld_benchmark
from graphptc.benchmarks.appworld.benchmark import _summarize, _terminal_task_ids
from graphptc.config import ExperimentConfig


def test_resume_only_skips_terminal_appworld_records() -> None:
    records = [
        {"task_id": "interrupted", "status": "started"},
        {"task_id": "finished", "status": "started"},
        {"task_id": "finished", "status": "finished"},
        {"task_id": "failed", "status": "failed"},
    ]

    assert _terminal_task_ids(records) == {"finished", "failed"}


def test_appworld_summary_separates_processing_completion_and_failures() -> None:
    records = [
        {
            "task_id": "success",
            "status": "finished",
            "task_completed": True,
            "execution_failures": 1,
            "official_evaluation": {"success": True},
            "evaluator_error": None,
        },
        {
            "task_id": "incomplete",
            "status": "finished",
            "task_completed": False,
            "execution_failures": 0,
            "official_evaluation": {"success": False},
            "evaluator_error": None,
        },
        {
            "task_id": "evaluator",
            "status": "finished",
            "task_completed": True,
            "execution_failures": 0,
            "official_evaluation": None,
            "evaluator_error": "failed",
        },
        {
            "task_id": "runner",
            "status": "failed",
            "task_completed": False,
            "execution_failures": 0,
            "official_evaluation": None,
            "evaluator_error": None,
        },
    ]

    summary = _summarize(
        ["success", "incomplete", "evaluator", "runner"], records, "signature"
    )

    assert summary.processed == 4
    assert summary.task_completed == 2
    assert summary.official_failures == 1
    assert summary.execution_failure_tasks == 1
    assert summary.execution_failure_blocks == 1
    assert summary.incomplete_tasks == 2
    assert summary.evaluator_failures == 1
    assert summary.runner_failures == 1


def test_runner_closes_world_and_marks_completed_graph(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    config = ExperimentConfig.from_toml(
        "configs/appworld/appworld.graphptc-test-normal.toml"
    )
    config = replace(
        config,
        appworld=replace(
            config.appworld,
            results_path=tmp_path / "results.jsonl",
            report_path=tmp_path / "report.json",
            graph_dir=tmp_path / "graphs",
            experiment_name="graphptc-test",
        ),
    )
    runtimes: list[Any] = []

    class FakeRuntime:
        def __init__(self, **_: Any) -> None:
            self.task_completed = False
            self.closed = False
            self.metadata = {
                "instruction": "finish the task",
                "appworld_version": "fake",
                "data_version": "fake-data",
            }
            runtimes.append(self)

        def evaluate(self) -> dict[str, Any]:
            return {"success": True}

        def close(self) -> None:
            self.closed = True

        def telemetry(self) -> dict[str, Any]:
            return {"closed": self.closed, "task_completed": self.task_completed}

    class FakeAgent:
        def __init__(self, *, program_runtime: FakeRuntime, **_: Any) -> None:
            self.runtime = program_runtime

        def run(self, _: str) -> Any:
            self.runtime.task_completed = True
            return SimpleNamespace(
                blocks=[],
                to_dict=lambda: {"status": "success", "blocks": []},
            )

    inspection = {
        "appworld_version": "fake",
        "data_version": "fake-data",
        "dataset_name": "dev",
        "dataset_hash": "dataset",
        "task_ids": ["task"],
    }
    monkeypatch.setenv("MIMO_API_KEY", "test")
    monkeypatch.setattr(appworld_benchmark, "inspect_appworld", lambda _: inspection)
    monkeypatch.setattr(appworld_benchmark, "AppWorldProgramRuntime", FakeRuntime)
    monkeypatch.setattr(appworld_benchmark, "OpenAIChatModel", lambda *args: object())
    monkeypatch.setattr(appworld_benchmark, "OriginalPTCAgent", FakeAgent)

    summary = appworld_benchmark.run_appworld_benchmark(
        config,
        task_ids=["task"],
        restart=True,
    )

    report = json.loads(config.appworld.report_path.read_text(encoding="utf-8"))
    graph = json.loads((config.appworld.graph_dir / "task.json").read_text(encoding="utf-8"))
    task_node = next(node for node in graph["nodes"] if node["id"] == "task")
    assert summary.task_completed == 1
    assert task_node["data"]["status"] == "COMPLETE"
    assert any(
        node["kind"] == "ACTION_INTENT" and node["data"]["action"] == "ANSWER"
        for node in graph["nodes"]
    )
    assert report["tasks"][0]["runtime_final"]["closed"] is True
    assert report["resolved_config"]["appworld"]["report_path"] == str(
        config.appworld.report_path
    )
    assert report["resolved_config_sha256"] == appworld_benchmark._sha256(
        report["resolved_config"]
    )
    assert runtimes[0].closed is True


def test_aggregate_evaluator_requires_the_exact_saved_run_signature(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    config = ExperimentConfig.from_toml(
        "configs/appworld/appworld.graphptc-test-normal.toml"
    )
    config = replace(
        config,
        appworld=replace(
            config.appworld,
            results_path=tmp_path / "results.jsonl",
            report_path=tmp_path / "report.json",
            graph_dir=tmp_path / "graphs",
            experiment_name="graphptc-evaluate-test",
        ),
    )
    inspection = {
        "appworld_version": "fake",
        "data_version": "fake-data",
        "dataset_name": "dev",
        "dataset_hash": "dataset",
        "task_ids": ["task"],
    }
    payload = appworld_benchmark._signature_payload(config, inspection, ["task"])
    signature = appworld_benchmark._sha256(payload)
    config.appworld.results_path.write_text(
        json.dumps(
            {
                "task_id": "task",
                "status": "finished",
                "run_signature": signature,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    config.appworld.report_path.write_text(
        json.dumps(
            {
                "summary": {"run_signature": signature},
                "run_signature_payload": payload,
                "tasks": [],
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(appworld_benchmark, "inspect_appworld", lambda _: inspection)
    monkeypatch.setattr(
        appworld_benchmark,
        "_worker_request",
        lambda *args, **kwargs: {
            "type": "aggregate_evaluation",
            "evaluation": {"task_goal_completion": 1.0},
            "appworld_version": "fake",
            "data_version": "fake-data",
        },
    )

    evaluation = appworld_benchmark.evaluate_appworld_benchmark(config)

    assert evaluation == {"task_goal_completion": 1.0}
    report = json.loads(config.appworld.report_path.read_text(encoding="utf-8"))
    assert report["official_evaluation_provenance"] == {
        "run_signature": signature,
        "appworld_version": "fake",
        "data_version": "fake-data",
        "task_ids": ["task"],
    }

    changed = replace(config, runtime=replace(config.runtime, max_turns=31))
    with pytest.raises(ValueError, match="run signature"):
        appworld_benchmark.evaluate_appworld_benchmark(changed)
