from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

from graphptc.config import ExperimentConfig
from graphptc.observability import ExecutionObserver
from graphptc.stage1 import run_stage1_browsecomp_plus


def test_stage1_adapter_reuses_fewshot_runner_and_writes_events(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = ExperimentConfig.from_toml(
        "configs/browsecomp_plus.fewshot-ptc-v1-turn30-pilot20.toml"
    )
    config = replace(
        config,
        benchmark=replace(
            config.benchmark,
            responses_path=tmp_path / "responses.jsonl",
        ),
    )
    captured: dict[str, Any] = {}
    sentinel = object()

    def fake_run(received: ExperimentConfig, **kwargs: Any) -> object:
        captured["config"] = received
        captured.update(kwargs)
        observer = kwargs["observer_factory"]("qid-1", "signature-1")
        assert isinstance(observer, ExecutionObserver)
        observer.emit("episode.started", data={"task": "question"})
        return sentinel

    monkeypatch.setattr(
        "graphptc.stage1.run_browsecomp_plus_benchmark",
        fake_run,
    )

    result = run_stage1_browsecomp_plus(
        config,
        limit=2,
        example_ids=("qid-1",),
        resume=False,
    )

    assert result is sentinel
    assert captured["config"] is config
    assert captured["limit"] == 2
    assert captured["example_ids"] == ("qid-1",)
    assert captured["resume"] is False
    events_path = tmp_path / "events.jsonl"
    event = json.loads(events_path.read_text(encoding="utf-8"))
    assert event["episode_id"] == "signature-1:qid-1"
    assert event["task_id"] == "qid-1"
    assert captured["post_episode_callback"] is None
    assert captured["active_repair_callback_factory"] is None
    assert captured["checkpoint_archive_dir"] is None


def test_stage1_passes_checkpoint_archive_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = ExperimentConfig.from_toml(
        "configs/browsecomp_plus.fewshot-ptc-v1-turn30-pilot20.toml"
    )
    captured: dict[str, Any] = {}

    def fake_run(received: ExperimentConfig, **kwargs: Any) -> object:
        captured.update(kwargs)
        return object()

    monkeypatch.setattr("graphptc.stage1.run_browsecomp_plus_benchmark", fake_run)
    archive = tmp_path / "archive"

    run_stage1_browsecomp_plus(config, checkpoint_archive_dir=archive)

    assert captured["checkpoint_archive_dir"] == archive


def test_stage1_shadow_is_opt_in_and_does_not_change_runner_result(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = ExperimentConfig.from_toml(
        "configs/browsecomp_plus.fewshot-ptc-v1-turn30-pilot20.toml"
    )
    config = replace(
        config,
        benchmark=replace(config.benchmark, responses_path=tmp_path / "responses.jsonl"),
    )
    sentinel = object()

    def fake_run(received: ExperimentConfig, **kwargs: Any) -> object:
        observer = kwargs["observer_factory"]("qid-1", "signature-1")
        observer.emit("episode.started", data={"task": "question"})
        observer.emit(
            "episode.finished",
            data={"status": "success", "answer": "done", "error": None, "ptc_blocks": 0},
        )
        kwargs["post_episode_callback"](
            "qid-1",
            "signature-1",
            {"status": "success", "prediction": "done"},
        )
        return sentinel

    monkeypatch.setattr("graphptc.stage1.run_browsecomp_plus_benchmark", fake_run)

    result = run_stage1_browsecomp_plus(
        config,
        events_path=tmp_path / "events.jsonl",
        shadow_output_path=tmp_path / "shadow.jsonl",
        resume=False,
    )

    assert result is sentinel
    record = json.loads((tmp_path / "shadow.jsonl").read_text(encoding="utf-8"))
    assert record["example_id"] == "qid-1"
    assert record["primary_status"] == "success"
    assert record["shadow"]["status"] == "no_repairable_failure"
    assert record["shadow"]["model_request_count"] == 0


def test_stage1_shadow_error_is_recorded_without_changing_runner_result(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = ExperimentConfig.from_toml(
        "configs/browsecomp_plus.fewshot-ptc-v1-turn30-pilot20.toml"
    )
    config = replace(
        config,
        benchmark=replace(config.benchmark, responses_path=tmp_path / "responses.jsonl"),
    )
    sentinel = object()

    def fake_run(received: ExperimentConfig, **kwargs: Any) -> object:
        observer = kwargs["observer_factory"]("qid-1", "signature-1")
        observer.emit("episode.started", data={"task": "question"})
        kwargs["post_episode_callback"](
            "qid-1",
            "signature-1",
            {"status": "success", "prediction": "done"},
        )
        return sentinel

    monkeypatch.setattr("graphptc.stage1.run_browsecomp_plus_benchmark", fake_run)

    result = run_stage1_browsecomp_plus(
        config,
        events_path=tmp_path / "events.jsonl",
        shadow_output_path=tmp_path / "shadow.jsonl",
        resume=False,
    )

    assert result is sentinel
    record = json.loads((tmp_path / "shadow.jsonl").read_text(encoding="utf-8"))
    assert record["primary_status"] == "success"
    assert record["shadow"]["status"] == "shadow_error"
    assert record["shadow"]["commit"] is None


def test_stage1_adapter_rejects_non_fewshot_prompt_variant(tmp_path: Path) -> None:
    config = ExperimentConfig.from_toml(
        "configs/browsecomp_plus.original-ptc-v1-turn30-pilot20.toml"
    )
    config = replace(
        config,
        browsecomp_plus=replace(
            config.browsecomp_plus,
            prompt_variant="original-ptc-v1",
        ),
    )

    with pytest.raises(ValueError, match="fewshot-ptc-v1"):
        run_stage1_browsecomp_plus(config, events_path=tmp_path / "events.jsonl")


def test_stage1_rejects_shadow_and_active_repair_together(tmp_path: Path) -> None:
    config = ExperimentConfig.from_toml(
        "configs/browsecomp_plus.fewshot-ptc-v1-turn30-pilot20.toml"
    )

    with pytest.raises(ValueError, match="mutually exclusive"):
        run_stage1_browsecomp_plus(
            config,
            events_path=tmp_path / "events.jsonl",
            shadow_output_path=tmp_path / "shadow.jsonl",
            active_repair_output_path=tmp_path / "active.jsonl",
        )
