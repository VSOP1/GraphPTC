from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

from graphptc.config import ExperimentConfig
from graphptc.tau3_benchmark import (
    TAU3_OFFICIAL_COMMIT,
    _ProgressLog,
    _safe_task_key,
    _summarize,
    _tau3_agent_name,
    _tau3_prompt_bundle,
    _tau3_ptc_spec,
    _worker_request_with_retry,
    validate_tau3_alignment,
)


def test_prompt_keeps_official_policy_and_shared_ptc_semantics() -> None:
    graph_prompt, graph_demos = _tau3_prompt_bundle(
        "tau3-ptc-fewshot", graph_adaptation_mode="generic"
    )
    baseline_prompt, baseline_demos = _tau3_prompt_bundle(
        "tau3-ptc-fewshot", graph_adaptation_mode="off"
    )
    for prompt in (graph_prompt, baseline_prompt):
        assert "authoritative domain policy" in prompt
        assert "one semantically coherent phase" in prompt
        assert "official environment" in prompt
        assert "Do not invent tool names" in prompt
    assert "GRAPH_DELTA" in graph_prompt
    assert "GRAPH_DELTA" not in baseline_prompt
    assert len(graph_demos) == len(baseline_demos) > 0


def test_arms_only_change_graph_control_and_output_paths() -> None:
    for suffix in ("-smoke", "-pilot", ""):
        graph = ExperimentConfig.from_toml(f"configs/tau3/graphptc{suffix}.toml")
        baseline = ExperimentConfig.from_toml(
            f"configs/tau3/fewshot-ptc{suffix}.toml"
        )
        assert graph.model == baseline.model
        assert replace(graph.runtime, graph_adaptation_mode="off") == baseline.runtime
        assert graph.runtime.max_stdout_chars == baseline.runtime.max_stdout_chars == 8_000
        graph_tau3 = replace(
            graph.tau3,
            results_path=baseline.tau3.results_path,
            report_path=baseline.tau3.report_path,
            artifact_dir=baseline.tau3.artifact_dir,
            graph_dir=baseline.tau3.graph_dir,
            progress_path=baseline.tau3.progress_path,
        )
        assert graph_tau3 == baseline.tau3
        assert set(_tau3_ptc_spec(graph)["function"]["parameters"]["properties"]) == {
            "code", "action", "target", "expected_change"
        }
        assert set(_tau3_ptc_spec(baseline)["function"]["parameters"]["properties"]) == {"code"}
        assert _tau3_agent_name(graph.runtime.graph_adaptation_mode) == "graphptc"
        assert _tau3_agent_name(baseline.runtime.graph_adaptation_mode) == "fewshot_ptc"


def test_alignment_fails_closed_on_official_defaults() -> None:
    config = ExperimentConfig.from_toml("configs/tau3/graphptc.toml")
    inspection = {
        "official_commit": TAU3_OFFICIAL_COMMIT,
        "package_version": "1.0.1",
        "python_version": "3.12.8",
        "data_verified": True,
        "domains": {domain: {"task_ids": ["1"]} for domain in config.tau3.domains},
        "official_defaults": {
            "max_steps": 200,
            "max_errors": 10,
            "seed": 300,
            "max_concurrency": 3,
            "agent_temperature": 0.0,
            "user_temperature": 0.0,
            "enforce_communication_protocol": False,
            "max_retries": 3,
            "retry_delay": 1.0,
        },
    }
    validate_tau3_alignment(config, inspection)
    bad = dict(inspection)
    bad["official_defaults"] = {**inspection["official_defaults"], "max_steps": 199}
    try:
        validate_tau3_alignment(config, bad)
    except ValueError as exc:
        assert "max_steps" in str(exc)
    else:
        raise AssertionError("alignment mismatch should fail")

    wrong_temperature = replace(config, model=replace(config.model, temperature=None))
    try:
        validate_tau3_alignment(wrong_temperature, inspection)
    except ValueError as exc:
        assert "temperature" in str(exc)
    else:
        raise AssertionError("agent temperature mismatch should fail")


def test_worker_retry_matches_official_task_retry_semantics(monkeypatch) -> None:
    attempts = 0

    def flaky(*args, **kwargs):
        nonlocal attempts
        attempts += 1
        if attempts < 3:
            raise RuntimeError(f"transient-{attempts}")
        return {"status": "finished", "reward": 1.0}

    monkeypatch.setattr("graphptc.tau3_benchmark._worker_request", flaky)
    response, errors = _worker_request_with_retry(
        ("worker",),
        {"type": "run"},
        timeout=10,
        env_names=("MIMO_API_KEY",),
        max_retries=3,
        retry_delay=0.0,
    )
    assert response["reward"] == 1.0
    assert attempts == 3
    assert errors == ["RuntimeError: transient-1", "RuntimeError: transient-2"]


def test_worker_retry_raises_after_initial_attempt_plus_retries(monkeypatch) -> None:
    attempts = 0

    def broken(*args, **kwargs):
        nonlocal attempts
        attempts += 1
        raise RuntimeError("offline")

    monkeypatch.setattr("graphptc.tau3_benchmark._worker_request", broken)
    try:
        _worker_request_with_retry(
            ("worker",),
            {"type": "run"},
            timeout=10,
            max_retries=3,
            retry_delay=0.0,
        )
    except RuntimeError as exc:
        assert str(exc) == "offline"
    else:
        raise AssertionError("permanent worker failure should be raised")
    assert attempts == 4


def test_official_failures_stay_in_denominator() -> None:
    summary = _summarize(
        [("airline", "1", 0), ("retail", "2", 0)],
        [
            {"domain": "airline", "task_id": "1", "trial": 0, "status": "finished", "reward": 1.0},
            {
                "domain": "retail",
                "task_id": "2",
                "trial": 0,
                "status": "failed",
                "evaluator_failed": True,
                "runner_retry_count": 3,
            },
        ],
        "sig",
    )
    assert summary.processed == 2
    assert summary.pass_hat_1 == 0.5
    assert summary.evaluator_failures == 1
    assert summary.runner_retry_tasks == 1
    assert summary.runner_retry_attempts == 3


def test_progress_goes_to_log_not_stdout(tmp_path: Path, capsys) -> None:
    progress = _ProgressLog(tmp_path / "progress.jsonl")
    progress({"domain": "airline", "task_id": "1", "trial": 0, "status": "finished"})
    assert capsys.readouterr().out == ""
    assert json.loads(progress.path.read_text(encoding="utf-8"))["status"] == "finished"


def test_task_artifact_key_is_cross_platform_safe_and_collision_resistant() -> None:
    task_id = "[mobile_data_issue]airplane_mode_on|user:abroad"
    key = _safe_task_key(task_id)
    assert key.startswith("mobile_data_issue-airplane_mode_on-user-abroad-")
    assert all(character.isalnum() or character in "-_" for character in key)
    assert key != _safe_task_key(task_id + "-different")
