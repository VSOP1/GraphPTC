from __future__ import annotations

import json
from dataclasses import replace
from types import SimpleNamespace

import pytest

from graphptc.benchmarks.appworld import benchmark as appworld_benchmark
from graphptc.benchmarks.appworld.benchmark import (
    APPWORLD_DIRECT_TOOL_SPECS,
    _appworld_direct_functions,
    _appworld_prompt_bundle,
    _appworld_ptc_spec,
)
from graphptc.config import ExperimentConfig


def test_ptc_semantic_prompt_preserves_programmatic_tool_call_semantics() -> None:
    system_prompt, demonstrations = _appworld_prompt_bundle("appworld-ptc-semantics")
    normalized = " ".join(system_prompt.split())

    assert "semantically coherent phase" in normalized
    assert "Only printed stdout" in normalized
    assert "loops, pagination, filtering, joins, and aggregation" in normalized
    assert "new semantic decision" in normalized
    assert "successful login does not create implicit authentication state" in normalized
    assert "Before first using an API" in normalized
    assert "phone contacts" in normalized
    assert "current date or time" in normalized
    assert "file-system app" in normalized
    assert "minimal direct value" in normalized
    assert demonstrations == ()

    code_description = _appworld_ptc_spec(
        ExperimentConfig.from_toml(
            "configs/appworld/appworld.graphptc-test-normal.toml"
        )
    )["function"]["parameters"]["properties"]["code"]["description"]
    assert "directly in the persistent AppWorld shell" in code_description
    assert "Registered research functions" not in code_description


def test_appworld_fewshot_uses_only_outer_ptc_calls_and_graph_intent() -> None:
    system_prompt, demonstrations = _appworld_prompt_bundle("appworld-ptc-fewshot")

    assert "semantically coherent phase" in system_prompt
    tool_calls = [
        call
        for message in demonstrations
        for call in message.get("tool_calls", ())
    ]
    assert len(tool_calls) == 2
    assert {call["function"]["name"] for call in tool_calls} == {
        "programmatic_tool_call"
    }
    payloads = [json.loads(call["function"]["arguments"]) for call in tool_calls]
    assert all(payload["action"] == "CONTINUE" for payload in payloads)
    assert all(payload["target"] == "task" for payload in payloads)
    assert all(payload["expected_change"] for payload in payloads)
    assert "for page_index in range" in payloads[-1]["code"]
    assert "complete_task(answer=" in payloads[-1]["code"]


def test_appworld_fewshot_baseline_removes_only_graph_contract() -> None:
    graph_prompt, graph_demonstrations = _appworld_prompt_bundle(
        "appworld-ptc-fewshot",
        graph_adaptation_mode="generic",
    )
    baseline_prompt, baseline_demonstrations = _appworld_prompt_bundle(
        "appworld-ptc-fewshot",
        graph_adaptation_mode="off",
    )

    assert "semantically coherent phase" in baseline_prompt
    assert "graph-control fields" in graph_prompt
    assert "graph-control fields" not in baseline_prompt
    graph_payloads = [
        json.loads(call["function"]["arguments"])
        for message in graph_demonstrations
        for call in message.get("tool_calls", ())
    ]
    baseline_payloads = [
        json.loads(call["function"]["arguments"])
        for message in baseline_demonstrations
        for call in message.get("tool_calls", ())
    ]
    assert [payload["code"] for payload in baseline_payloads] == [
        payload["code"] for payload in graph_payloads
    ]
    assert all(set(payload) == {"code"} for payload in baseline_payloads)
    assert all(
        "GRAPH_DELTA" not in str(message.get("content", ""))
        for message in baseline_demonstrations
    )

    config = ExperimentConfig.from_toml(
        "configs/appworld/appworld.graphptc-test-normal.toml"
    )
    baseline_config = replace(
        config,
        runtime=replace(config.runtime, graph_adaptation_mode="off"),
    )
    baseline_spec = _appworld_ptc_spec(baseline_config)
    assert "action" not in baseline_spec["function"]["parameters"]["properties"]


def test_unknown_appworld_prompt_variant_is_rejected() -> None:
    with pytest.raises(ValueError, match="unsupported AppWorld prompt variant"):
        _appworld_prompt_bundle("unknown")


def test_appworld_direct_tools_expose_docs_and_single_api_calls() -> None:
    system_prompt, demonstrations = _appworld_prompt_bundle(
        "appworld-direct-tools-v1",
        graph_adaptation_mode="off",
    )
    names = [spec["function"]["name"] for spec in APPWORLD_DIRECT_TOOL_SPECS]

    assert names == ["appworld_api_docs", "appworld_api_call"]
    assert "programmatic_tool_call" not in system_prompt
    assert "complete_task" in system_prompt
    assert demonstrations == ()

    with pytest.raises(ValueError, match="requires"):
        _appworld_prompt_bundle(
            "appworld-direct-tools-v1",
            graph_adaptation_mode="generic",
        )


def test_appworld_direct_functions_dispatch_only_valid_api_names() -> None:
    executed: list[str] = []

    class FakeRuntime:
        def execute(self, code: str) -> SimpleNamespace:
            executed.append(code)
            return SimpleNamespace(
                timed_out=False,
                return_code=0,
                stdout='{"ok": true}',
                stderr="",
            )

    functions = _appworld_direct_functions(FakeRuntime())  # type: ignore[arg-type]

    assert functions["appworld_api_docs"](app_name="gmail", api_name="search_emails") == {
        "ok": True
    }
    assert functions["appworld_api_call"](
        app_name="supervisor",
        api_name="complete_task",
        arguments={"answer": "42"},
    ) == {"ok": True}
    assert "show_api_doc" in executed[0]
    assert "getattr(getattr(apis, 'supervisor'), 'complete_task')" in executed[1]

    with pytest.raises(ValueError, match="identifier"):
        functions["appworld_api_call"](
            app_name="supervisor; import os",
            api_name="complete_task",
            arguments={},
        )


def test_appworld_signature_records_exact_prompt_and_dirty_source_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = ExperimentConfig.from_toml(
        "configs/appworld/appworld.graphptc-test-normal.toml"
    )
    monkeypatch.setattr(appworld_benchmark, "_git_commit", lambda: "commit")
    monkeypatch.setattr(appworld_benchmark, "_git_dirty", lambda: True)
    monkeypatch.setattr(appworld_benchmark, "_source_hash", lambda: "source")

    payload = appworld_benchmark._signature_payload(
        config,
        {
            "appworld_version": "appworld",
            "data_version": "data",
            "dataset_hash": "dataset",
            "task_ids": ["task"],
        },
        ["task"],
    )

    system_prompt, demonstrations = _appworld_prompt_bundle(
        config.appworld.prompt_variant,
    )
    tool_spec = _appworld_ptc_spec(config)
    assert payload["schema_version"] == 2
    assert payload["prompt"] == {
        "variant": "appworld-ptc-fewshot",
        "system_prompt_sha256": appworld_benchmark._sha256(system_prompt),
        "demonstrations_sha256": appworld_benchmark._sha256(demonstrations),
        "tool_spec_sha256": appworld_benchmark._sha256(tool_spec),
    }
    assert payload["behavior_config_sha256"] == appworld_benchmark._sha256(
        {
            "model": payload["model"],
            "runtime": payload["runtime"],
            "appworld": payload["appworld"],
        }
    )
    assert payload["graphptc_commit"] == "commit"
    assert payload["graphptc_git_dirty"] is True
    assert payload["graphptc_source_hash"] == "source"
