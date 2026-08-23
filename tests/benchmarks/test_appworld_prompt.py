from __future__ import annotations

import json

import pytest

from dataclasses import replace

from graphptc import appworld_benchmark
from graphptc.appworld_benchmark import _appworld_prompt_bundle, _appworld_ptc_spec
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
        ExperimentConfig.from_toml("configs/appworld.graphptc-dev-fewshot-smoke.toml")
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
    assert not any("INSPECT" in json.dumps(payload) for payload in payloads)


def test_unknown_appworld_prompt_variant_is_rejected() -> None:
    with pytest.raises(ValueError, match="unsupported AppWorld prompt variant"):
        _appworld_prompt_bundle("unknown")


def test_appworld_inspection_schema_is_explicitly_opt_in() -> None:
    config = ExperimentConfig.from_toml("configs/appworld.graphptc-dev-fewshot-smoke.toml")

    disabled = _appworld_ptc_spec(config)
    enabled = _appworld_ptc_spec(
        replace(
            config,
            runtime=replace(config.runtime, graph_inspection_enabled=True),
        )
    )

    assert "INSPECT" not in disabled["function"]["parameters"]["properties"]["action"][
        "enum"
    ]
    inspection = enabled["function"]["parameters"]["properties"]["inspection"]
    assert inspection["required"] == ["view"]
    assert inspection["additionalProperties"] is False


def test_appworld_signature_records_exact_prompt_and_dirty_source_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = ExperimentConfig.from_toml(
        "configs/appworld.graphptc-dev-inspection-smoke.toml"
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
        graph_inspection_enabled=True,
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
