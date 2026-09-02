from __future__ import annotations

import gzip
import json
from dataclasses import replace
from pathlib import Path

import pytest

import graphptc.benchmarks.browsecomp_plus.benchmark as benchmark
from graphptc.benchmarks.browsecomp_plus.dataset import BrowseCompPlusExample
from graphptc.config import ExperimentConfig


RETRIEVER_METADATA = {
    "backend": "browsecomp_plus_official_bm25",
    "top_k": 5,
    "snippet_max_tokens": 512,
    "index_revision": "test-index",
    "index_manifest_sha256": "test-manifest",
    "tokenizer": "test-tokenizer",
    "tokenizer_revision": "test-tokenizer-revision",
}


def test_original_ptc_prompt_matches_official_retriever_contract() -> None:
    prompt = benchmark.BROWSECOMP_PLUS_ORIGINAL_PTC_SYSTEM_PROMPT

    manifest = benchmark.BROWSECOMP_PLUS_RUNTIME_TOOL_MANIFEST
    assert benchmark.BROWSECOMP_PLUS_RUNTIME_TOOL_MANIFEST_JSON in prompt
    assert [tool["name"] for tool in manifest] == ["search", "fetch"]
    assert all(
        tool["allowed_callers"] == ["programmatic_tool_call"]
        for tool in manifest
    )
    assert manifest[0]["input_schema"] == {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "minLength": 1,
                "description": "The BM25 query string.",
            }
        },
        "required": ["query"],
        "additionalProperties": False,
    }
    assert manifest[1]["input_schema"]["required"] == ["docid"]
    assert manifest[1]["input_schema"]["additionalProperties"] is False
    assert "zero to five" in manifest[0]["description"]
    assert "first 512 tokenizer tokens" in prompt
    assert "there is no title or url field" in prompt
    assert "Scores are comparable only" in prompt
    assert "def collect_evidence" not in prompt
    assert "no required number of calls" in prompt

    tool_description = benchmark.BROWSECOMP_PLUS_ORIGINAL_PTC_TOOL_SPEC["function"][
        "parameters"
    ]["properties"]["code"]["description"]
    assert "PTC_ERROR" in tool_description
    assert "exactly docid, score" in tool_description
    assert "fetch(*, docid: str)" in tool_description
    assert "include it in this program" in tool_description
    assert benchmark.BROWSECOMP_PLUS_ORIGINAL_PTC_TOOL_SPEC["function"]["parameters"][
        "properties"
    ]["code"]["minLength"] == 1


def test_original_ptc_runtime_tools_are_visible_but_not_directly_callable(
    tmp_path: Path,
) -> None:
    config = _config(tmp_path)
    config = replace(
        config,
        browsecomp_plus=replace(
            config.browsecomp_plus,
            prompt_variant="original-ptc-v1",
        ),
    )

    direct_tools = [benchmark._ptc_tool_spec(config)]
    payload = benchmark._run_signature_payload(config, RETRIEVER_METADATA)

    assert [tool["function"]["name"] for tool in direct_tools] == [
        "programmatic_tool_call"
    ]
    assert payload["runtime_tool_manifest"] == (
        benchmark.BROWSECOMP_PLUS_RUNTIME_TOOL_MANIFEST
    )
    assert {tool["name"] for tool in payload["runtime_tool_manifest"]} == {
        "search",
        "fetch",
    }
    assert all("function" not in tool for tool in payload["runtime_tool_manifest"])


def test_original_ptc_v1_has_raw_semantic_prompt_without_code_skeleton(
    tmp_path: Path,
) -> None:
    config = _config(tmp_path)
    config = replace(
        config,
        browsecomp_plus=replace(
            config.browsecomp_plus,
            prompt_variant="original-ptc-v1",
        ),
    )

    payload = benchmark._run_signature_payload(config, RETRIEVER_METADATA)
    prompt = payload["system_prompt"]

    assert payload["runtime_tool_manifest"] == benchmark.BROWSECOMP_PLUS_RUNTIME_TOOL_MANIFEST
    assert payload["ptc_tool_spec"] is benchmark.BROWSECOMP_PLUS_ORIGINAL_PTC_TOOL_SPEC
    assert payload["user_prompt_template"] == (
        benchmark.BROWSECOMP_PLUS_ORIGINAL_PTC_USER_PROMPT_TEMPLATE
    )
    assert "First plan out" not in payload["user_prompt_template"]
    assert "many searches" not in payload["user_prompt_template"]
    assert "only directly callable tool is programmatic_tool_call" in prompt
    assert "Only printed stdout is returned" in prompt
    assert "multiple times" in prompt
    assert "intermediate data-processing layer" in prompt
    assert "counting, arithmetic, or joining records" in prompt
    assert "foreseeable runtime calls" in prompt
    assert "instead of dumping raw results" in prompt
    assert "another model turn or PTC block" in prompt
    assert "no fixed program template" in prompt
    assert "queries =" not in prompt
    assert "seen =" not in prompt
    assert "setdefault" not in prompt
    assert "task_specific_selection" not in prompt


def test_direct_variant_exposes_search_and_fetch_as_native_model_tools(tmp_path: Path) -> None:
    config = _config(tmp_path)
    config = replace(
        config,
        browsecomp_plus=replace(
            config.browsecomp_plus,
            prompt_variant="direct-tools-v1",
        ),
    )

    payload = benchmark._run_signature_payload(config, RETRIEVER_METADATA)
    direct = payload["direct_tool_specs"]

    assert payload["ptc_tool_spec"] is None
    assert payload["runtime_tool_manifest"] == ()
    assert [item["function"]["name"] for item in direct] == ["search", "fetch"]
    assert all(item["type"] == "function" for item in direct)
    assert direct[0]["function"]["parameters"]["required"] == ["query"]
    assert direct[1]["function"]["parameters"]["required"] == ["docid"]
    assert "programmatic_tool_call" not in payload["system_prompt"]


def test_prompt_variant_changes_frozen_run_signature(tmp_path: Path) -> None:
    config = _config(tmp_path)
    original = benchmark._run_signature(config, RETRIEVER_METADATA)
    direct = benchmark._run_signature(
        replace(
            config,
            browsecomp_plus=replace(
                config.browsecomp_plus,
                prompt_variant="direct-tools-v1",
            ),
        ),
        RETRIEVER_METADATA,
    )

    assert direct != original


def test_generic_graph_adaptation_keeps_domain_neutral_contract(tmp_path: Path) -> None:
    config = _config(tmp_path)
    adapted = replace(
        config,
        runtime=replace(config.runtime, graph_adaptation_mode="generic"),
        browsecomp_plus=replace(config.browsecomp_plus, prompt_variant="fewshot-ptc-v1"),
    )

    manifest = benchmark._runtime_tool_manifest(adapted)
    spec = benchmark._ptc_tool_spec(adapted)

    assert [item["name"] for item in manifest] == ["search", "fetch"]
    parameters = spec["function"]["parameters"]
    assert parameters["required"] == ["code", "action", "expected_change"]
    assert parameters["properties"]["action"]["enum"] == [
        "CONTINUE",
        "PATCH",
        "REPLAN",
    ]
    system_prompt, _ = benchmark._prompt_pair(adapted)
    assert "domain-neutral effect frontier" in system_prompt
    assert "graph_add_constraint" not in system_prompt
    demos = benchmark._demonstration_messages(adapted)
    calls = [
        json.loads(call["function"]["arguments"])
        for message in demos
        for call in message.get("tool_calls", ())
    ]
    assert calls
    assert all(call["action"] == "CONTINUE" for call in calls)
    assert all("expected_change" in call for call in calls)
    assert benchmark._run_signature(adapted, RETRIEVER_METADATA) != benchmark._run_signature(
        config, RETRIEVER_METADATA
    )

def test_historical_online_graph_adaptation_is_rejected(tmp_path: Path) -> None:
    config = _config(tmp_path)
    invalid = replace(
        config,
        runtime=replace(config.runtime, graph_adaptation_mode="online"),
    )

    with pytest.raises(ValueError, match="must be one of off, generic"):
        benchmark._ptc_tool_spec(invalid)


def test_unknown_prompt_variant_is_rejected(tmp_path: Path) -> None:
    config = _config(tmp_path)
    config = replace(
        config,
        browsecomp_plus=replace(config.browsecomp_plus, prompt_variant="missing"),
    )

    with pytest.raises(ValueError, match="Unknown BrowseComp-Plus prompt variant"):
        benchmark._run_signature(config, RETRIEVER_METADATA)


def test_checkpoint_archive_preserves_exact_snapshot(tmp_path: Path) -> None:
    checkpoint = tmp_path / "checkpoints" / "task.json"
    archive = tmp_path / "archive"
    payload = {
        "run_signature": "signature",
        "example_id": "qid-1",
        "next_turn": 3,
        "messages": [{"role": "user", "content": "GRAPH_PROGRESS_SNAPSHOT exact"}],
        "agent": {"ptc_blocks": 2},
    }

    benchmark._write_checkpoint_bundle(checkpoint, payload, archive_dir=archive)

    assert json.loads(checkpoint.read_text(encoding="utf-8")) == payload
    archived = list(archive.glob("*/turn-003.json.gz"))
    assert len(archived) == 1
    with gzip.open(archived[0], "rt", encoding="utf-8") as handle:
        assert json.load(handle) == payload


def _config(tmp_path: Path) -> ExperimentConfig:
    config = ExperimentConfig.from_toml(
        "configs/browsecomp_plus/browsecomp_plus.fewshot-ptc-full.toml"
    )
    (tmp_path / "questions.jsonl").write_text("fixture\n", encoding="utf-8")
    return replace(
        config,
        benchmark=replace(
            config.benchmark,
            dataset_path=tmp_path / "questions.jsonl",
            responses_path=tmp_path / "responses.jsonl",
            grades_path=tmp_path / "grades.jsonl",
            report_path=tmp_path / "report.json",
        ),
    )


def test_resume_keeps_terminal_failed_response(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _config(tmp_path)
    example = BrowseCompPlusExample("1", "question", "answer")
    monkeypatch.setattr(
        benchmark, "load_browsecomp_plus", lambda path, **kwargs: [example]
    )
    monkeypatch.setattr(
        benchmark, "_retriever_metadata", lambda config: RETRIEVER_METADATA
    )
    record = {
        "example_id": "1",
        "status": "failed",
        "prediction": "",
        "run_signature": benchmark._run_signature(config, RETRIEVER_METADATA),
    }
    config.benchmark.responses_path.write_text(
        json.dumps(record) + "\n", encoding="utf-8"
    )

    result = benchmark.run_browsecomp_plus_benchmark(config)

    assert result.completed == 0
    assert result.skipped_existing == 1
    assert json.loads(
        config.benchmark.responses_path.read_text(encoding="utf-8")
    ) == record


def test_resume_discards_only_truncated_last_jsonl_record(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _config(tmp_path)
    example = BrowseCompPlusExample("1", "question", "answer")
    monkeypatch.setattr(
        benchmark, "load_browsecomp_plus", lambda path, **kwargs: [example]
    )
    monkeypatch.setattr(
        benchmark, "_retriever_metadata", lambda config: RETRIEVER_METADATA
    )
    monkeypatch.delenv("MIMO_API_KEY", raising=False)
    config.benchmark.responses_path.write_text('{"example_id":"1"', encoding="utf-8")

    with pytest.raises(ValueError, match="MIMO_API_KEY"):
        benchmark.run_browsecomp_plus_benchmark(config)

    assert config.benchmark.responses_path.read_bytes() == b""


def test_evaluation_rejects_incomplete_responses_before_grading(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _config(tmp_path)
    examples = [
        BrowseCompPlusExample("1", "question one", "answer one"),
        BrowseCompPlusExample("2", "question two", "answer two"),
    ]
    monkeypatch.setattr(
        benchmark, "load_browsecomp_plus", lambda path, **kwargs: examples
    )
    monkeypatch.setattr(
        benchmark, "_retriever_metadata", lambda config: RETRIEVER_METADATA
    )
    record = {
        "example_id": "1",
        "status": "success",
        "prediction": "answer one",
        "run_signature": benchmark._run_signature(config, RETRIEVER_METADATA),
        "retriever": RETRIEVER_METADATA,
    }
    config.benchmark.responses_path.write_text(
        json.dumps(record) + "\n", encoding="utf-8"
    )
    monkeypatch.setattr(
        benchmark, "_create_judge", lambda config: pytest.fail("grader was created")
    )

    with pytest.raises(ValueError, match=r"missing=\['2'\]"):
        benchmark.evaluate_browsecomp_plus_benchmark(config)
