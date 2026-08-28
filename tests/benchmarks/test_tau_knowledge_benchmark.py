from __future__ import annotations

from dataclasses import replace

from graphptc.config import ExperimentConfig
from graphptc.tau3_benchmark import _tau3_prompt_bundle, _tau3_ptc_spec
from graphptc.tau_knowledge_benchmark import (
    _graph_delta_mechanism,
    _paired_summary,
    _with_partial_artifact_metrics,
    load_tau_knowledge_protocol,
    validate_tau_knowledge_alignment,
    validate_tau_knowledge_arm_pair,
)


def _configs(smoke: bool = True) -> tuple[ExperimentConfig, ExperimentConfig]:
    suffix = "-smoke" if smoke else ""
    return (
        ExperimentConfig.from_toml(f"configs/tau_knowledge/graphptc{suffix}.toml"),
        ExperimentConfig.from_toml(f"configs/tau_knowledge/fewshot-ptc{suffix}.toml"),
    )


def _inspection() -> dict:
    protocol = load_tau_knowledge_protocol()
    probe = {
        "visible_tool_names": ["KB_search", "unlock_discoverable_agent_tool"],
        "visible_schema_sha256": "schema",
        "hidden_tool_count": 10,
        "hidden_tool_names_sha256": "hidden",
        "hidden_names_exposed": False,
        "policy_sha256": "policy",
        "query_output_sha256": ["a", "b", "c"],
    }
    return {
        "official_commit": protocol["official_commit"],
        "source_provenance": {
            "transport": "git",
            "url": protocol["source_repository"]["url"],
            "tag": protocol["source_repository"]["tag"],
            "commit": protocol["official_commit"],
        },
        "required_runtime_files": protocol["required_runtime_files"],
        "package_version": protocol["official_version"],
        "data_verified": True,
        "official_defaults": {
            "max_steps": 200,
            "max_errors": 10,
            "seed": 300,
            "max_concurrency": 3,
            "agent_temperature": 0.0,
            "user_temperature": 0.0,
        },
        "task_count": 97,
        "task_ids": [f"task_{index:03d}" for index in range(1, 98)],
        "task_files": {
            "count": 97,
            "git_manifest_sha256": protocol["task_git_manifest_sha256"],
        },
        "knowledge_documents": {
            "count": 698,
            "git_manifest_sha256": protocol["knowledge_base"][
                "document_git_manifest_sha256"
            ],
        },
        "knowledge_prompts": {
            "count": 17,
            "git_manifest_sha256": protocol["knowledge_base"][
                "prompt_git_manifest_sha256"
            ],
        },
        "retrieval": {
            "config": "bm25",
            "config_kwargs": {"top_k": 10},
            "offline_bm25_only": True,
            "arms_identical": True,
            "graphptc_probe": probe,
            "fewshot_ptc_probe": dict(probe),
            "variant": {
                "kb_search": {
                    "type": "bm25",
                    "embedder_type": None,
                    "embedder_model": None,
                    "top_k": 10,
                    "reranker": False,
                    "reranker_min_score": 5,
                }
            },
        },
    }


def test_frozen_configs_only_change_graph_control_and_paths() -> None:
    for smoke in (True, False):
        graph, baseline = _configs(smoke)
        validate_tau_knowledge_arm_pair(graph, baseline)
        assert (
            graph.runtime.max_stdout_chars == baseline.runtime.max_stdout_chars == 8_000
        )
        assert graph.runtime.graph_inspection_enabled is False
        assert baseline.runtime.graph_inspection_enabled is False
        assert graph.tau3.domains == baseline.tau3.domains == ("banking_knowledge",)
        assert graph.tau3.trials == baseline.tau3.trials == 1
        assert graph.tau3.task_max_retries == baseline.tau3.task_max_retries == 0


def test_graph_arm_keeps_base_ptc_prompt_and_adds_only_graph_contract() -> None:
    graph, baseline = _configs()
    graph_prompt, graph_demos = _tau3_prompt_bundle(
        graph.tau3.prompt_variant,
        graph_adaptation_mode=graph.runtime.graph_adaptation_mode,
    )
    baseline_prompt, baseline_demos = _tau3_prompt_bundle(
        baseline.tau3.prompt_variant,
        graph_adaptation_mode=baseline.runtime.graph_adaptation_mode,
    )
    assert graph_prompt.startswith(baseline_prompt)
    assert "Graph control" in graph_prompt
    assert "GRAPH_DELTA" in graph_prompt
    assert "GRAPH_DELTA" not in baseline_prompt
    assert len(graph_demos) == len(baseline_demos)
    assert set(_tau3_ptc_spec(graph)["function"]["parameters"]["properties"]) == {
        "code",
        "action",
        "target",
        "expected_change",
    }
    assert set(_tau3_ptc_spec(baseline)["function"]["parameters"]["properties"]) == {
        "code"
    }


def test_alignment_requires_offline_identical_bm25_and_frozen_data() -> None:
    protocol = load_tau_knowledge_protocol()
    graph, _ = _configs()
    inspection = _inspection()
    validate_tau_knowledge_alignment(graph, inspection, protocol)

    bad = {
        **inspection,
        "retrieval": {**inspection["retrieval"], "arms_identical": False},
    }
    try:
        validate_tau_knowledge_alignment(graph, bad, protocol)
    except ValueError as exc:
        assert "probes differ" in str(exc)
    else:
        raise AssertionError("mismatched retrieval probes must fail closed")

    retried = replace(graph, tau3=replace(graph.tau3, task_max_retries=1))
    try:
        validate_tau_knowledge_alignment(retried, inspection, protocol)
    except ValueError as exc:
        assert "must not be retried" in str(exc)
    else:
        raise AssertionError("selected-task retry must fail closed")


def test_paired_reward_counts_wins_losses_and_ties() -> None:
    pairs = [
        (
            {"status": "finished", "reward": 1},
            {"status": "finished", "reward": 0},
        ),
        (
            {"status": "finished", "reward": 0},
            {"status": "finished", "reward": 1},
        ),
        (
            {"status": "finished", "reward": 1},
            {"status": "finished", "reward": 1},
        ),
    ]
    assert _paired_summary(pairs) == {
        "pairs": 3,
        "evaluable_pairs": 3,
        "unevaluated_pairs": 0,
        "graphptc_wins": 1,
        "graphptc_losses": 1,
        "ties": 1,
        "mean_reward_delta": 0.0,
    }


def test_paired_reward_excludes_runner_and_evaluator_failures() -> None:
    pairs = [
        (
            {"status": "finished", "reward": 0.0, "evaluator_failed": False},
            {"status": "finished", "reward": 0.0, "evaluator_failed": False},
        ),
        (
            {"status": "failed", "runner_error": "budget"},
            {"status": "failed", "runner_error": "budget"},
        ),
        (
            {"status": "finished", "reward": 0.0, "evaluator_failed": True},
            {"status": "finished", "reward": 0.0, "evaluator_failed": False},
        ),
    ]
    assert _paired_summary(pairs) == {
        "pairs": 3,
        "evaluable_pairs": 1,
        "unevaluated_pairs": 2,
        "graphptc_wins": 0,
        "graphptc_losses": 0,
        "ties": 1,
        "mean_reward_delta": 0.0,
    }


def test_failed_record_uses_saved_partial_artifact_metrics(tmp_path) -> None:
    graph, _ = _configs()
    graph = replace(
        graph,
        tau3=replace(
            graph.tau3,
            artifact_dir=tmp_path / "artifacts",
            graph_dir=tmp_path / "graphs",
        ),
    )
    task_key = "task_046-b405fd5b732c"
    agent_path = (
        graph.tau3.artifact_dir
        / "banking_knowledge"
        / task_key
        / "trial-0.agent.json"
    )
    agent_path.parent.mkdir(parents=True)
    agent_path.write_text(
        """{
          "blocks": [{"runtime_trace": {"external_actions": [
            {"name": "KB_search", "state_changed": false},
            {"name": "unlock_discoverable_agent_tool", "state_changed": true}
          ]}}],
          "telemetry": {"model_requests": 2, "execution_failures": 1,
            "usage": {"input_tokens": 10, "output_tokens": 3,
              "cached_input_tokens": 4}}
        }""",
        encoding="utf-8",
    )
    enriched = _with_partial_artifact_metrics(
        graph,
        {
            "domain": "banking_knowledge",
            "task_id": "task_046",
            "trial": 0,
            "status": "failed",
            "started_at": "2026-01-01T00:00:00+00:00",
            "finished_at": "2026-01-01T00:00:02+00:00",
        },
    )
    assert enriched["partial_artifact_metrics"] is True
    assert enriched["execution_failures"] == 1
    assert enriched["runtime_metrics"] == {
        "model_turns": 2,
        "ptc_blocks": 1,
        "tool_calls": 2,
        "retrieval_calls": 1,
        "unlock_calls": 1,
        "dynamic_tool_calls": 0,
        "state_change_calls": 1,
        "input_tokens": 10,
        "output_tokens": 3,
        "cached_input_tokens": 4,
        "duration_seconds": 2.0,
    }


def test_graph_delta_report_does_not_claim_counterfactual_influence() -> None:
    rows = [
        {
            "telemetry": {
                "graph": {
                    "action_history": [
                        {"action": "CONTINUE", "realized": False},
                        {"action": "PATCH", "realized": True},
                    ]
                }
            }
        }
    ]
    mechanism = _graph_delta_mechanism(rows)
    assert mechanism["graph_deltas"] == 2
    assert mechanism["deltas_preceding_later_action"] == 1
    assert mechanism["unrealized_delta_followed_by_patch_or_replan"] == 1
    assert mechanism["causal_influence_established"] is False
