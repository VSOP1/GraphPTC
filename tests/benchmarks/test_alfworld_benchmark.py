from __future__ import annotations

import json
from dataclasses import replace

import pytest

from graphptc.alfworld_benchmark import (
    _artifact_key,
    _prompt_bundle,
    _ptc_spec,
    _summarize,
    validate_alfworld_alignment,
    validate_alfworld_arm_pair,
)
from graphptc.alfworld_worker import _effect, _extract_task
from graphptc.config import ExperimentConfig


def _configs() -> tuple[ExperimentConfig, ExperimentConfig]:
    return (
        ExperimentConfig.from_toml("configs/alfworld/graphptc-valid-seen.toml"),
        ExperimentConfig.from_toml("configs/alfworld/fewshot-ptc-valid-seen.toml"),
    )


def _inspection() -> dict[str, object]:
    return {
        "alfworld_version": "0.4.2",
        "textworld_version": "1.7.0",
        "placement_command": "move OBJECT to RECEPTACLE",
        "split": "eval_in_distribution",
        "adapter_batch_size": 1,
        "official_defaults": {
            "env_type": "AlfredTWEnv",
            "domain_randomization": False,
            "task_types": [1, 2, 3, 4, 5, 6],
            "random_seed": 42,
            "training_method": "dagger",
            "eval_batch_size": 3,
            "dagger_action_space": "generation",
            "max_steps": 50,
            "num_eval_games": -1,
        },
    }


@pytest.mark.parametrize(
    ("graph_path", "baseline_path"),
    [
        (
            "configs/alfworld/graphptc-smoke.toml",
            "configs/alfworld/fewshot-ptc-smoke.toml",
        ),
        (
            "configs/alfworld/graphptc-valid-seen.toml",
            "configs/alfworld/fewshot-ptc-valid-seen.toml",
        ),
        (
            "configs/alfworld/graphptc-valid-unseen.toml",
            "configs/alfworld/fewshot-ptc-valid-unseen.toml",
        ),
    ],
)
def test_alfworld_arms_only_change_graph_control_and_output_paths(
    graph_path: str, baseline_path: str
) -> None:
    graph = ExperimentConfig.from_toml(graph_path)
    baseline = ExperimentConfig.from_toml(baseline_path)
    validate_alfworld_arm_pair(graph, baseline)
    assert set(_ptc_spec(graph)["function"]["parameters"]["properties"]) == {
        "code",
        "action",
        "target",
        "expected_change",
    }
    assert set(_ptc_spec(baseline)["function"]["parameters"]["properties"]) == {"code"}
    graph_prompt, graph_demos = _prompt_bundle(
        graph.alfworld.prompt_variant, graph_adaptation_mode="generic"
    )
    baseline_prompt, baseline_demos = _prompt_bundle(
        baseline.alfworld.prompt_variant, graph_adaptation_mode="off"
    )
    assert "graph-control" in graph_prompt
    assert "graph-control" not in baseline_prompt
    assert "move ... to ..." in graph_prompt
    assert "put ... in/on ..." not in graph_prompt
    demo_codes = [
        json.loads(call["function"]["arguments"])["code"]
        for message in graph_demos
        for call in message.get("tool_calls", ())
    ]
    assert any("move sample 1 to box 1" in code for code in demo_codes)
    assert all("put sample 1 in box 1" not in code for code in demo_codes)
    assert len(graph_demos) == len(baseline_demos) > 0
    assert "GRAPH_DELTA" not in str(baseline_demos)


def test_alignment_fails_closed_on_official_environment_defaults() -> None:
    graph, _ = _configs()
    validate_alfworld_alignment(graph, _inspection())
    changed = dict(_inspection())
    changed["official_defaults"] = {
        **changed["official_defaults"],  # type: ignore[arg-type]
        "max_steps": 49,
    }
    with pytest.raises(ValueError, match="max_steps"):
        validate_alfworld_alignment(graph, changed)

    nonzero_temperature = replace(graph, model=replace(graph.model, temperature=None))
    with pytest.raises(ValueError, match="agent_temperature"):
        validate_alfworld_alignment(nonzero_temperature, _inspection())


def test_summary_keeps_failures_in_the_official_denominator() -> None:
    summary = _summarize(
        ["success", "loss", "runner"],
        [
            {
                "task_id": "success",
                "status": "finished",
                "episode_done": True,
                "execution_failures": 1,
                "official_evaluation": {
                    "success": True,
                    "goal_condition_success_rate": 1.0,
                    "steps": 7,
                },
            },
            {
                "task_id": "loss",
                "status": "finished",
                "episode_done": True,
                "execution_failures": 0,
                "official_evaluation": {
                    "success": False,
                    "goal_condition_success_rate": 0.5,
                    "steps": 50,
                },
            },
            {"task_id": "runner", "status": "failed", "execution_failures": 0},
        ],
        "signature",
    )
    assert summary.successes == 1
    assert summary.success_rate == pytest.approx(1 / 3)
    assert summary.mean_goal_condition_success_rate == pytest.approx(0.5)
    assert summary.mean_steps == pytest.approx(19.0)
    assert summary.runner_failures == 1
    assert summary.execution_failure_tasks == 1


def test_worker_parses_official_task_text_and_action_effects() -> None:
    task, observation = _extract_task(
        "You are in a kitchen.\n\nYour task is to: put the apple in the fridge"
    )
    assert task == "put the apple in the fridge"
    assert observation == "You are in a kitchen."
    assert _effect("look") == "read"
    assert _effect("open fridge 1") == "write"
    assert _artifact_key("task/a:b") == _artifact_key("task/a:b")
    assert _artifact_key("task/a:b") != _artifact_key("task-a-b")
