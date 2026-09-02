import json
from dataclasses import replace

import pytest

from graphptc.config import ExperimentConfig
from graphptc.benchmarks.toolhop.benchmark import (
    _load_manifest,
    _validate_arm_pair,
    score_answer,
)


def test_toolhop_arms_are_matched_except_graph_adaptation() -> None:
    graph = ExperimentConfig.from_toml("configs/toolhop/graphptc.toml")
    baseline = ExperimentConfig.from_toml("configs/toolhop/fewshot-ptc.toml")
    _validate_arm_pair(graph, baseline)
    assert graph.toolhop.workers == baseline.toolhop.workers == 4
    assert graph.model.temperature == baseline.model.temperature == 0


def test_toolhop_data_hash_has_one_manifest_source(tmp_path) -> None:
    config = ExperimentConfig.from_toml("configs/toolhop/graphptc.toml")
    assert not hasattr(config.toolhop, "data_sha256")
    assert len(_load_manifest(config)["data_sha256"]) == 64

    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "official_commit": config.toolhop.official_commit,
                "data_sha256": "invalid",
            }
        ),
        encoding="utf-8",
    )
    config = replace(
        config,
        toolhop=replace(config.toolhop, task_manifest_path=manifest_path),
    )
    with pytest.raises(ValueError, match="valid frozen data SHA-256"):
        _load_manifest(config)


def test_toolhop_official_and_strict_scoring_are_separate() -> None:
    exact = score_answer("Paris", "<answer>Paris</answer>", '"Paris"')
    shortcut = score_answer("Paris", "<answer>unknown</answer>", '"Paris, France"')
    assert exact["official_passed"] is True
    assert exact["strict_passed"] is True
    assert shortcut["official_passed"] is True
    assert shortcut["official_branch"] == "last_tool_output"
    assert shortcut["strict_passed"] is False
    date = score_answer("2017-10-16", "<answer>2017-10-16</answer>", None)
    assert date["official_passed"] is True
    assert date["strict_passed"] is True
