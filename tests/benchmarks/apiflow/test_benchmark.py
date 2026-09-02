from __future__ import annotations

import json
from dataclasses import replace

import pytest

from graphptc.benchmarks.apiflow.benchmark import _load_manifest, _validate_arm_pair
from graphptc.config import ExperimentConfig


def test_apiflow_scored_arms_are_matched_except_graph_adaptation() -> None:
    graph = ExperimentConfig.from_toml("configs/apiflow/graphptc.toml")
    baseline = ExperimentConfig.from_toml("configs/apiflow/fewshot-ptc.toml")

    _validate_arm_pair(graph, baseline)

    assert graph.apiflow.epochs == baseline.apiflow.epochs == 1
    assert graph.model.temperature == baseline.model.temperature == 1
    assert graph.model.max_retries == baseline.model.max_retries == -1
    assert graph.apiflow.workers == baseline.apiflow.workers == 4
    assert graph.runtime.task_timeout_seconds == baseline.runtime.task_timeout_seconds == 7200
    assert graph.runtime.graph_adaptation_mode == "generic"
    assert baseline.runtime.graph_adaptation_mode == "off"


def test_apiflow_bank_hash_has_one_manifest_source(tmp_path) -> None:
    config = ExperimentConfig.from_toml("configs/apiflow/graphptc.toml")
    assert not hasattr(config.apiflow, "bank_sha256")
    assert len(_load_manifest(config)["bank_sha256"]) == 64

    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps({"bank_sha256": "invalid"}), encoding="utf-8")
    config = replace(
        config,
        apiflow=replace(config.apiflow, task_manifest_path=manifest_path),
    )
    with pytest.raises(ValueError, match="valid frozen bank SHA-256"):
        _load_manifest(config)
