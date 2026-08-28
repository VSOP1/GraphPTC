from __future__ import annotations

from graphptc.apiflow_benchmark import _validate_arm_pair
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
    assert graph.runtime.graph_inspection_enabled is False
