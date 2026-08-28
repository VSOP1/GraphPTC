from graphptc.config import ExperimentConfig
from graphptc.toolhop_benchmark import (
    _validate_arm_pair,
    score_answer,
)


def test_toolhop_arms_are_matched_except_graph_adaptation() -> None:
    graph = ExperimentConfig.from_toml("configs/toolhop/graphptc.toml")
    baseline = ExperimentConfig.from_toml("configs/toolhop/fewshot-ptc.toml")
    _validate_arm_pair(graph, baseline)
    assert graph.toolhop.workers == baseline.toolhop.workers == 4
    assert graph.model.temperature == baseline.model.temperature == 0
    assert graph.runtime.graph_inspection_enabled is False


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
