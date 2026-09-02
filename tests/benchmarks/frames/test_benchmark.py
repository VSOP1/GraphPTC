from __future__ import annotations

from dataclasses import replace

import pytest

from graphptc.benchmarks.frames.benchmark import (
    DIRECT_TOOL_SPECS,
    DIRECT_USER_PROMPT,
    _validate_config,
)
from graphptc.config import ExperimentConfig


def test_frames_direct_config_uses_native_wikipedia_tools() -> None:
    config = ExperimentConfig.from_toml("configs/frames/direct-tools-test.toml")

    _validate_config(config)

    assert config.runtime.graph_adaptation_mode == "off"
    assert config.frames.prompt_variant == "frames-direct-tools-v1"
    assert "{task}" in DIRECT_USER_PROMPT
    assert [spec["function"]["name"] for spec in DIRECT_TOOL_SPECS] == [
        "wiki_search",
        "wiki_content",
    ]


def test_frames_direct_config_rejects_graph_control() -> None:
    config = ExperimentConfig.from_toml("configs/frames/direct-tools-test.toml")
    invalid = replace(
        config,
        runtime=replace(config.runtime, graph_adaptation_mode="generic"),
    )

    with pytest.raises(ValueError, match="requires"):
        _validate_config(invalid)
