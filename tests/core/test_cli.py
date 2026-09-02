from __future__ import annotations

from pathlib import Path

from graphptc.cli import (
    AGENT_DIFF_CONFIG,
    BROWSECOMP_PLUS_CONFIG,
    _build_parser,
)


def test_browsecomp_plus_commands_use_local_benchmark_config() -> None:
    parser = _build_parser()

    download_args = parser.parse_args(["download-browsecomp-plus"])
    inspect_args = parser.parse_args(["inspect-browsecomp-plus"])
    run_args = parser.parse_args(["run-browsecomp-plus", "--example-id", "769"])
    evaluate_args = parser.parse_args(["evaluate-browsecomp-plus"])

    assert download_args.config == Path(BROWSECOMP_PLUS_CONFIG)
    assert inspect_args.config == Path(BROWSECOMP_PLUS_CONFIG)
    assert run_args.config == Path(BROWSECOMP_PLUS_CONFIG)
    assert run_args.example_id == ["769"]
    assert evaluate_args.config == Path(BROWSECOMP_PLUS_CONFIG)


def test_agent_diff_commands_use_separate_default_config() -> None:
    parser = _build_parser()

    download_args = parser.parse_args(["download-agent-diff"])
    run_args = parser.parse_args(
        ["run-agent-diff", "--task-id", "box_145", "--trial", "2"]
    )
    evaluate_args = parser.parse_args(["evaluate-agent-diff"])

    assert download_args.config == Path(AGENT_DIFF_CONFIG)
    assert run_args.config == Path(AGENT_DIFF_CONFIG)
    assert run_args.task_id == ["box_145"]
    assert run_args.trial == [2]
    assert evaluate_args.config == Path(AGENT_DIFF_CONFIG)
