from __future__ import annotations

from pathlib import Path

from graphptc.cli import (
    BROWSECOMP_CONFIG,
    BROWSECOMP_PLUS_CONFIG,
    _build_parser,
)


def test_evaluate_is_a_normal_command_without_gate_flags() -> None:
    args = _build_parser().parse_args(["evaluate"])

    assert args.command == "evaluate"
    assert not hasattr(args, "allow_ungated")


def test_browsecomp_commands_use_separate_default_config() -> None:
    parser = _build_parser()

    run_args = parser.parse_args(["run-browsecomp", "--example-id", "7"])
    evaluate_args = parser.parse_args(["evaluate-browsecomp"])

    assert run_args.config == Path(BROWSECOMP_CONFIG)
    assert run_args.example_id == ["7"]
    assert evaluate_args.config == Path(BROWSECOMP_CONFIG)


def test_browsecomp_plus_commands_use_local_benchmark_config() -> None:
    parser = _build_parser()

    download_args = parser.parse_args(["download-browsecomp-plus"])
    run_args = parser.parse_args(["run-browsecomp-plus", "--example-id", "769"])
    evaluate_args = parser.parse_args(["evaluate-browsecomp-plus"])
    stage1_args = parser.parse_args(
        ["run-graphptc-stage1", "--example-id", "769"]
    )

    assert download_args.config == Path(BROWSECOMP_PLUS_CONFIG)
    assert run_args.config == Path(BROWSECOMP_PLUS_CONFIG)
    assert run_args.example_id == ["769"]
    assert evaluate_args.config == Path(BROWSECOMP_PLUS_CONFIG)
    assert stage1_args.config == Path(BROWSECOMP_PLUS_CONFIG)
    assert stage1_args.example_id == ["769"]
    assert stage1_args.events_path is None
    assert stage1_args.shadow_output_path is None
    assert stage1_args.active_repair_output_path is None

    shadow_args = parser.parse_args(
        ["run-graphptc-stage1", "--shadow-output-path", "shadow.jsonl"]
    )
    assert shadow_args.shadow_output_path == Path("shadow.jsonl")

    active_args = parser.parse_args(
        [
            "run-graphptc-stage1",
            "--active-repair-output-path",
            "active.jsonl",
        ]
    )
    assert active_args.active_repair_output_path == Path("active.jsonl")
