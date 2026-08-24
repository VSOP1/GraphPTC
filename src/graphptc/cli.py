from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

from dotenv import load_dotenv

from .benchmark import evaluate_benchmark, run_benchmark
from .browsecomp import BrowseCompError, download_browsecomp
from .browsecomp_benchmark import (
    evaluate_browsecomp_benchmark,
    run_browsecomp_benchmark,
)
from .browsecomp_plus import BrowseCompPlusError, download_browsecomp_plus
from .browsecomp_plus_benchmark import (
    evaluate_browsecomp_plus_benchmark,
    run_browsecomp_plus_benchmark,
)
from .config import ConfigError, ExperimentConfig
from .appworld_benchmark import (
    evaluate_appworld_benchmark,
    inspect_appworld,
    run_appworld_benchmark,
)
from .deepsearchqa import DeepSearchQAError, download_deepsearchqa
from .toolsandbox_benchmark import (
    evaluate_toolsandbox_benchmark,
    inspect_toolsandbox,
    run_toolsandbox_benchmark,
)


DEFAULT_CONFIG = "configs/deepsearchqa.example.toml"
BROWSECOMP_CONFIG = "configs/browsecomp.example.toml"
BROWSECOMP_PLUS_CONFIG = "configs/browsecomp_plus/browsecomp_plus.example.toml"
APPWORLD_CONFIG = "configs/appworld/appworld.graphptc-dev-smoke.toml"
TOOL_SANDBOX_CONFIG = "configs/toolsandbox/graphptc-smoke.toml"


def main(argv: Sequence[str] | None = None) -> int:
    load_dotenv()
    parser = _build_parser()
    args = parser.parse_args(argv)
    try:
        return _run_command(args, parser)
    except (
        BrowseCompError,
        BrowseCompPlusError,
        ConfigError,
        DeepSearchQAError,
        FileNotFoundError,
        ValueError,
    ) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


def _run_command(
    args: argparse.Namespace, parser: argparse.ArgumentParser
) -> int:
    config = ExperimentConfig.from_toml(args.config)
    if args.command == "download-data":
        path = download_deepsearchqa(config.benchmark.dataset_path)
        print(path)
        return 0

    if args.command == "run":
        summary = run_benchmark(
            config,
            limit=args.limit,
            example_ids=args.example_id,
            resume=not args.restart,
            progress=_print_progress,
        )
        print(json.dumps(summary.to_dict(), ensure_ascii=False, indent=2))
        return 0 if summary.failed == 0 else 1

    if args.command == "evaluate":
        result = evaluate_benchmark(config)
        print(json.dumps(result.summary.to_dict(), ensure_ascii=False, indent=2))
        return 0

    if args.command == "download-browsecomp":
        path = download_browsecomp(config.benchmark.dataset_path)
        print(path)
        return 0

    if args.command == "run-browsecomp":
        summary = run_browsecomp_benchmark(
            config,
            limit=args.limit,
            example_ids=args.example_id,
            resume=not args.restart,
            progress=_print_progress,
        )
        print(json.dumps(summary.to_dict(), ensure_ascii=False, indent=2))
        return 0 if summary.failed == 0 else 1

    if args.command == "evaluate-browsecomp":
        result = evaluate_browsecomp_benchmark(config)
        print(json.dumps(result.summary.to_dict(), ensure_ascii=False, indent=2))
        return 0

    if args.command == "download-browsecomp-plus":
        path = download_browsecomp_plus(
            config.browsecomp_plus,
            config.benchmark.dataset_path,
            config.browsecomp_plus.source_browsecomp_path,
            progress=lambda message: print(message, file=sys.stderr, flush=True),
        )
        print(path)
        return 0

    if args.command == "run-browsecomp-plus":
        summary = run_browsecomp_plus_benchmark(
            config,
            limit=args.limit,
            example_ids=args.example_id,
            resume=not args.restart,
            progress=_print_progress,
        )
        print(json.dumps(summary.to_dict(), ensure_ascii=False, indent=2))
        return 0 if summary.failed == 0 else 1

    if args.command == "evaluate-browsecomp-plus":
        result = evaluate_browsecomp_plus_benchmark(config)
        print(json.dumps(result.summary.to_dict(), ensure_ascii=False, indent=2))
        return 0

    if args.command == "inspect-appworld":
        print(json.dumps(inspect_appworld(config), ensure_ascii=False, indent=2))
        return 0

    if args.command == "run-appworld":
        summary = run_appworld_benchmark(
            config,
            limit=args.limit,
            task_ids=args.task_id,
            restart=args.restart,
        )
        print(json.dumps(summary.to_dict(), ensure_ascii=False, indent=2))
        return 0 if summary.evaluator_failures == 0 and summary.runner_failures == 0 else 1

    if args.command == "evaluate-appworld":
        print(json.dumps(evaluate_appworld_benchmark(config), ensure_ascii=False, indent=2))
        return 0

    if args.command == "inspect-toolsandbox":
        inspection = inspect_toolsandbox(config)
        inspection.pop("scenario_names", None)
        print(json.dumps(inspection, ensure_ascii=False, indent=2))
        return 0

    if args.command == "run-toolsandbox":
        summary = run_toolsandbox_benchmark(
            config,
            limit=args.limit,
            scenario_names=args.scenario_name,
            restart=args.restart,
            progress=None,
        )
        print(json.dumps(summary.to_dict(), ensure_ascii=False, indent=2))
        return 0 if summary.runner_failures == 0 else 1

    if args.command == "evaluate-toolsandbox":
        print(json.dumps(evaluate_toolsandbox_benchmark(config), ensure_ascii=False, indent=2))
        return 0

    parser.error(f"Unknown command: {args.command}")
    return 2


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="graphptc",
        description="Original PTC baseline runner for agentic-search benchmarks.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    download = subparsers.add_parser(
        "download-data",
        help="Download and verify the official 900-example dataset.",
    )
    _add_config_argument(download)

    run = subparsers.add_parser(
        "run",
        help="Run MiMo with autonomous PTC blocks and save JSONL records.",
    )
    _add_config_argument(run)
    run.add_argument("--limit", type=int, help="Run only the first N selected examples.")
    run.add_argument(
        "--example-id",
        action="append",
        default=[],
        help="Run a specific example ID; repeat for multiple IDs.",
    )
    run.add_argument(
        "--restart",
        action="store_true",
        help="Replace the response file instead of resuming completed IDs.",
    )

    evaluate = subparsers.add_parser(
        "evaluate",
        help="Grade saved predictions with the configured DeepSearchQA judge.",
    )
    _add_config_argument(evaluate)

    browsecomp_download = subparsers.add_parser(
        "download-browsecomp",
        help="Download and verify the official encrypted BrowseComp dataset.",
    )
    _add_config_argument(browsecomp_download, default=BROWSECOMP_CONFIG)

    browsecomp_run = subparsers.add_parser(
        "run-browsecomp",
        help="Run MiMo with autonomous PTC blocks on BrowseComp.",
    )
    _add_config_argument(browsecomp_run, default=BROWSECOMP_CONFIG)
    browsecomp_run.add_argument(
        "--limit", type=int, help="Run only the first N selected examples."
    )
    browsecomp_run.add_argument(
        "--example-id",
        action="append",
        default=[],
        help="Run a specific BrowseComp example ID; repeat for multiple IDs.",
    )
    browsecomp_run.add_argument(
        "--restart",
        action="store_true",
        help="Replace the BrowseComp response file instead of resuming.",
    )

    browsecomp_evaluate = subparsers.add_parser(
        "evaluate-browsecomp",
        help="Grade BrowseComp predictions with the configured A/B/C judge.",
    )
    _add_config_argument(browsecomp_evaluate, default=BROWSECOMP_CONFIG)

    browsecomp_plus_download = subparsers.add_parser(
        "download-browsecomp-plus",
        help="Download the frozen BrowseComp-Plus corpus and build its local index.",
    )
    _add_config_argument(browsecomp_plus_download, default=BROWSECOMP_PLUS_CONFIG)

    browsecomp_plus_run = subparsers.add_parser(
        "run-browsecomp-plus",
        help="Run autonomous PTC against the local BrowseComp-Plus corpus.",
    )
    _add_config_argument(browsecomp_plus_run, default=BROWSECOMP_PLUS_CONFIG)
    browsecomp_plus_run.add_argument(
        "--limit", type=int, help="Run only the first N selected examples."
    )
    browsecomp_plus_run.add_argument(
        "--example-id",
        action="append",
        default=[],
        help="Run a specific BrowseComp-Plus query ID; repeat for multiple IDs.",
    )
    browsecomp_plus_run.add_argument(
        "--restart",
        action="store_true",
        help="Replace the BrowseComp-Plus response file instead of resuming.",
    )

    browsecomp_plus_evaluate = subparsers.add_parser(
        "evaluate-browsecomp-plus",
        help="Grade BrowseComp-Plus predictions with the configured development judge.",
    )
    _add_config_argument(browsecomp_plus_evaluate, default=BROWSECOMP_PLUS_CONFIG)

    appworld_inspect = subparsers.add_parser(
        "inspect-appworld", help="Inspect the isolated official AppWorld installation and dev split."
    )
    _add_config_argument(appworld_inspect, default=APPWORLD_CONFIG)

    appworld_run = subparsers.add_parser(
        "run-appworld", help="Run GraphPTC in isolated AppWorld task worlds."
    )
    _add_config_argument(appworld_run, default=APPWORLD_CONFIG)
    appworld_run.add_argument("--limit", type=int)
    appworld_run.add_argument("--task-id", action="append", default=[])
    appworld_run.add_argument("--restart", action="store_true")

    appworld_evaluate = subparsers.add_parser(
        "evaluate-appworld", help="Run the official AppWorld evaluator over saved task worlds."
    )
    _add_config_argument(appworld_evaluate, default=APPWORLD_CONFIG)

    toolsandbox_inspect = subparsers.add_parser(
        "inspect-toolsandbox", help="Inspect the isolated official ToolSandbox installation."
    )
    _add_config_argument(toolsandbox_inspect, default=TOOL_SANDBOX_CONFIG)

    toolsandbox_run = subparsers.add_parser(
        "run-toolsandbox", help="Run GraphPTC or Fewshot PTC on official ToolSandbox scenarios."
    )
    _add_config_argument(toolsandbox_run, default=TOOL_SANDBOX_CONFIG)
    toolsandbox_run.add_argument("--limit", type=int)
    toolsandbox_run.add_argument("--scenario-name", action="append", default=[])
    toolsandbox_run.add_argument("--restart", action="store_true")

    toolsandbox_evaluate = subparsers.add_parser(
        "evaluate-toolsandbox", help="Aggregate saved official ToolSandbox evaluation results."
    )
    _add_config_argument(toolsandbox_evaluate, default=TOOL_SANDBOX_CONFIG)

    return parser


def _add_config_argument(
    parser: argparse.ArgumentParser, *, default: str = DEFAULT_CONFIG
) -> None:
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(default),
        help=f"Experiment TOML path (default: {default}).",
    )


def _print_progress(index: int, total: int, record: dict[str, object]) -> None:
    print(
        f"[{index}/{total}] example={record['example_id']} status={record['status']}",
        file=sys.stderr,
        flush=True,
    )
