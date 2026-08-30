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
from .alfworld_benchmark import (
    evaluate_alfworld_benchmark,
    inspect_alfworld,
    run_alfworld_benchmark,
)
from .deepsearchqa import DeepSearchQAError, download_deepsearchqa
from .toolsandbox_benchmark import (
    evaluate_toolsandbox_benchmark,
    inspect_toolsandbox,
    run_toolsandbox_benchmark,
)
from .agentdiff_benchmark import (
    download_agent_diff,
    evaluate_agent_diff_benchmark,
    inspect_agent_diff,
    run_agent_diff_benchmark,
)
from .tau3_benchmark import (
    evaluate_tau3_benchmark,
    inspect_tau3,
    run_tau3_benchmark,
)
from .mcpmark_benchmark import (
    compare_mcpmark_benchmarks,
    evaluate_mcpmark_benchmark,
    inspect_mcpmark,
    run_mcpmark_benchmark,
)
from .apiflow_benchmark import (
    compare_apiflow_benchmarks,
    evaluate_apiflow_benchmark,
    inspect_apiflow,
    run_apiflow_benchmark,
)
from .toolhop_benchmark import (
    compare_toolhop_benchmarks,
    evaluate_toolhop_benchmark,
    inspect_toolhop,
    run_toolhop_benchmark,
)
from .fanoutqa_benchmark import (
    compare_fanoutqa_benchmarks,
    evaluate_fanoutqa_benchmark,
    inspect_fanoutqa,
    probe_fanoutqa_wikipedia,
    run_fanoutqa_benchmark,
)
from .frames_benchmark import (
    compare_frames_benchmarks,
    evaluate_frames_benchmark,
    inspect_frames,
    probe_frames_wikipedia,
    run_frames_benchmark,
)
from .deepplanning_benchmark import (
    compare_deepplanning_benchmarks,
    compare_deepplanning_configs,
    evaluate_deepplanning_benchmark,
    inspect_deepplanning,
    probe_deepplanning_api,
    run_deepplanning_benchmark,
)
from .intercode_benchmark import (
    compare_intercode_benchmarks,
    evaluate_intercode_benchmark,
    inspect_intercode,
    run_intercode_benchmark,
)


DEFAULT_CONFIG = "configs/deepsearchqa.example.toml"
BROWSECOMP_CONFIG = "configs/browsecomp.example.toml"
BROWSECOMP_PLUS_CONFIG = "configs/browsecomp_plus/browsecomp_plus.example.toml"
APPWORLD_CONFIG = "configs/appworld/appworld.graphptc-dev-smoke.toml"
ALFWORLD_CONFIG = "configs/alfworld/graphptc-smoke.toml"
TOOL_SANDBOX_CONFIG = "configs/toolsandbox/graphptc-smoke.toml"
AGENT_DIFF_CONFIG = "configs/agent_diff/graphptc-smoke.toml"
TAU3_CONFIG = "configs/tau3/graphptc-smoke.toml"
MCPMARK_CONFIG = "configs/mcpmark/graphptc-smoke5.toml"
APIFLOW_CONFIG = "configs/apiflow/graphptc-smoke.toml"
TOOLHOP_CONFIG = "configs/toolhop/graphptc-smoke.toml"
FANOUTQA_CONFIG = "configs/fanoutqa/graphptc-dev.toml"
FRAMES_CONFIG = "configs/frames/graphptc-test.toml"
DEEPPLANNING_CONFIG = "configs/deepplanning/graphptc.toml"
INTERCODE_CONFIG = "configs/intercode/graphptc.toml"


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

    if args.command == "inspect-alfworld":
        inspection = inspect_alfworld(config)
        inspection.pop("task_ids", None)
        print(json.dumps(inspection, ensure_ascii=False, indent=2))
        return 0

    if args.command == "run-alfworld":
        summary = run_alfworld_benchmark(
            config,
            limit=args.limit,
            task_ids=args.task_id,
            restart=args.restart,
        )
        print(json.dumps(summary.to_dict(), ensure_ascii=False, indent=2))
        return 0 if summary.evaluator_failures == 0 and summary.runner_failures == 0 else 1

    if args.command == "evaluate-alfworld":
        print(json.dumps(evaluate_alfworld_benchmark(config), ensure_ascii=False, indent=2))
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

    if args.command == "download-agent-diff":
        print(download_agent_diff(config))
        return 0

    if args.command == "inspect-agent-diff":
        print(json.dumps(inspect_agent_diff(config), ensure_ascii=False, indent=2))
        return 0

    if args.command == "run-agent-diff":
        summary = run_agent_diff_benchmark(
            config,
            limit=args.limit,
            task_ids=args.task_id,
            trials=args.trial,
            restart=args.restart,
            progress=None,
        )
        print(json.dumps(summary.to_dict(), ensure_ascii=False, indent=2))
        return 0 if summary.runner_failures == 0 and summary.evaluator_failures == 0 else 1

    if args.command == "evaluate-agent-diff":
        print(json.dumps(evaluate_agent_diff_benchmark(config), ensure_ascii=False, indent=2))
        return 0

    if args.command == "inspect-tau3":
        inspection = inspect_tau3(config)
        for value in inspection.get("domains", {}).values():
            value.pop("task_ids", None)
        print(json.dumps(inspection, ensure_ascii=False, indent=2))
        return 0

    if args.command == "run-tau3":
        summary = run_tau3_benchmark(
            config,
            limit=args.limit,
            domains=args.domain,
            task_ids=args.task_id,
            trials=args.trial,
            restart=args.restart,
            progress=None,
        )
        print(json.dumps(summary.to_dict(), ensure_ascii=False, indent=2))
        return 0 if summary.runner_failures == 0 and summary.evaluator_failures == 0 else 1

    if args.command == "evaluate-tau3":
        print(json.dumps(evaluate_tau3_benchmark(config), ensure_ascii=False, indent=2))
        return 0

    if args.command == "inspect-mcpmark":
        manifest = inspect_mcpmark(config)
        print(
            json.dumps(
                {
                    "official_commit": manifest["official_commit"],
                    "task_suite": manifest["task_suite"],
                    "expected_tasks": manifest["expected_tasks"],
                    "tasks_sha256": manifest["tasks_sha256"],
                    "task_manifest_path": str(config.mcpmark.task_manifest_path),
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0

    if args.command == "run-mcpmark":
        summary = run_mcpmark_benchmark(
            config,
            limit=args.limit,
            task_ids=args.task_id,
            progress=None,
        )
        print(json.dumps(summary.to_dict(), ensure_ascii=False, indent=2))
        return 0 if not (
            summary.setup_failures
            or summary.execution_failures
            or summary.evaluator_failures
            or summary.cleanup_failures
        ) else 1

    if args.command == "evaluate-mcpmark":
        report = evaluate_mcpmark_benchmark(config)
        print(json.dumps(report["summary"], ensure_ascii=False, indent=2))
        return 0

    if args.command == "compare-mcpmark":
        baseline = ExperimentConfig.from_toml(args.baseline_config)
        report = compare_mcpmark_benchmarks(config, baseline, args.output)
        print(json.dumps(report["overall"], ensure_ascii=False, indent=2))
        return 0

    if args.command == "inspect-apiflow":
        manifest = inspect_apiflow(config)
        print(
            json.dumps(
                {
                    "release": manifest["release"],
                    "bank_sha256": manifest["bank_sha256"],
                    "expected_tasks": manifest["expected_tasks"],
                    "epochs": manifest["epochs"],
                    "environment": manifest["environment"],
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0

    if args.command == "run-apiflow":
        summary = run_apiflow_benchmark(
            config, task_ids=args.task_id, limit=args.limit
        )
        print(json.dumps(summary.to_dict(), ensure_ascii=False, indent=2))
        return 0 if summary.runner_failures == 0 else 1

    if args.command == "evaluate-apiflow":
        report = evaluate_apiflow_benchmark(config)
        print(json.dumps(report["summary"], ensure_ascii=False, indent=2))
        return 0

    if args.command == "compare-apiflow":
        baseline = ExperimentConfig.from_toml(args.baseline_config)
        report = compare_apiflow_benchmarks(config, baseline, args.output)
        print(json.dumps(report["overall"], ensure_ascii=False, indent=2))
        return 0

    if args.command == "inspect-toolhop":
        manifest = inspect_toolhop(config)
        print(
            json.dumps(
                {
                    "scenario": manifest["scenario"],
                    "official_commit": manifest["official_commit"],
                    "data_sha256": manifest["data_sha256"],
                    "expected_tasks": manifest["expected_tasks"],
                    "environment": manifest["environment"],
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0

    if args.command == "run-toolhop":
        summary = run_toolhop_benchmark(
            config, task_ids=args.task_id, limit=args.limit
        )
        print(json.dumps(summary.to_dict(), ensure_ascii=False, indent=2))
        return 0 if summary.runner_failures == 0 else 1

    if args.command == "evaluate-toolhop":
        report = evaluate_toolhop_benchmark(config)
        print(json.dumps(report["summary"], ensure_ascii=False, indent=2))
        return 0

    if args.command == "compare-toolhop":
        baseline = ExperimentConfig.from_toml(args.baseline_config)
        report = compare_toolhop_benchmarks(config, baseline, args.output)
        print(json.dumps(report["official"], ensure_ascii=False, indent=2))
        return 0

    if args.command == "inspect-fanoutqa":
        print(json.dumps(inspect_fanoutqa(config), ensure_ascii=False, indent=2))
        return 0

    if args.command == "probe-fanoutqa-wikipedia":
        print(json.dumps(probe_fanoutqa_wikipedia(config), ensure_ascii=False, indent=2))
        return 0

    if args.command == "run-fanoutqa":
        summary = run_fanoutqa_benchmark(
            config,
            limit=args.limit,
            task_ids=args.task_id,
            restart=args.restart,
        )
        print(json.dumps(summary.to_dict(), ensure_ascii=False, indent=2))
        return 0 if summary.failed == 0 else 1

    if args.command == "evaluate-fanoutqa":
        report = evaluate_fanoutqa_benchmark(config)
        print(json.dumps(report["scoring"], ensure_ascii=False, indent=2))
        return 0

    if args.command == "compare-fanoutqa":
        baseline = ExperimentConfig.from_toml(args.baseline_config)
        report = compare_fanoutqa_benchmarks(config, baseline, args.output)
        print(json.dumps(report["difference"], ensure_ascii=False, indent=2))
        return 0

    if args.command == "inspect-frames":
        print(json.dumps(inspect_frames(config), ensure_ascii=False, indent=2))
        return 0

    if args.command == "probe-frames-wikipedia":
        print(json.dumps(probe_frames_wikipedia(config), ensure_ascii=False, indent=2))
        return 0

    if args.command == "run-frames":
        summary = run_frames_benchmark(
            config,
            limit=args.limit,
            task_ids=args.task_id,
            restart=args.restart,
        )
        print(json.dumps(summary.to_dict(), ensure_ascii=False, indent=2))
        return 0 if summary.failed == 0 else 1

    if args.command == "evaluate-frames":
        report = evaluate_frames_benchmark(config)
        print(json.dumps(report["scoring"], ensure_ascii=False, indent=2))
        return 0

    if args.command == "compare-frames":
        baseline = ExperimentConfig.from_toml(args.baseline_config)
        report = compare_frames_benchmarks(config, baseline, args.output)
        print(json.dumps(report["difference"], ensure_ascii=False, indent=2))
        return 0

    if args.command == "inspect-intercode":
        inspection = inspect_intercode(config)
        inspection.pop("tasks", None)
        print(json.dumps(inspection, ensure_ascii=False, indent=2))
        return 0

    if args.command == "run-intercode":
        summary = run_intercode_benchmark(
            config,
            limit=args.limit,
            task_ids=args.task_id,
            restart=args.restart,
        )
        print(json.dumps(summary.to_dict(), ensure_ascii=False, indent=2))
        return 0 if summary.runner_failures == 0 else 1

    if args.command == "evaluate-intercode":
        report = evaluate_intercode_benchmark(config)
        print(json.dumps(report["scoring"], ensure_ascii=False, indent=2))
        return 0

    if args.command == "compare-intercode":
        baseline = ExperimentConfig.from_toml(args.baseline_config)
        report = compare_intercode_benchmarks(config, baseline, args.output)
        print(json.dumps(report["difference"], ensure_ascii=False, indent=2))
        return 0

    if args.command == "inspect-deepplanning":
        print(json.dumps(inspect_deepplanning(config), ensure_ascii=False, indent=2))
        return 0

    if args.command == "probe-deepplanning-api":
        report = probe_deepplanning_api(
            config,
            concurrencies=args.concurrency or (10, 20, 40),
            waves=args.waves,
            output=args.output,
        )
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 0 if report["highest_stable_total_concurrency"] is not None else 1

    if args.command == "run-deepplanning":
        summary = run_deepplanning_benchmark(
            config,
            task_keys=args.task_key,
            domains=args.domain,
            run_index=args.run_index,
            run_label=args.run_label,
            limit=args.limit,
            restart=args.restart,
            progress=None,
        )
        print(json.dumps(summary.to_dict(), ensure_ascii=False, indent=2))
        return 0 if summary.runner_failures == 0 else 1

    if args.command == "compare-deepplanning-configs":
        baseline = ExperimentConfig.from_toml(args.baseline_config)
        print(json.dumps(compare_deepplanning_configs(config, baseline), ensure_ascii=False, indent=2))
        return 0

    if args.command == "evaluate-deepplanning":
        report = evaluate_deepplanning_benchmark(config, run_index=args.run_index, run_label=args.run_label)
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 0 if not any(report["failures"].values()) else 1

    if args.command == "compare-deepplanning":
        baseline = ExperimentConfig.from_toml(args.baseline_config)
        report = compare_deepplanning_benchmarks(
            config, baseline, run_label=args.run_label, run_index=args.run_index, output=args.output
        )
        print(json.dumps({"paired_score": report["paired_score"], "graph_control": report["graph_control"], "operational_deltas": report["operational_deltas"]}, ensure_ascii=False, indent=2))
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

    alfworld_inspect = subparsers.add_parser(
        "inspect-alfworld",
        help="Audit the isolated official ALFWorld text environment and split.",
    )
    _add_config_argument(alfworld_inspect, default=ALFWORLD_CONFIG)

    alfworld_run = subparsers.add_parser(
        "run-alfworld",
        help="Run matched GraphPTC or Fewshot PTC on official ALFWorld text episodes.",
    )
    _add_config_argument(alfworld_run, default=ALFWORLD_CONFIG)
    alfworld_run.add_argument("--limit", type=int)
    alfworld_run.add_argument("--task-id", action="append", default=[])
    alfworld_run.add_argument("--restart", action="store_true")

    alfworld_evaluate = subparsers.add_parser(
        "evaluate-alfworld", help="Validate and aggregate saved official ALFWorld metrics."
    )
    _add_config_argument(alfworld_evaluate, default=ALFWORLD_CONFIG)

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

    agent_diff_download = subparsers.add_parser(
        "download-agent-diff", help="Download and verify the frozen official Agent-Diff dataset."
    )
    _add_config_argument(agent_diff_download, default=AGENT_DIFF_CONFIG)

    agent_diff_inspect = subparsers.add_parser(
        "inspect-agent-diff", help="Inspect the isolated official Agent-Diff SDK and dataset."
    )
    _add_config_argument(agent_diff_inspect, default=AGENT_DIFF_CONFIG)

    agent_diff_run = subparsers.add_parser(
        "run-agent-diff", help="Run GraphPTC or Fewshot PTC on Agent-Diff."
    )
    _add_config_argument(agent_diff_run, default=AGENT_DIFF_CONFIG)
    agent_diff_run.add_argument("--limit", type=int)
    agent_diff_run.add_argument("--task-id", action="append", default=[])
    agent_diff_run.add_argument("--trial", action="append", type=int, default=[])
    agent_diff_run.add_argument("--restart", action="store_true")

    agent_diff_evaluate = subparsers.add_parser(
        "evaluate-agent-diff", help="Aggregate official Agent-Diff state-diff results."
    )
    _add_config_argument(agent_diff_evaluate, default=AGENT_DIFF_CONFIG)

    tau3_inspect = subparsers.add_parser(
        "inspect-tau3", help="Audit the isolated official tau3-bench text environment."
    )
    _add_config_argument(tau3_inspect, default=TAU3_CONFIG)

    tau3_run = subparsers.add_parser(
        "run-tau3", help="Run GraphPTC or Fewshot PTC on official tau3-bench text domains."
    )
    _add_config_argument(tau3_run, default=TAU3_CONFIG)
    tau3_run.add_argument("--limit", type=int)
    tau3_run.add_argument("--domain", action="append", default=[])
    tau3_run.add_argument("--task-id", action="append", default=[])
    tau3_run.add_argument("--trial", action="append", type=int, default=[])
    tau3_run.add_argument("--restart", action="store_true")

    tau3_evaluate = subparsers.add_parser(
        "evaluate-tau3", help="Aggregate saved official tau3-bench rewards."
    )
    _add_config_argument(tau3_evaluate, default=TAU3_CONFIG)

    mcpmark_inspect = subparsers.add_parser(
        "inspect-mcpmark",
        help="Audit the frozen MCPMark Verified checkout and write its task manifest.",
    )
    _add_config_argument(mcpmark_inspect, default=MCPMARK_CONFIG)

    mcpmark_run = subparsers.add_parser(
        "run-mcpmark",
        help="Run GraphPTC or Fewshot PTC through official MCPMark lifecycle and verifiers.",
    )
    _add_config_argument(mcpmark_run, default=MCPMARK_CONFIG)
    mcpmark_run.add_argument("--limit", type=int)
    mcpmark_run.add_argument(
        "--task-id",
        action="append",
        default=[],
        help="Exact service:category/task ID; repeat for multiple tasks.",
    )

    mcpmark_evaluate = subparsers.add_parser(
        "evaluate-mcpmark",
        help="Validate and summarize saved official MCPMark verifier results.",
    )
    _add_config_argument(mcpmark_evaluate, default=MCPMARK_CONFIG)

    mcpmark_compare = subparsers.add_parser(
        "compare-mcpmark",
        help="Validate and compare paired GraphPTC and Fewshot PTC MCPMark reports.",
    )
    _add_config_argument(mcpmark_compare, default="configs/mcpmark/graphptc.toml")
    mcpmark_compare.add_argument(
        "--baseline-config",
        type=Path,
        default=Path("configs/mcpmark/fewshot-ptc.toml"),
    )
    mcpmark_compare.add_argument(
        "--output",
        type=Path,
        default=Path("runs/mcpmark/paired-report.json"),
    )

    apiflow_inspect = subparsers.add_parser(
        "inspect-apiflow", help="Audit the frozen APIFlow-Bench 1.0 task bank."
    )
    _add_config_argument(apiflow_inspect, default=APIFLOW_CONFIG)

    apiflow_run = subparsers.add_parser(
        "run-apiflow", help="Run GraphPTC or Fewshot PTC on APIFlow-Bench 1.0."
    )
    _add_config_argument(apiflow_run, default=APIFLOW_CONFIG)
    apiflow_run.add_argument("--limit", type=int)
    apiflow_run.add_argument("--task-id", action="append", default=[])

    apiflow_evaluate = subparsers.add_parser(
        "evaluate-apiflow", help="Validate and summarize APIFlow-Bench results."
    )
    _add_config_argument(apiflow_evaluate, default=APIFLOW_CONFIG)

    apiflow_compare = subparsers.add_parser(
        "compare-apiflow", help="Compare paired GraphPTC and Fewshot PTC APIFlow results."
    )
    _add_config_argument(apiflow_compare, default="configs/apiflow/graphptc.toml")
    apiflow_compare.add_argument(
        "--baseline-config",
        type=Path,
        default=Path("configs/apiflow/fewshot-ptc.toml"),
    )
    apiflow_compare.add_argument(
        "--output",
        type=Path,
        default=Path("runs/apiflow/paired-report.json"),
    )

    toolhop_inspect = subparsers.add_parser(
        "inspect-toolhop", help="Audit the frozen official ToolHop task bank."
    )
    _add_config_argument(toolhop_inspect, default=TOOLHOP_CONFIG)

    toolhop_run = subparsers.add_parser(
        "run-toolhop", help="Run GraphPTC or Fewshot PTC on ToolHop Mandatory."
    )
    _add_config_argument(toolhop_run, default=TOOLHOP_CONFIG)
    toolhop_run.add_argument("--limit", type=int)
    toolhop_run.add_argument("--task-id", action="append", default=[])

    toolhop_evaluate = subparsers.add_parser(
        "evaluate-toolhop", help="Validate and summarize ToolHop results."
    )
    _add_config_argument(toolhop_evaluate, default=TOOLHOP_CONFIG)

    toolhop_compare = subparsers.add_parser(
        "compare-toolhop", help="Compare paired GraphPTC and Fewshot PTC ToolHop reports."
    )
    _add_config_argument(toolhop_compare, default="configs/toolhop/graphptc.toml")
    toolhop_compare.add_argument(
        "--baseline-config",
        type=Path,
        default=Path("configs/toolhop/fewshot-ptc.toml"),
    )
    toolhop_compare.add_argument(
        "--output",
        type=Path,
        default=Path("runs/toolhop/mandatory-temperature0-epoch1/paired-report.json"),
    )

    fanoutqa_inspect = subparsers.add_parser(
        "inspect-fanoutqa", help="Inspect the official FanOutQA split and adapter configuration."
    )
    _add_config_argument(fanoutqa_inspect, default=FANOUTQA_CONFIG)

    fanoutqa_probe = subparsers.add_parser(
        "probe-fanoutqa-wikipedia",
        help="Verify wiki_search and wiki_content against the local official snapshot.",
    )
    _add_config_argument(fanoutqa_probe, default=FANOUTQA_CONFIG)

    fanoutqa_run = subparsers.add_parser(
        "run-fanoutqa", help="Run GraphPTC or Fewshot PTC on FanOutQA open-book."
    )
    _add_config_argument(fanoutqa_run, default=FANOUTQA_CONFIG)
    fanoutqa_run.add_argument("--limit", type=int)
    fanoutqa_run.add_argument("--task-id", action="append", default=[])
    fanoutqa_run.add_argument("--restart", action="store_true")

    fanoutqa_evaluate = subparsers.add_parser(
        "evaluate-fanoutqa", help="Score FanOutQA dev outputs with official metrics and MiMo judge."
    )
    _add_config_argument(fanoutqa_evaluate, default=FANOUTQA_CONFIG)

    fanoutqa_compare = subparsers.add_parser(
        "compare-fanoutqa", help="Create the matched FanOutQA paired result report."
    )
    _add_config_argument(fanoutqa_compare, default=FANOUTQA_CONFIG)
    fanoutqa_compare.add_argument(
        "--baseline-config",
        type=Path,
        default=Path("configs/fanoutqa/fewshot-ptc-dev.toml"),
    )
    fanoutqa_compare.add_argument(
        "--output", type=Path, default=Path("runs/fanoutqa/dev/paired-report.json")
    )

    frames_inspect = subparsers.add_parser(
        "inspect-frames", help="Inspect the official FRAMES test set and retriever configuration."
    )
    _add_config_argument(frames_inspect, default=FRAMES_CONFIG)

    frames_probe = subparsers.add_parser(
        "probe-frames-wikipedia",
        help="Verify FRAMES BM25 search and article fetch against the official snapshot.",
    )
    _add_config_argument(frames_probe, default=FRAMES_CONFIG)

    frames_run = subparsers.add_parser(
        "run-frames", help="Run GraphPTC or the matched PTC baseline on FRAMES test."
    )
    _add_config_argument(frames_run, default=FRAMES_CONFIG)
    frames_run.add_argument("--limit", type=int)
    frames_run.add_argument("--task-id", action="append", default=[])
    frames_run.add_argument("--restart", action="store_true")

    frames_evaluate = subparsers.add_parser(
        "evaluate-frames", help="Score complete FRAMES outputs with the official MiMo judge prompt."
    )
    _add_config_argument(frames_evaluate, default=FRAMES_CONFIG)

    frames_compare = subparsers.add_parser(
        "compare-frames", help="Create the matched FRAMES paired result report."
    )
    _add_config_argument(frames_compare, default=FRAMES_CONFIG)
    frames_compare.add_argument(
        "--baseline-config",
        type=Path,
        default=Path("configs/frames/fewshot-ptc-test.toml"),
    )
    frames_compare.add_argument(
        "--output", type=Path, default=Path("runs/frames/test/paired-report.json")
    )

    intercode_inspect = subparsers.add_parser(
        "inspect-intercode", help="Inspect the pinned official InterCode Bash and SQL environments."
    )
    _add_config_argument(intercode_inspect, default=INTERCODE_CONFIG)

    intercode_run = subparsers.add_parser(
        "run-intercode", help="Run GraphPTC or the matched PTC baseline on official InterCode."
    )
    _add_config_argument(intercode_run, default=INTERCODE_CONFIG)
    intercode_run.add_argument("--limit", type=int)
    intercode_run.add_argument("--task-id", action="append", default=[])
    intercode_run.add_argument("--restart", action="store_true")

    intercode_evaluate = subparsers.add_parser(
        "evaluate-intercode", help="Aggregate official InterCode success and action metrics."
    )
    _add_config_argument(intercode_evaluate, default=INTERCODE_CONFIG)

    intercode_compare = subparsers.add_parser(
        "compare-intercode", help="Create the matched InterCode paired result report."
    )
    _add_config_argument(intercode_compare, default=INTERCODE_CONFIG)
    intercode_compare.add_argument(
        "--baseline-config",
        type=Path,
        default=Path("configs/intercode/baseline.toml"),
    )
    intercode_compare.add_argument(
        "--output", type=Path, default=Path("runs/intercode/paired-report.json")
    )

    deepplanning_inspect = subparsers.add_parser(
        "inspect-deepplanning", help="Audit the pinned official DeepPlanning v1.1 installation."
    )
    _add_config_argument(deepplanning_inspect, default=DEEPPLANNING_CONFIG)

    deepplanning_probe = subparsers.add_parser(
        "probe-deepplanning-api", help="Probe raw model API stability without DeepPlanning tasks or retries."
    )
    _add_config_argument(deepplanning_probe, default=DEEPPLANNING_CONFIG)
    deepplanning_probe.add_argument("--concurrency", action="append", type=int, default=[])
    deepplanning_probe.add_argument("--waves", type=int, default=2)
    deepplanning_probe.add_argument("--output", type=Path)

    deepplanning_run = subparsers.add_parser(
        "run-deepplanning", help="Run GraphPTC or Fewshot PTC on official DeepPlanning tools."
    )
    _add_config_argument(deepplanning_run, default=DEEPPLANNING_CONFIG)
    deepplanning_run.add_argument("--task-key", action="append", default=[])
    deepplanning_run.add_argument("--domain", action="append", default=[])
    deepplanning_run.add_argument("--run-index", type=int, default=0)
    deepplanning_run.add_argument("--run-label", default="full")
    deepplanning_run.add_argument("--limit", type=int)
    deepplanning_run.add_argument("--restart", action="store_true")

    deepplanning_compare = subparsers.add_parser(
        "compare-deepplanning-configs", help="Verify the matched DeepPlanning arm configs."
    )
    _add_config_argument(deepplanning_compare, default=DEEPPLANNING_CONFIG)
    deepplanning_compare.add_argument(
        "--baseline-config", type=Path, default=Path("configs/deepplanning/fewshot-ptc.toml")
    )

    deepplanning_evaluate = subparsers.add_parser(
        "evaluate-deepplanning", help="Run official DeepPlanning conversion, evaluators, and aggregation."
    )
    _add_config_argument(deepplanning_evaluate, default=DEEPPLANNING_CONFIG)
    deepplanning_evaluate.add_argument("--run-index", type=int, default=0)
    deepplanning_evaluate.add_argument("--run-label", default="full")

    deepplanning_result_compare = subparsers.add_parser(
        "compare-deepplanning", help="Create a matched paired DeepPlanning result report."
    )
    _add_config_argument(deepplanning_result_compare, default=DEEPPLANNING_CONFIG)
    deepplanning_result_compare.add_argument("--baseline-config", type=Path, default=Path("configs/deepplanning/fewshot-ptc.toml"))
    deepplanning_result_compare.add_argument("--run-label", default="full")
    deepplanning_result_compare.add_argument("--run-index", type=int, default=0)
    deepplanning_result_compare.add_argument("--output", type=Path)

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
