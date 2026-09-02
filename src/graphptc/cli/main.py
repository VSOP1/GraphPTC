from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

from dotenv import load_dotenv

from ..benchmarks.browsecomp_plus.dataset import (
    BrowseCompPlusError,
    download_browsecomp_plus,
)
from ..benchmarks.browsecomp_plus.benchmark import (
    evaluate_browsecomp_plus_benchmark,
    inspect_browsecomp_plus,
    run_browsecomp_plus_benchmark,
)
from ..config import ConfigError, ExperimentConfig
from ..benchmarks.appworld.benchmark import (
    evaluate_appworld_benchmark,
    inspect_appworld,
    run_appworld_benchmark,
)
from ..benchmarks.alfworld.benchmark import (
    evaluate_alfworld_benchmark,
    inspect_alfworld,
    run_alfworld_benchmark,
)
from ..benchmarks.toolsandbox.benchmark import (
    evaluate_toolsandbox_benchmark,
    inspect_toolsandbox,
    run_toolsandbox_benchmark,
)
from ..benchmarks.agent_diff.benchmark import (
    download_agent_diff,
    evaluate_agent_diff_benchmark,
    inspect_agent_diff,
    run_agent_diff_benchmark,
)
from ..benchmarks.apiflow.benchmark import (
    compare_apiflow_benchmarks,
    evaluate_apiflow_benchmark,
    inspect_apiflow,
    run_apiflow_benchmark,
)
from ..benchmarks.toolhop.benchmark import (
    compare_toolhop_benchmarks,
    evaluate_toolhop_benchmark,
    inspect_toolhop,
    run_toolhop_benchmark,
)
from ..benchmarks.fanoutqa.benchmark import (
    compare_fanoutqa_benchmarks,
    evaluate_fanoutqa_benchmark,
    inspect_fanoutqa,
    probe_fanoutqa_wikipedia,
    run_fanoutqa_benchmark,
)
from ..benchmarks.frames.benchmark import (
    compare_frames_benchmarks,
    evaluate_frames_benchmark,
    inspect_frames,
    probe_frames_wikipedia,
    run_frames_benchmark,
)
from ..benchmarks.deepplanning.benchmark import (
    compare_deepplanning_benchmarks,
    compare_deepplanning_configs,
    evaluate_deepplanning_benchmark,
    inspect_deepplanning,
    probe_deepplanning_api,
    run_deepplanning_benchmark,
)
from ..benchmarks.intercode.benchmark import (
    compare_intercode_benchmarks,
    evaluate_intercode_benchmark,
    inspect_intercode,
    run_intercode_benchmark,
)


from .parser import _build_parser

def main(argv: Sequence[str] | None = None) -> int:
    load_dotenv()
    parser = _build_parser()
    args = parser.parse_args(argv)
    try:
        return _run_command(args, parser)
    except (
        BrowseCompPlusError,
        ConfigError,
        FileNotFoundError,
        ValueError,
    ) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


def _run_command(
    args: argparse.Namespace, parser: argparse.ArgumentParser
) -> int:
    config = ExperimentConfig.from_toml(args.config)
    if args.command == "download-browsecomp-plus":
        path = download_browsecomp_plus(
            config.browsecomp_plus,
            config.benchmark.dataset_path,
            config.browsecomp_plus.source_browsecomp_path,
            progress=lambda message: print(message, file=sys.stderr, flush=True),
        )
        print(path)
        return 0

    if args.command == "inspect-browsecomp-plus":
        print(json.dumps(inspect_browsecomp_plus(config), ensure_ascii=False, indent=2))
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


def _print_progress(index: int, total: int, record: dict[str, object]) -> None:
    print(
        f"[{index}/{total}] example={record['example_id']} status={record['status']}",
        file=sys.stderr,
        flush=True,
    )
