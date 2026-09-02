from __future__ import annotations

import argparse
from pathlib import Path

BROWSECOMP_PLUS_CONFIG = "configs/browsecomp_plus/browsecomp_plus.graphptc-full.toml"
APPWORLD_CONFIG = "configs/appworld/appworld.graphptc-test-normal.toml"
ALFWORLD_CONFIG = "configs/alfworld/graphptc-valid-seen.toml"
TOOL_SANDBOX_CONFIG = "configs/toolsandbox/graphptc.toml"
AGENT_DIFF_CONFIG = "configs/agent_diff/graphptc.toml"
APIFLOW_CONFIG = "configs/apiflow/graphptc.toml"
TOOLHOP_CONFIG = "configs/toolhop/graphptc.toml"
FANOUTQA_CONFIG = "configs/fanoutqa/graphptc-dev.toml"
FRAMES_CONFIG = "configs/frames/graphptc-test.toml"
DEEPPLANNING_CONFIG = "configs/deepplanning/graphptc.toml"
INTERCODE_CONFIG = "configs/intercode/graphptc.toml"


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="graphptc",
        description="GraphPTC and matched-baseline runner for agentic benchmarks.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    browsecomp_plus_download = subparsers.add_parser(
        "download-browsecomp-plus",
        help="Download the frozen BrowseComp-Plus corpus and build its local index.",
    )
    _add_config_argument(browsecomp_plus_download, default=BROWSECOMP_PLUS_CONFIG)

    browsecomp_plus_inspect = subparsers.add_parser(
        "inspect-browsecomp-plus",
        help="Inspect the complete 830-question dataset and retriever metadata.",
    )
    _add_config_argument(browsecomp_plus_inspect, default=BROWSECOMP_PLUS_CONFIG)

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
        "inspect-appworld",
        help="Inspect the isolated official AppWorld installation and dev split.",
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
        "evaluate-appworld",
        help="Run the official AppWorld evaluator over saved task worlds.",
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
        "evaluate-alfworld",
        help="Validate and aggregate saved official ALFWorld metrics.",
    )
    _add_config_argument(alfworld_evaluate, default=ALFWORLD_CONFIG)

    toolsandbox_inspect = subparsers.add_parser(
        "inspect-toolsandbox",
        help="Inspect the isolated official ToolSandbox installation.",
    )
    _add_config_argument(toolsandbox_inspect, default=TOOL_SANDBOX_CONFIG)

    toolsandbox_run = subparsers.add_parser(
        "run-toolsandbox",
        help="Run GraphPTC or Fewshot PTC on official ToolSandbox scenarios.",
    )
    _add_config_argument(toolsandbox_run, default=TOOL_SANDBOX_CONFIG)
    toolsandbox_run.add_argument("--limit", type=int)
    toolsandbox_run.add_argument("--scenario-name", action="append", default=[])
    toolsandbox_run.add_argument("--restart", action="store_true")

    toolsandbox_evaluate = subparsers.add_parser(
        "evaluate-toolsandbox",
        help="Aggregate saved official ToolSandbox evaluation results.",
    )
    _add_config_argument(toolsandbox_evaluate, default=TOOL_SANDBOX_CONFIG)

    agent_diff_download = subparsers.add_parser(
        "download-agent-diff",
        help="Download and verify the frozen official Agent-Diff dataset.",
    )
    _add_config_argument(agent_diff_download, default=AGENT_DIFF_CONFIG)

    agent_diff_inspect = subparsers.add_parser(
        "inspect-agent-diff",
        help="Inspect the isolated official Agent-Diff SDK and dataset.",
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
        "compare-apiflow",
        help="Compare paired GraphPTC and Fewshot PTC APIFlow results.",
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
        "compare-toolhop",
        help="Compare paired GraphPTC and Fewshot PTC ToolHop reports.",
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
        "inspect-fanoutqa",
        help="Inspect the official FanOutQA split and adapter configuration.",
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
        "evaluate-fanoutqa",
        help="Score FanOutQA dev outputs with official metrics and MiMo judge.",
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
        "inspect-frames",
        help="Inspect the official FRAMES test set and retriever configuration.",
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
        "evaluate-frames",
        help="Score complete FRAMES outputs with the official MiMo judge prompt.",
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
        "inspect-intercode",
        help="Inspect the pinned official InterCode Bash and SQL environments.",
    )
    _add_config_argument(intercode_inspect, default=INTERCODE_CONFIG)

    intercode_run = subparsers.add_parser(
        "run-intercode",
        help="Run GraphPTC or the matched PTC baseline on official InterCode.",
    )
    _add_config_argument(intercode_run, default=INTERCODE_CONFIG)
    intercode_run.add_argument("--limit", type=int)
    intercode_run.add_argument("--task-id", action="append", default=[])
    intercode_run.add_argument("--restart", action="store_true")

    intercode_evaluate = subparsers.add_parser(
        "evaluate-intercode",
        help="Aggregate official InterCode success and action metrics.",
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
        "inspect-deepplanning",
        help="Audit the pinned official DeepPlanning v1.1 installation.",
    )
    _add_config_argument(deepplanning_inspect, default=DEEPPLANNING_CONFIG)

    deepplanning_probe = subparsers.add_parser(
        "probe-deepplanning-api",
        help="Probe raw model API stability without DeepPlanning tasks or retries.",
    )
    _add_config_argument(deepplanning_probe, default=DEEPPLANNING_CONFIG)
    deepplanning_probe.add_argument(
        "--concurrency", action="append", type=int, default=[]
    )
    deepplanning_probe.add_argument("--waves", type=int, default=2)
    deepplanning_probe.add_argument("--output", type=Path)

    deepplanning_run = subparsers.add_parser(
        "run-deepplanning",
        help="Run GraphPTC or Fewshot PTC on official DeepPlanning tools.",
    )
    _add_config_argument(deepplanning_run, default=DEEPPLANNING_CONFIG)
    deepplanning_run.add_argument("--task-key", action="append", default=[])
    deepplanning_run.add_argument("--domain", action="append", default=[])
    deepplanning_run.add_argument("--run-index", type=int, default=0)
    deepplanning_run.add_argument("--run-label", default="full")
    deepplanning_run.add_argument("--limit", type=int)
    deepplanning_run.add_argument("--restart", action="store_true")

    deepplanning_compare = subparsers.add_parser(
        "compare-deepplanning-configs",
        help="Verify the matched DeepPlanning arm configs.",
    )
    _add_config_argument(deepplanning_compare, default=DEEPPLANNING_CONFIG)
    deepplanning_compare.add_argument(
        "--baseline-config",
        type=Path,
        default=Path("configs/deepplanning/fewshot-ptc.toml"),
    )

    deepplanning_evaluate = subparsers.add_parser(
        "evaluate-deepplanning",
        help="Run official DeepPlanning conversion, evaluators, and aggregation.",
    )
    _add_config_argument(deepplanning_evaluate, default=DEEPPLANNING_CONFIG)
    deepplanning_evaluate.add_argument("--run-index", type=int, default=0)
    deepplanning_evaluate.add_argument("--run-label", default="full")

    deepplanning_result_compare = subparsers.add_parser(
        "compare-deepplanning",
        help="Create a matched paired DeepPlanning result report.",
    )
    _add_config_argument(deepplanning_result_compare, default=DEEPPLANNING_CONFIG)
    deepplanning_result_compare.add_argument(
        "--baseline-config",
        type=Path,
        default=Path("configs/deepplanning/fewshot-ptc.toml"),
    )
    deepplanning_result_compare.add_argument("--run-label", default="full")
    deepplanning_result_compare.add_argument("--run-index", type=int, default=0)
    deepplanning_result_compare.add_argument("--output", type=Path)

    return parser


def _add_config_argument(parser: argparse.ArgumentParser, *, default: str) -> None:
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(default),
        help=f"Experiment TOML path (default: {default}).",
    )
