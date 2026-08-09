from __future__ import annotations

import argparse
from pathlib import Path

from graphptc.config import ExperimentConfig
from graphptc.observability import ExecutionObserver, JsonlEventSink
from graphptc.stage1 import run_stage1_browsecomp_plus


def main() -> None:
    parser = argparse.ArgumentParser(description="Run one Stage 7.4 graph-progress pilot arm.")
    parser.add_argument("config_path", type=Path)
    parser.add_argument("events_path", type=Path)
    args = parser.parse_args()
    config = ExperimentConfig.from_toml(args.config_path)
    if config.browsecomp_plus.prompt_variant != "fewshot-ptc-v1":
        raise ValueError("Stage 7.4 requires fewshot-ptc-v1")
    if config.runtime.graph_progress_mode not in {
        "off",
        "placebo",
        "graph",
        "placebo_auto",
        "graph_auto",
    }:
        raise ValueError("unsupported Stage 7.4 graph progress mode")
    run_stage1_browsecomp_plus(
        config,
        events_path=args.events_path,
        limit=20,
        resume=True,
    )


if __name__ == "__main__":
    main()
