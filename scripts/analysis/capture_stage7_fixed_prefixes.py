from __future__ import annotations

import argparse
from pathlib import Path

from graphptc.config import ExperimentConfig
from graphptc.stage1 import run_stage1_browsecomp_plus


def main() -> None:
    parser = argparse.ArgumentParser(description="Capture exact Stage 7.5b model prefixes.")
    parser.add_argument("config_path", type=Path)
    parser.add_argument("gate_path", type=Path)
    parser.add_argument("events_path", type=Path)
    parser.add_argument("archive_dir", type=Path)
    args = parser.parse_args()
    import json

    config = ExperimentConfig.from_toml(args.config_path)
    gate = json.loads(args.gate_path.read_text(encoding="utf-8"))
    if config.browsecomp_plus.prompt_variant != gate["acceptance"]["prompt_variant"]:
        raise ValueError("Stage 7.5b requires the frozen fewshot prompt")
    if config.runtime.graph_progress_mode != gate["acceptance"]["graph_progress_mode"]:
        raise ValueError("Stage 7.5b requires graph_auto prefix capture")
    run_stage1_browsecomp_plus(
        config,
        events_path=args.events_path,
        example_ids=gate["capture_example_ids"],
        resume=False,
        checkpoint_archive_dir=args.archive_dir,
    )


if __name__ == "__main__":
    main()
