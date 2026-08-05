from __future__ import annotations

import argparse
import json
import os
from dataclasses import replace
from pathlib import Path

from dotenv import load_dotenv

from graphptc.config import ExperimentConfig
from graphptc.browsecomp_plus_benchmark import BROWSECOMP_PLUS_RUNTIME_TOOL_MANIFEST
from graphptc.model import OpenAIChatModel
from graphptc.patch_controller import GRAPHPTC_REPAIR_PROMPT_VARIANT
from graphptc.stage4_gate import write_stage4_model_gate_report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the bounded no-retry Stage 4 model repair gate."
    )
    parser.add_argument("config_path", type=Path)
    parser.add_argument("graph_path", type=Path)
    parser.add_argument("expectations_path", type=Path)
    parser.add_argument("output_path", type=Path)
    args = parser.parse_args()

    load_dotenv(".env")
    config = ExperimentConfig.from_toml(args.config_path)
    if config.browsecomp_plus.prompt_variant != GRAPHPTC_REPAIR_PROMPT_VARIANT:
        raise ValueError("Stage 4 model gate requires prompt_variant='fewshot-ptc-v1'")
    api_key = os.environ.get(config.model.api_key_env)
    if not api_key:
        raise ValueError(f"Missing environment variable: {config.model.api_key_env}")
    model_config = replace(config.model, max_retries=0)
    report = write_stage4_model_gate_report(
        OpenAIChatModel(model_config, api_key),
        args.graph_path,
        args.expectations_path,
        args.output_path,
        runtime_tool_manifest=BROWSECOMP_PLUS_RUNTIME_TOOL_MANIFEST,
    )
    print(
        json.dumps(
            {
                "passed": report["passed"],
                "case_count": report["case_count"],
                "model_request_count": report["model_request_count"],
                "location_match_rate": report["location_match_rate"],
                "patch_valid_rate": report["patch_valid_rate"],
                "reexecution_success_rate": report["reexecution_success_rate"],
                "output_path": str(args.output_path),
            },
            ensure_ascii=False,
        )
    )
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
