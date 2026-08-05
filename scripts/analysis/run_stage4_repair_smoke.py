from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict
from pathlib import Path

from dotenv import load_dotenv

from graphptc.config import ExperimentConfig
from graphptc.browsecomp_plus_benchmark import BROWSECOMP_PLUS_RUNTIME_TOOL_MANIFEST
from graphptc.failure_attribution import build_failure_contexts
from graphptc.model import OpenAIChatModel
from graphptc.patch_controller import (
    GRAPHPTC_REPAIR_PROMPT_VARIANT,
    apply_local_patch,
    build_repair_context,
)
from graphptc.stage2_graph import load_dependency_graph_report
from graphptc.stage4_repair import request_local_patch, reexecute_patch_prefix


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run one bounded Stage 4 repair and full-prefix reexecution smoke."
    )
    parser.add_argument("config_path", type=Path)
    parser.add_argument("graph_path", type=Path)
    parser.add_argument("episode_id")
    parser.add_argument("output_path", type=Path)
    args = parser.parse_args()

    load_dotenv(".env")
    config = ExperimentConfig.from_toml(args.config_path)
    if config.browsecomp_plus.prompt_variant != GRAPHPTC_REPAIR_PROMPT_VARIANT:
        raise ValueError("Stage 4 repair smoke requires prompt_variant='fewshot-ptc-v1'")
    api_key = os.environ.get(config.model.api_key_env)
    if not api_key:
        raise ValueError(f"Missing environment variable: {config.model.api_key_env}")
    graphs = [
        graph
        for graph in load_dependency_graph_report(args.graph_path)
        if graph.episode_id == args.episode_id
    ]
    if len(graphs) != 1:
        raise ValueError(f"Expected exactly one graph for episode: {args.episode_id}")
    graph = graphs[0]
    contexts = build_failure_contexts(graph)
    if len(contexts) != 1:
        raise ValueError("Stage 4 repair smoke requires exactly one failure context")

    repair = build_repair_context(
        graph,
        contexts[0],
        runtime_tool_manifest=BROWSECOMP_PLUS_RUNTIME_TOOL_MANIFEST,
    )
    generated = request_local_patch(
        OpenAIChatModel(config.model, api_key),
        repair,
        timeout_seconds=config.model.timeout_seconds,
        max_completion_tokens=min(config.model.max_completion_tokens, 1024),
    )
    application = apply_local_patch(graph, repair, generated.proposal)
    reexecution = reexecute_patch_prefix(
        graph,
        application,
        timeout_seconds=config.runtime.code_timeout_seconds,
    )
    report = {
        "schema_version": 1,
        "episode_id": graph.episode_id,
        "task_id": graph.task_id,
        "source_events_sha256": graph.source_events_sha256,
        "model": config.model.model,
        "thinking": config.model.thinking,
        "prompt_variant": repair.prompt_variant,
        "repair_context": repair.to_dict(),
        "generated_patch": asdict(generated),
        "application": application.to_dict(),
        "reexecution": asdict(reexecution),
    }
    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    args.output_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "episode_id": graph.episode_id,
                "patch_valid": True,
                "reexecution_success": reexecution.success,
                "model_requests": 1,
                "output_path": str(args.output_path),
            },
            ensure_ascii=False,
        )
    )
    if not reexecution.success:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
