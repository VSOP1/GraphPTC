from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

from graphptc.config import ExperimentConfig


def main() -> None:
    parser = argparse.ArgumentParser(description="Freeze the Stage 7.4c matched auto-progress pilot.")
    parser.add_argument("gate_path", type=Path)
    parser.add_argument("control_config", type=Path)
    parser.add_argument("placebo_config", type=Path)
    parser.add_argument("graph_config", type=Path)
    parser.add_argument("output_path", type=Path)
    args = parser.parse_args()
    paths = {
        "control": args.control_config,
        "placebo_auto": args.placebo_config,
        "graph_auto": args.graph_config,
    }
    configs = {name: ExperimentConfig.from_toml(path) for name, path in paths.items()}
    gate = json.loads(args.gate_path.read_text(encoding="utf-8"))
    modes = {name: config.runtime.graph_progress_mode for name, config in configs.items()}
    output_paths = [
        path
        for config in configs.values()
        for path in (
            config.benchmark.responses_path,
            config.benchmark.grades_path,
            config.benchmark.report_path,
        )
    ]
    checks = {
        "three_expected_arms": list(paths) == gate["arms"],
        "expected_modes": modes == {"control": "off", "placebo_auto": "placebo_auto", "graph_auto": "graph_auto"},
        "fewshot_prompt": all(config.browsecomp_plus.prompt_variant == gate["acceptance"]["prompt_variant"] for config in configs.values()),
        "matched_nonintervention_config": len({_matched_payload(config) for config in configs.values()}) == 1,
        "distinct_output_paths": len({str(path) for path in output_paths}) == len(output_paths),
        "outputs_absent_before_run": not any(path.exists() for path in output_paths),
        "stateful_tool_support_disabled": gate["acceptance"]["stateful_tool_support"] is False,
    }
    source_paths = [
        Path("src/graphptc/graph_progress.py"),
        Path("src/graphptc/ptc.py"),
        Path("src/graphptc/codeact_agent.py"),
        Path("src/graphptc/browsecomp_plus_benchmark.py"),
        Path("src/graphptc/config.py"),
    ]
    report = {
        "schema_version": 1,
        "stage": "7.4c",
        "mode": "preregistered-matched-auto-progress-pilot",
        "official_benchmark_result": False,
        "passed": all(checks.values()),
        "checks": checks,
        "acceptance": gate["acceptance"],
        "arms": {
            name: {
                "mode": modes[name],
                "config": str(paths[name]),
                "config_sha256": _sha256(paths[name]),
                "responses_path": str(configs[name].benchmark.responses_path),
            }
            for name in paths
        },
        "implementation": {str(path): _sha256(path) for path in source_paths},
        "gate_sha256": _sha256(args.gate_path),
        "boundary": "thresholds and implementation are frozen before any Stage 7.4c model call",
    }
    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    args.output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"passed": report["passed"], "checks": checks, "modes": modes}))
    if not report["passed"]:
        raise SystemExit(1)


def _matched_payload(config: ExperimentConfig) -> str:
    payload: dict[str, Any] = asdict(config)
    payload["runtime"].pop("graph_progress_mode")
    payload["benchmark"].pop("responses_path")
    payload["benchmark"].pop("grades_path")
    payload["benchmark"].pop("report_path")
    return json.dumps(payload, sort_keys=True, default=str, separators=(",", ":"))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


if __name__ == "__main__":
    main()
