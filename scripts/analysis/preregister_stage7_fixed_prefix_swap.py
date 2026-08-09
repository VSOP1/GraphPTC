from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from graphptc.config import ExperimentConfig


def main() -> None:
    parser = argparse.ArgumentParser(description="Freeze the Stage 7.5b prefix-swap micro-gate.")
    parser.add_argument("gate_path", type=Path)
    parser.add_argument("capture_config_path", type=Path)
    parser.add_argument("events_path", type=Path)
    parser.add_argument("archive_dir", type=Path)
    parser.add_argument("selection_path", type=Path)
    parser.add_argument("swap_output_path", type=Path)
    parser.add_argument("output_path", type=Path)
    args = parser.parse_args()
    gate = _json(args.gate_path)
    config = ExperimentConfig.from_toml(args.capture_config_path)
    outputs = [
        config.benchmark.responses_path,
        config.benchmark.grades_path,
        config.benchmark.report_path,
        args.events_path,
        args.archive_dir,
        args.selection_path,
        args.swap_output_path,
    ]
    checks = {
        "fewshot_prompt": config.browsecomp_plus.prompt_variant == gate["acceptance"]["prompt_variant"],
        "graph_auto_capture": config.runtime.graph_progress_mode == gate["acceptance"]["graph_progress_mode"],
        "four_fixed_examples": len(gate["capture_example_ids"]) == gate["acceptance"]["expected_capture_examples"],
        "outputs_absent_before_run": not any(path.exists() for path in outputs),
        "stateful_tool_support_disabled": gate["acceptance"]["stateful_tool_support"] is False,
        "single_action_no_execution": gate["intervention"]["execute_generated_tools"] is False,
    }
    source_paths = [
        Path("src/graphptc/graph_progress.py"),
        Path("src/graphptc/ptc.py"),
        Path("src/graphptc/browsecomp_plus_benchmark.py"),
        Path("src/graphptc/stage1.py"),
        Path("scripts/analysis/capture_stage7_fixed_prefixes.py"),
        Path("scripts/analysis/select_stage7_fixed_prefixes.py"),
        Path("scripts/analysis/run_stage7_fixed_prefix_swap.py"),
        Path("scripts/analysis/audit_stage7_fixed_prefix_swap.py"),
    ]
    report = {
        "schema_version": 1,
        "stage": "7.5b",
        "mode": "preregistered-fixed-prefix-last-capsule-swap",
        "official_benchmark_result": False,
        "passed": all(checks.values()),
        "checks": checks,
        "gate": gate,
        "capture_config_sha256": _sha256(args.capture_config_path),
        "gate_sha256": _sha256(args.gate_path),
        "implementation": {str(path): _sha256(path) for path in source_paths},
        "outputs": [str(path) for path in outputs],
        "boundary": "selection, intervention, implementation, and output paths frozen before capture calls",
    }
    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    args.output_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"passed": report["passed"], "checks": checks}))
    if not report["passed"]:
        raise SystemExit(1)


def _json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


if __name__ == "__main__":
    main()
