from __future__ import annotations

import argparse
import json
from pathlib import Path

from dotenv import load_dotenv

from graphptc.config import ExperimentConfig
from graphptc.tau_knowledge_benchmark import (
    DEFAULT_PROTOCOL_PATH,
    compare_tau_knowledge_benchmarks,
    inspect_tau_knowledge,
    load_tau_knowledge_protocol,
    run_tau_knowledge_benchmark,
    validate_tau_knowledge_alignment,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Frozen tau-Knowledge matched evaluation"
    )
    parser.add_argument("--protocol", default=str(DEFAULT_PROTOCOL_PATH))
    subparsers = parser.add_subparsers(dest="command", required=True)

    inspect = subparsers.add_parser("inspect")
    inspect.add_argument("--config", default="configs/tau_knowledge/graphptc.toml")

    run = subparsers.add_parser("run")
    run.add_argument("--config", required=True)
    run.add_argument("--smoke", action="store_true")
    run.add_argument("--restart", action="store_true")

    compare = subparsers.add_parser("compare")
    compare.add_argument("--graph-config", required=True)
    compare.add_argument("--baseline-config", required=True)
    compare.add_argument("--output", required=True)
    return parser


def main() -> int:
    load_dotenv()
    args = _parser().parse_args()
    protocol = load_tau_knowledge_protocol(args.protocol)
    if args.command == "inspect":
        config = ExperimentConfig.from_toml(args.config)
        inspection = inspect_tau_knowledge(config, protocol)
        validate_tau_knowledge_alignment(config, inspection, protocol)
        print(json.dumps(inspection, ensure_ascii=False, indent=2))
        return 0
    if args.command == "run":
        config = ExperimentConfig.from_toml(args.config)
        task_ids = protocol["smoke_task_ids"] if args.smoke else ()
        summary = run_tau_knowledge_benchmark(
            config,
            protocol=protocol,
            task_ids=task_ids,
            restart=args.restart,
        )
        print(json.dumps(summary.to_dict(), ensure_ascii=False, indent=2))
        return 0
    report = compare_tau_knowledge_benchmarks(
        ExperimentConfig.from_toml(args.graph_config),
        ExperimentConfig.from_toml(args.baseline_config),
        output_path=Path(args.output),
        protocol=protocol,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
