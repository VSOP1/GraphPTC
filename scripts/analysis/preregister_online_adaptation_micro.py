from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from graphptc.browsecomp_plus import load_browsecomp_plus
from graphptc.browsecomp_plus_benchmark import (
    _retriever_metadata,
    _run_signature,
    _run_signature_payload,
)
from graphptc.config import ExperimentConfig


def main() -> None:
    parser = argparse.ArgumentParser(description="Preregister a matched online Adapt gate.")
    parser.add_argument("control_config", type=Path)
    parser.add_argument("adapt_config", type=Path)
    parser.add_argument("gate_config", type=Path)
    parser.add_argument("output_path", type=Path)
    args = parser.parse_args()
    control = ExperimentConfig.from_toml(args.control_config)
    adapt = ExperimentConfig.from_toml(args.adapt_config)
    gate = json.loads(args.gate_config.read_text(encoding="utf-8"))
    metadata = _retriever_metadata(control)
    control_payload = _run_signature_payload(control, metadata)
    adapt_payload = _run_signature_payload(adapt, metadata)
    _assert_matched(control_payload, adapt_payload)
    examples = load_browsecomp_plus(
        control.benchmark.dataset_path,
        expected_examples=control.browsecomp_plus.expected_examples,
    )[: gate["acceptance"]["expected_examples_per_arm"]]
    example_ids = [item.example_id for item in examples]
    if example_ids != gate["example_ids"]:
        raise ValueError(f"micro example IDs changed: {example_ids}")
    report = {
        "schema_version": 1,
        "mode": f"preregistered-{gate['mode']}",
        "gate_mode": gate["mode"],
        "official_benchmark_result": False,
        "development_subset": True,
        "example_ids": example_ids,
        "control_run_signature": _run_signature(control, metadata),
        "adapt_run_signature": _run_signature(adapt, metadata),
        "retriever_metadata": metadata,
        "acceptance": gate["acceptance"],
        "boundary": gate["boundary"],
        "artifacts": {
            str(path).replace("\\", "/"): _sha256(path)
            for path in (
                args.control_config,
                args.adapt_config,
                args.gate_config,
                control.benchmark.dataset_path,
            )
        },
    }
    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    args.output_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "example_ids": example_ids,
        "control_run_signature": report["control_run_signature"],
        "adapt_run_signature": report["adapt_run_signature"],
        "output": str(args.output_path),
    }))


def _assert_matched(control: dict[str, Any], adapt: dict[str, Any]) -> None:
    left = json.loads(json.dumps(control))
    right = json.loads(json.dumps(adapt))
    if left["runtime"].pop("graph_adaptation_mode") != "off":
        raise ValueError("control graph adaptation must be off")
    if right["runtime"].pop("graph_adaptation_mode") != "online":
        raise ValueError("adapt graph adaptation must be online")
    left.pop("runtime_tool_manifest")
    right.pop("runtime_tool_manifest")
    left.pop("ptc_tool_spec")
    right.pop("ptc_tool_spec")
    if left != right:
        raise ValueError("control and Adapt differ outside the registered intervention")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


if __name__ == "__main__":
    main()
