from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from graphptc.browsecomp_plus_benchmark import (
    _retriever_metadata,
    _run_signature,
    _run_signature_payload,
)
from graphptc.config import ExperimentConfig


def main() -> None:
    parser = argparse.ArgumentParser(description="Freeze GraphPTC active-v1 inputs.")
    parser.add_argument("control_config", type=Path)
    parser.add_argument("active_config", type=Path)
    parser.add_argument("gate_path", type=Path)
    parser.add_argument("stage63_report", type=Path)
    parser.add_argument("output_path", type=Path)
    args = parser.parse_args()

    control = ExperimentConfig.from_toml(args.control_config)
    active = ExperimentConfig.from_toml(args.active_config)
    metadata = _retriever_metadata(control)
    control_payload = _run_signature_payload(control, metadata)
    active_payload = _run_signature_payload(active, metadata)
    if control_payload != active_payload:
        raise ValueError("control and active run-signature payloads differ")

    root = Path(__file__).parents[2]
    source_paths = [
        root / "src" / "graphptc" / name
        for name in (
            "browsecomp_plus_benchmark.py",
            "codeact_agent.py",
            "failure_attribution.py",
            "invalidation.py",
            "observability.py",
            "patch_controller.py",
            "persistent_runtime.py",
            "persistent_worker.py",
            "replay_commit.py",
            "selective_replay.py",
            "stage1.py",
            "stage2_graph.py",
            "stage4_repair.py",
            "stage6_active.py",
        )
    ]
    artifact_paths = [
        args.control_config,
        args.active_config,
        args.gate_path,
        args.stage63_report,
        control.benchmark.dataset_path,
        control.browsecomp_plus.qrels_gold_path,
        control.browsecomp_plus.qrels_evidence_path,
        *source_paths,
    ]
    signature = _run_signature(control, metadata)
    report = {
        "schema_version": 1,
        "variant": "graphptc-active-v1",
        "official_benchmark_result": False,
        "matched_run_signature": signature,
        "control_active_payload_equal": True,
        "run_signature_payload": control_payload,
        "artifacts": {
            str(path.resolve().relative_to(root.resolve())).replace("\\", "/"): _sha256(
                path
            )
            for path in artifact_paths
        },
    }
    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    args.output_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({"signature": signature, "output": str(args.output_path)}))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


if __name__ == "__main__":
    main()
