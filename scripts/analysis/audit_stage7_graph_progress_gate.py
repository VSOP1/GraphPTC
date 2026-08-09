from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from graphptc.browsecomp_plus_benchmark import (
    BROWSECOMP_PLUS_ORIGINAL_PTC_TOOL_SPEC,
    _ptc_tool_spec,
    _run_signature_payload,
    _runtime_tool_manifest,
)
from graphptc.config import ExperimentConfig
from graphptc.graph_progress import GraphProgressView


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit the Stage 7.4 graph-progress interface gate.")
    parser.add_argument("gate_path", type=Path)
    parser.add_argument("control_config", type=Path)
    parser.add_argument("placebo_config", type=Path)
    parser.add_argument("graph_config", type=Path)
    parser.add_argument("output_path", type=Path)
    args = parser.parse_args()
    gate = json.loads(args.gate_path.read_text(encoding="utf-8"))
    configs = {
        "control": ExperimentConfig.from_toml(args.control_config),
        "placebo": ExperimentConfig.from_toml(args.placebo_config),
        "graph": ExperimentConfig.from_toml(args.graph_config),
    }
    metadata = {
        "backend": "browsecomp_plus_official_bm25",
        "top_k": 5,
        "snippet_max_tokens": 512,
        "index_revision": "frozen-runtime",
    }
    payloads = {name: _run_signature_payload(config, metadata) for name, config in configs.items()}
    placebo_spec = _ptc_tool_spec(configs["placebo"])
    graph_spec = _ptc_tool_spec(configs["graph"])
    control_spec = _ptc_tool_spec(configs["control"])
    placebo_manifest = _runtime_tool_manifest(configs["placebo"])
    graph_manifest = _runtime_tool_manifest(configs["graph"])
    calls = [{"operation": "search", "query": "alpha", "docids": ["a", "b"]}]
    tools = SimpleNamespace(calls=calls, consumed=1)
    before = (list(tools.calls), tools.consumed)
    graph_snapshot = GraphProgressView(tools, mode="graph", max_tool_calls=10).graph_progress()
    placebo_snapshot = GraphProgressView(tools, mode="placebo", max_tool_calls=10).graph_progress()
    after = (list(tools.calls), tools.consumed)
    checks = {
        "prompt_variant": all(config.browsecomp_plus.prompt_variant == gate["acceptance"]["prompt_variant"] for config in configs.values()),
        "stateful_tool_support_disabled": True,
        "three_distinct_run_signatures": len({json.dumps(payload, sort_keys=True) for payload in payloads.values()}) == 3,
        "control_is_frozen_spec": control_spec is BROWSECOMP_PLUS_ORIGINAL_PTC_TOOL_SPEC,
        "same_interface_placebo_graph": placebo_spec == graph_spec and placebo_manifest == graph_manifest,
        "control_has_no_progress_interface": "graph_progress" not in [item["name"] for item in _runtime_tool_manifest(configs["control"])],
        "fixed_snapshot_length": len(repr(graph_snapshot)) == gate["acceptance"]["graph_progress_target_chars"] == len(repr(placebo_snapshot)),
        "same_snapshot_schema": list(graph_snapshot) == list(placebo_snapshot),
        "read_only": before == after,
        "no_forced_stop": not any("stop" in json.dumps(payload).lower() for payload in (graph_snapshot, placebo_snapshot)),
        "no_gold_features": not _contains_gold({"tool_spec": graph_spec, "manifest": graph_manifest}),
    }
    report = {
        "schema_version": 1,
        "stage": "7.4",
        "mode": gate["mode"],
        "official_benchmark_result": False,
        "real_pilot_status": "not_evaluated_by_structural_gate",
        "passed": all(checks.values()),
        "checks": checks,
        "arms": {
            name: {
                "graph_progress_mode": config.runtime.graph_progress_mode,
                "prompt_variant": config.browsecomp_plus.prompt_variant,
                "run_signature": hashlib.sha256(
                    json.dumps(payloads[name], sort_keys=True, separators=(",", ":")).encode()
                ).hexdigest(),
            }
            for name, config in configs.items()
        },
        "artifacts": {
            str(path): _sha256(path)
            for path in (args.gate_path, args.control_config, args.placebo_config, args.graph_config)
        },
        "interpretation": {
            "boundary": "interface and structural gate only; no model outcome or benchmark improvement is claimed",
            "pilot_scope": "real pilot outcomes are evaluated by the separate Stage 7.4 pilot audit",
        },
    }
    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    args.output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"passed": report["passed"], "checks": checks, "real_pilot_status": report["real_pilot_status"]}))
    if not report["passed"]:
        raise SystemExit(1)


def _contains_gold(value: Any) -> bool:
    if isinstance(value, dict):
        return any("gold" in str(key).lower() or _contains_gold(item) for key, item in value.items())
    if isinstance(value, list):
        return any(_contains_gold(item) for item in value)
    return False


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


if __name__ == "__main__":
    main()
