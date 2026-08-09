from __future__ import annotations

import argparse
import ast
import hashlib
import json
from pathlib import Path
from typing import Any


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare fixed-prefix capsule effects with graph self-drift.")
    parser.add_argument("config_path", type=Path)
    parser.add_argument("output_path", type=Path)
    args = parser.parse_args()
    config = _json(args.config_path)
    inputs = config["inputs"]
    paths = {
        "selection": Path(inputs["selection_path"]),
        "swap": Path(inputs["swap_path"]),
        "capture_events": Path(inputs["capture_events_path"]),
    }
    hash_checks = {
        name: _sha256(path) == inputs[f"{name}_sha256"]
        for name, path in paths.items()
    }
    selection = _json(paths["selection"])
    swaps = {item["prefix_id"]: item for item in _jsonl(paths["swap"])}
    original = _original_actions(paths["capture_events"])
    comparisons = []
    for prefix in selection["prefixes"]:
        prefix_id = prefix["prefix_id"]
        swap = swaps[prefix_id]
        key = (str(prefix["example_id"]), int(prefix["next_turn"]))
        original_action = original.get(key)
        graph_action = swap["conditions"]["graph"].get("action")
        placebo_action = swap["conditions"]["placebo"].get("action")
        comparisons.append({
            "prefix_id": prefix_id,
            "selection_reason": prefix["selection_reason"],
            "original_graph_action": original_action,
            "replayed_graph_action": graph_action,
            "replayed_placebo_action": placebo_action,
            "self_drift_coarse": _coarse(original_action) != _coarse(graph_action),
            "capsule_difference_coarse": _coarse(graph_action) != _coarse(placebo_action),
            "self_drift_exact": _exact(original_action) != _exact(graph_action),
            "capsule_difference_exact": _exact(graph_action) != _exact(placebo_action),
        })
    self_coarse = sum(item["self_drift_coarse"] for item in comparisons)
    capsule_coarse = sum(item["capsule_difference_coarse"] for item in comparisons)
    checks = {
        "input_hash_match": all(hash_checks.values()),
        "all_selected_prefixes_compared": len(comparisons) == len(selection["prefixes"]),
        "original_actions_available": all(item["original_graph_action"] is not None for item in comparisons),
    }
    complete = all(checks.values())
    distinguishable = complete and capsule_coarse > self_coarse
    report = {
        "schema_version": 1,
        "stage": "7.5c",
        "mode": config["mode"],
        "official_benchmark_result": False,
        "development_subset": True,
        "diagnostic_complete": complete,
        "local_capsule_effect_distinguishable": distinguishable,
        "checks": checks,
        "paired_prefixes": len(comparisons),
        "self_drift_coarse_divergences": self_coarse,
        "capsule_coarse_divergences": capsule_coarse,
        "self_drift_exact_divergences": sum(item["self_drift_exact"] for item in comparisons),
        "capsule_exact_divergences": sum(item["capsule_difference_exact"] for item in comparisons),
        "comparisons": comparisons,
        "interpretation_rule": config["interpretation_rule"],
        "boundary": config["boundary"],
    }
    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    args.output_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "diagnostic_complete": complete,
        "local_capsule_effect_distinguishable": distinguishable,
        "self_drift_coarse_divergences": self_coarse,
        "capsule_coarse_divergences": capsule_coarse,
        "self_drift_exact_divergences": report["self_drift_exact_divergences"],
        "capsule_exact_divergences": report["capsule_exact_divergences"],
    }))
    if not complete:
        raise SystemExit(1)


def _original_actions(path: Path) -> dict[tuple[str, int], dict[str, Any]]:
    values: dict[tuple[str, int], dict[str, Any]] = {}
    for event in _jsonl(path):
        if event.get("type") != "block.started":
            continue
        code = str(event.get("data", {}).get("code", ""))
        values[(str(event["task_id"]), int(event["data"]["turn"]))] = _code_action(code)
    return values


def _code_action(code: str) -> dict[str, Any]:
    search_sites = 0
    fetch_sites = 0
    syntax_valid = True
    try:
        tree = ast.parse(code)
    except SyntaxError:
        syntax_valid = False
    else:
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Name):
                continue
            search_sites += node.func.id == "search"
            fetch_sites += node.func.id == "fetch"
    return {
        "kind": "tool",
        "tool_calls": 1,
        "code_sha256": [hashlib.sha256(code.encode()).hexdigest()],
        "code_chars": len(code),
        "syntax_valid": syntax_valid,
        "static_search_sites": search_sites,
        "static_fetch_sites": fetch_sites,
    }


def _coarse(action: dict[str, Any] | None) -> tuple[Any, ...] | None:
    if action is None:
        return None
    return (action["kind"], action["static_search_sites"], action["static_fetch_sites"])


def _exact(action: dict[str, Any] | None) -> Any:
    return None if action is None else action["code_sha256"]


def _json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


if __name__ == "__main__":
    main()
