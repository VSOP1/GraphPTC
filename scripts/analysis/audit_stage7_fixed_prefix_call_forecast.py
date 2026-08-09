from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from graphptc.static_tool_forecast import forecast_tool_calls


def main() -> None:
    parser = argparse.ArgumentParser(description="Forecast literal-loop calls in Stage 7.5b actions.")
    parser.add_argument("selection_path", type=Path)
    parser.add_argument("swap_path", type=Path)
    parser.add_argument("capture_events_path", type=Path)
    parser.add_argument("output_path", type=Path)
    args = parser.parse_args()
    selection = _json(args.selection_path)
    swaps = {item["prefix_id"]: item for item in _jsonl(args.swap_path)}
    original = _original_blocks(args.capture_events_path)
    comparisons = []
    for prefix in selection["prefixes"]:
        prefix_id = prefix["prefix_id"]
        pair = swaps[prefix_id]
        key = (str(prefix["example_id"]), int(prefix["next_turn"]))
        original_block = original[key]
        graph_code = _code(pair, "graph")
        placebo_code = _code(pair, "placebo")
        comparisons.append({
            "prefix_id": prefix_id,
            "original": {
                "forecast": forecast_tool_calls(original_block["code"]),
                "actual_search_calls": original_block["search_calls"],
                "actual_fetch_calls": original_block["fetch_calls"],
            },
            "graph": forecast_tool_calls(graph_code),
            "placebo": forecast_tool_calls(placebo_code),
        })
    validation = [
        item
        for item in comparisons
        if item["original"]["forecast"]["fully_determined"]
    ]
    exact_validation = all(
        item["original"]["forecast"]["known_search_calls"] == item["original"]["actual_search_calls"]
        and item["original"]["forecast"]["known_fetch_calls"] == item["original"]["actual_fetch_calls"]
        for item in validation
    )
    comparable = [
        item for item in comparisons
        if item["graph"]["fully_determined"] and item["placebo"]["fully_determined"]
    ]
    report = {
        "schema_version": 1,
        "stage": "7.5d",
        "mode": "fixed-prefix-literal-loop-call-forecast",
        "official_benchmark_result": False,
        "development_subset": True,
        "diagnostic_complete": bool(validation) and exact_validation and bool(comparable),
        "validated_original_actions": len(validation),
        "original_forecasts_exact": exact_validation,
        "comparable_replay_pairs": len(comparable),
        "forecast_totals": {
            condition: {
                "search_calls": sum(item[condition]["known_search_calls"] for item in comparable),
                "fetch_calls": sum(item[condition]["known_fetch_calls"] for item in comparable),
            }
            for condition in ("graph", "placebo")
        },
        "comparisons": comparisons,
        "boundary": "static forecast expands only provably bounded loops; generated tools remain unexecuted",
    }
    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    args.output_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "diagnostic_complete": report["diagnostic_complete"],
        "validated_original_actions": len(validation),
        "original_forecasts_exact": exact_validation,
        "comparable_replay_pairs": len(comparable),
        "forecast_totals": report["forecast_totals"],
    }))
    if not report["diagnostic_complete"]:
        raise SystemExit(1)


def _original_blocks(path: Path) -> dict[tuple[str, int], dict[str, Any]]:
    blocks: dict[str, dict[str, Any]] = {}
    values: dict[tuple[str, int], dict[str, Any]] = {}
    for event in _jsonl(path):
        if event["type"] == "block.started":
            blocks[str(event["block_id"])] = {
                "task_id": str(event["task_id"]),
                "turn": int(event["data"]["turn"]),
                "code": str(event["data"]["code"]),
                "search_calls": 0,
                "fetch_calls": 0,
            }
        elif event["type"] == "tool.called":
            block = blocks[str(event["block_id"])]
            tool = event["data"]["tool"]
            block["search_calls"] += tool == "search"
            block["fetch_calls"] += tool == "fetch"
        elif event["type"] == "block.finished":
            block = blocks[str(event["block_id"])]
            values[(block["task_id"], block["turn"])] = block
    return values


def _code(pair: dict[str, Any], condition: str) -> str:
    calls = pair["conditions"][condition]["assistant_message"].get("tool_calls") or []
    if len(calls) != 1:
        return ""
    arguments = json.loads(calls[0]["function"]["arguments"])
    return str(arguments.get("code", ""))


def _json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


if __name__ == "__main__":
    main()
