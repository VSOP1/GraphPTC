from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare exact-cache and graph-lineage diagnostics.")
    parser.add_argument("gate_path", type=Path)
    parser.add_argument("control_events", type=Path)
    parser.add_argument("reuse_events", type=Path)
    parser.add_argument("output_path", type=Path)
    args = parser.parse_args()
    gate = json.loads(args.gate_path.read_text(encoding="utf-8"))
    control_events = _jsonl(args.control_events)
    reuse_events = _jsonl(args.reuse_events)
    first = {
        "control": _audit_run(control_events),
        "reuse": _audit_run(reuse_events),
    }
    second = {
        "control": _audit_run(control_events),
        "reuse": _audit_run(reuse_events),
    }
    expected = gate["acceptance"]["expected_episodes_per_run"]
    all_signals = [
        signal
        for run in first.values()
        for episode in run["episodes"]
        for signal in episode["graph_incremental_signals"]
    ]
    checks = {
        "expected_episodes_per_run": all(
            run["episode_count"] == expected for run in first.values()
        ),
        "deterministic": first == second,
        "no_gold_features": not _contains_gold(first),
        "no_action_taken": all(run["action_taken"] is None for run in first.values()),
        "graph_incremental_signals_positive": all(
            run["totals"]["graph_incremental_zero_novelty_searches"] > 0
            for run in first.values()
        ),
        "graph_signals_disjoint_from_exact_cache": all(
            signal["exact_query_repeat"] is False for signal in all_signals
        ),
    }
    report = {
        "schema_version": 1,
        "stage": "7.3b",
        "mode": gate["mode"],
        "official_benchmark_result": False,
        "passed": all(checks.values()),
        "checks": checks,
        "runs": first,
        "artifacts": {
            str(args.gate_path): _sha256(args.gate_path),
            str(args.control_events): _sha256(args.control_events),
            str(args.reuse_events): _sha256(args.reuse_events),
        },
        "interpretation": {
            "simple_baseline": "exact normalized query/docid repetition",
            "graph_increment": "new query text whose result documents are all already present in the episode lineage",
            "boundary": "diagnostic opportunity only; no stopping or accuracy benefit is claimed",
        },
    }
    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    args.output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "passed": report["passed"],
                "checks": checks,
                "control": first["control"]["totals"],
                "reuse": first["reuse"]["totals"],
            }
        )
    )
    if not report["passed"]:
        raise SystemExit(1)


def _audit_run(events: list[dict[str, Any]]) -> dict[str, Any]:
    segments = _successful_segments(events)
    episodes = [_audit_episode(task_id, segment) for task_id, segment in sorted(segments.items(), key=lambda item: int(item[0]))]
    return {
        "episode_count": len(episodes),
        "model_visible": False,
        "action_taken": None,
        "totals": {
            "tool_calls": sum(item["tool_calls"] for item in episodes),
            "exact_query_repeats": sum(item["exact_query_repeats"] for item in episodes),
            "exact_fetch_repeats": sum(item["exact_fetch_repeats"] for item in episodes),
            "graph_incremental_zero_novelty_searches": sum(
                len(item["graph_incremental_signals"]) for item in episodes
            ),
            "episodes_with_graph_increment": sum(
                bool(item["graph_incremental_signals"]) for item in episodes
            ),
        },
        "episodes": episodes,
    }


def _audit_episode(task_id: str, events: list[dict[str, Any]]) -> dict[str, Any]:
    seen_queries: set[str] = set()
    seen_fetches: set[str] = set()
    seen_docids: set[str] = set()
    exact_query_repeats = 0
    exact_fetch_repeats = 0
    tool_calls = 0
    signals = []
    for event in events:
        if event["type"] != "tool.called":
            continue
        tool_calls += 1
        data = event["data"]
        arguments = data.get("arguments", {})
        if data.get("tool") == "search":
            query = _normalize(arguments.get("query"))
            exact_repeat = query in seen_queries
            result = data.get("result")
            docids = {
                str(item["docid"])
                for item in (result if isinstance(result, list) else [])
                if isinstance(item, dict) and item.get("docid") is not None
            }
            new_docids = docids - seen_docids
            if exact_repeat:
                exact_query_repeats += 1
            elif docids and not new_docids:
                signals.append(
                    {
                        "sequence": event["sequence"],
                        "block_id": event.get("block_id"),
                        "query": query,
                        "result_docids": sorted(docids),
                        "exact_query_repeat": False,
                        "new_docid_count": 0,
                    }
                )
            seen_queries.add(query)
            seen_docids.update(docids)
        elif data.get("tool") == "fetch":
            docid = str(arguments.get("docid", ""))
            if docid in seen_fetches:
                exact_fetch_repeats += 1
            seen_fetches.add(docid)
    return {
        "example_id": task_id,
        "tool_calls": tool_calls,
        "exact_query_repeats": exact_query_repeats,
        "exact_fetch_repeats": exact_fetch_repeats,
        "graph_incremental_signals": signals,
    }


def _successful_segments(events: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    complete: dict[str, list[list[dict[str, Any]]]] = defaultdict(list)
    active: dict[str, list[list[dict[str, Any]]]] = defaultdict(list)
    for event in events:
        task_id = str(event["task_id"])
        if event["type"] == "episode.started":
            active[task_id].append([event])
            continue
        candidates = [
            segment
            for segment in active[task_id]
            if int(segment[-1]["sequence"]) + 1 == int(event["sequence"])
        ]
        if len(candidates) != 1:
            continue
        segment = candidates[0]
        segment.append(event)
        if event["type"] == "episode.finished":
            active[task_id].remove(segment)
            complete[task_id].append(segment)
    result = {}
    for task_id, candidates in complete.items():
        successful = [
            segment
            for segment in candidates
            if segment[-1].get("data", {}).get("status") == "success"
        ]
        if successful:
            result[task_id] = successful[-1]
    return result


def _contains_gold(value: Any) -> bool:
    if isinstance(value, dict):
        return any("gold" in str(key).lower() or _contains_gold(item) for key, item in value.items())
    if isinstance(value, list):
        return any(_contains_gold(item) for item in value)
    return False


def _normalize(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "").strip().lower())


def _jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


if __name__ == "__main__":
    main()
