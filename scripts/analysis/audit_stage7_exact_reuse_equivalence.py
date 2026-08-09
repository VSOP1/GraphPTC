from __future__ import annotations

import argparse
import copy
import hashlib
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

from graphptc.exact_reuse import ExactReuseSearchTools
from graphptc.persistent_runtime import PersistentIpcRuntime


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit frozen exact-reuse equivalence.")
    parser.add_argument("gate_path", type=Path)
    parser.add_argument("events_path", type=Path)
    parser.add_argument("output_path", type=Path)
    args = parser.parse_args()
    gate = json.loads(args.gate_path.read_text(encoding="utf-8"))
    events = _jsonl(args.events_path)
    segments = _successful_segments(events)
    cases = []
    for example_id in gate["example_ids"]:
        source = segments.get(example_id)
        if source is None:
            raise ValueError(f"Missing successful frozen episode: {example_id}")
        blocks = [event["data"]["code"] for event in source if event["type"] == "block.started"]
        fixtures, duplicates_stable = _fixtures(source)
        base_tools = FrozenTools(fixtures)
        cache_inner = FrozenTools(fixtures)
        cache_tools = ExactReuseSearchTools(cache_inner, max_tool_calls=1_000)
        base = _execute(blocks, base_tools)
        cached = _execute(blocks, cache_tools)
        cases.append(
            {
                "example_id": example_id,
                "block_count": len(blocks),
                "duplicate_results_stable": duplicates_stable,
                "block_outputs_equal": base["outputs"] == cached["outputs"],
                "runtime_results_equal": base["results"] == cached["results"],
                "final_state_equal": base["final_state"] == cached["final_state"],
                "logical_budget_equal": base_tools.consumed == cache_tools.consumed,
                "base_logical_calls": base_tools.consumed,
                "cache_logical_calls": cache_tools.consumed,
                "cache_live_calls": cache_inner.consumed,
                "cache_hits": cache_tools.cache_hits,
                "avoided_calls": cache_tools.consumed - cache_inner.consumed,
            }
        )
    acceptance = gate["acceptance"]
    checks = {
        "expected_cases": len(cases) == acceptance["expected_cases"],
        "block_outputs_equal_all": all(case["block_outputs_equal"] for case in cases),
        "runtime_results_equal_all": all(case["runtime_results_equal"] for case in cases),
        "final_state_equal_all": all(case["final_state_equal"] for case in cases),
        "logical_budget_equal_all": all(case["logical_budget_equal"] for case in cases),
        "duplicate_results_stable_all": all(case["duplicate_results_stable"] for case in cases),
        "cache_hits_positive_all": all(case["cache_hits"] > 0 for case in cases),
        "external_tool_calls": acceptance["external_tool_calls"] == 0,
    }
    report = {
        "schema_version": 1,
        "stage": "7.3a",
        "mode": gate["mode"],
        "official_benchmark_result": False,
        "passed": all(checks.values()),
        "checks": checks,
        "totals": {
            "logical_calls": sum(case["base_logical_calls"] for case in cases),
            "cache_live_calls": sum(case["cache_live_calls"] for case in cases),
            "cache_hits": sum(case["cache_hits"] for case in cases),
            "avoided_calls": sum(case["avoided_calls"] for case in cases),
        },
        "cases": cases,
        "artifacts": {
            str(args.gate_path): _sha256(args.gate_path),
            str(args.events_path): _sha256(args.events_path),
        },
    }
    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    args.output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"passed": report["passed"], "checks": checks, "totals": report["totals"]}))
    if not report["passed"]:
        raise SystemExit(1)


class FrozenTools:
    def __init__(self, fixtures: dict[tuple[str, str], Any]) -> None:
        self._fixtures = fixtures
        self._calls: list[dict[str, Any]] = []

    @property
    def calls(self) -> list[dict[str, Any]]:
        return list(self._calls)

    @property
    def consumed(self) -> int:
        return len(self._calls)

    def metadata(self) -> dict[str, Any]:
        return {"mode": "frozen"}

    def search(self, *, query: str) -> list[dict[str, Any]]:
        result = self._fixtures[("search", _normalize_query(query))]
        self._calls.append({"operation": "search", "query": query})
        return copy.deepcopy(result)

    def fetch(self, *, docid: str) -> dict[str, Any]:
        result = self._fixtures[("fetch", str(docid))]
        self._calls.append({"operation": "fetch", "docid": str(docid)})
        return copy.deepcopy(result)


def _execute(blocks: list[str], tools: Any) -> dict[str, Any]:
    runtime = PersistentIpcRuntime()
    outputs = []
    results = []
    try:
        for code in blocks:
            result = runtime.execute(
                code,
                namespace={"search": tools.search, "fetch": tools.fetch},
                timeout=120,
            )
            outputs.append({"stdout": result.stdout, "stderr": result.stderr})
            results.append(
                {
                    "return_code": result.return_code,
                    "timed_out": result.timed_out,
                }
            )
        final_state = runtime.last_state
    finally:
        runtime.close()
    return {"outputs": outputs, "results": results, "final_state": final_state}


def _successful_segments(events: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    complete: dict[str, list[list[dict[str, Any]]]] = defaultdict(list)
    current: dict[str, list[dict[str, Any]]] = {}
    for event in events:
        task_id = str(event["task_id"])
        if event["type"] == "episode.started":
            current[task_id] = []
        if task_id not in current:
            continue
        current[task_id].append(event)
        if event["type"] == "episode.finished":
            complete[task_id].append(current.pop(task_id))
    return {
        task_id: candidates[-1]
        for task_id, segments in complete.items()
        if (candidates := [
            segment
            for segment in segments
            if segment[-1].get("data", {}).get("status") == "success"
        ])
    }


def _fixtures(events: list[dict[str, Any]]) -> tuple[dict[tuple[str, str], Any], bool]:
    values: dict[tuple[str, str], Any] = {}
    stable = True
    for event in events:
        if event["type"] != "tool.called":
            continue
        data = event["data"]
        if data.get("success") is not True or "result" not in data:
            raise ValueError("Frozen equivalence cases require successful tool results")
        tool = str(data["tool"])
        arguments = data.get("arguments", {})
        key = (
            (tool, _normalize_query(arguments.get("query")))
            if tool == "search"
            else (tool, str(arguments.get("docid")))
        )
        result = data["result"]
        if key in values and _canonical(values[key]) != _canonical(result):
            stable = False
        values.setdefault(key, result)
    return values, stable


def _normalize_query(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "").strip().lower())


def _canonical(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


if __name__ == "__main__":
    main()
