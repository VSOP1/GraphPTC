from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from graphptc.evidence_consumption import (
    FETCH_CLASSES,
    QUERY_CLASSES,
    project_evidence_consumption,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit the offline evidence-consumption frontier.")
    parser.add_argument("gate_path", type=Path)
    parser.add_argument("control_events", type=Path)
    parser.add_argument("reuse_events", type=Path)
    parser.add_argument("output_path", type=Path)
    args = parser.parse_args()
    gate = json.loads(args.gate_path.read_text(encoding="utf-8"))
    sources = {
        "control": args.control_events,
        "reuse": args.reuse_events,
    }
    first = {name: _audit_run(_jsonl(path)) for name, path in sources.items()}
    second = {name: _audit_run(_jsonl(path)) for name, path in sources.items()}
    expected = int(gate["acceptance"]["expected_episodes_per_run"])
    checks = {
        "expected_episodes_per_run": all(run["episode_count"] == expected for run in first.values()),
        "deterministic": first == second,
        "no_gold_features": not _contains_gold(first),
        "no_action_taken": all(run["action_taken"] is None for run in first.values()),
        "all_fetches_classified": all(
            run["totals"]["successful_fetches"]
            == sum(run["totals"]["fetch_classifications"].values())
            for run in first.values()
        ),
        "all_zero_novelty_queries_classified": all(
            run["totals"]["zero_novelty_queries"]
            == sum(run["totals"]["query_classifications"].values())
            for run in first.values()
        ),
        "observable_consumption_positive": all(
            run["totals"]["fetch_classifications"]["answer_lexical_support"]
            + run["totals"]["fetch_classifications"]["stdout_lineage"]
            + run["totals"]["fetch_classifications"]["later_state_load"]
            > 0
            for run in first.values()
        ),
    }
    report = {
        "schema_version": 1,
        "stage": "7.3c",
        "mode": gate["mode"],
        "official_benchmark_result": False,
        "passed": all(checks.values()),
        "checks": checks,
        "runs": first,
        "artifacts": {
            str(args.gate_path): _sha256(args.gate_path),
            **{str(path): _sha256(path) for path in sources.values()},
        },
        "interpretation": {
            "answer_lexical_support": "the model final answer appears verbatim in fetched content; this is support compatibility, not causal attribution",
            "stdout_lineage": "the execution graph has a static DATA path from the fetch/search call to block stdout",
            "later_state_load": "fetch-derived persistent state is loaded by a later block",
            "unresolved": "no observable lineage was recovered; this must not be interpreted as unused or redundant",
            "boundary": "offline shadow only; no prompt, tool result, budget, stopping, or final answer is changed",
        },
    }
    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    args.output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"passed": report["passed"], "checks": checks, "runs": {name: run["totals"] for name, run in first.items()}}))
    if not report["passed"]:
        raise SystemExit(1)


def _audit_run(events: list[dict[str, Any]]) -> dict[str, Any]:
    segments = _successful_segments(events)
    episodes = [
        project_evidence_consumption(segment)
        for _, segment in sorted(segments.items(), key=lambda item: int(item[0]))
    ]
    fetch_counts: Counter[str] = Counter()
    query_counts: Counter[str] = Counter()
    for episode in episodes:
        fetch_counts.update(episode["metrics"]["fetch_classifications"])
        query_counts.update(episode["metrics"]["query_classifications"])
    return {
        "episode_count": len(episodes),
        "model_visible": False,
        "action_taken": None,
        "totals": {
            "successful_fetches": sum(item["metrics"]["successful_fetches"] for item in episodes),
            "fetch_classifications": {name: fetch_counts[name] for name in FETCH_CLASSES},
            "zero_novelty_queries": sum(item["metrics"]["zero_novelty_queries"] for item in episodes),
            "query_classifications": {name: query_counts[name] for name in QUERY_CLASSES},
        },
        "episodes": episodes,
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
    return {
        task_id: successful[-1]
        for task_id, candidates in complete.items()
        if (successful := [
            segment
            for segment in candidates
            if segment[-1].get("data", {}).get("status") == "success"
        ])
    }


def _contains_gold(value: Any) -> bool:
    if isinstance(value, dict):
        return any("gold" in str(key).lower() or _contains_gold(item) for key, item in value.items())
    if isinstance(value, list):
        return any(_contains_gold(item) for item in value)
    return False


def _jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


if __name__ == "__main__":
    main()
