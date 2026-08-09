from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from graphptc.actionable_frontier import load_jsonl, project_actionable_frontier


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit Stage 7.6a actionable frontiers.")
    parser.add_argument("config_path", type=Path)
    parser.add_argument("output_path", type=Path)
    args = parser.parse_args()
    config = _json(args.config_path)
    max_items = config["frontier"]["max_items"]
    arms: dict[str, dict[str, Any]] = {}
    hash_checks: dict[str, bool] = {}
    for name, value in config["inputs"].items():
        path = Path(value["events_path"])
        hash_checks[name] = _sha256(path) == value["events_sha256"]
        arms[name] = project_actionable_frontier(load_jsonl(path), max_items=max_items)
    combined = _combined(arms)
    acceptance = config["acceptance"]
    graph_coverage = combined["ranking"]["graph"]["opportunity_coverage"]
    simple_coverage = max(
        combined["ranking"][name]["opportunity_coverage"]
        for name in ("recency", "first_seen")
    )
    checks = {
        "input_hash_match": all(hash_checks.values()),
        "expected_episodes_each": all(
            arm["episode_count"] == acceptance["expected_episodes_per_arm"]
            for arm in arms.values()
        ),
        "trigger_rate_selective": acceptance["minimum_trigger_rate"]
        <= combined["trigger_rate"]
        <= acceptance["maximum_trigger_rate"],
        "target_opportunities_sufficient": combined["target_opportunities"]
        >= acceptance["minimum_target_opportunities"],
        "graph_top3_coverage": graph_coverage
        >= acceptance["minimum_graph_top3_target_coverage"],
        "graph_beats_simple_frontier": graph_coverage >= simple_coverage,
        "frontier_bounded": all(
            len(item["frontiers"][ranking]) <= acceptance["maximum_frontier_items"]
            for arm in arms.values()
            for item in arm["opportunities"]
            for ranking in ("graph", "recency", "first_seen")
        ),
        "concrete_lineage_complete": _lineage_complete(arms),
        "no_gold_features": acceptance["gold_features_allowed"] is False,
    }
    passed = all(checks.values())
    report = {
        "schema_version": 1,
        "stage": "7.6a",
        "mode": config["mode"],
        "official_benchmark_result": False,
        "development_subset": True,
        "passed": passed,
        "online_frontier_eligible": passed,
        "checks": checks,
        "arms": arms,
        "combined": combined,
        "boundary": config["boundary"],
        "interpretation": {
            "coverage": "coverage measures observed next first-time fetches, not answer relevance or causal benefit",
            "next": "implement no online frontier if this gate fails; revise ranking offline against frozen trajectories",
        },
    }
    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    args.output_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "passed": passed,
        "checks": checks,
        "combined": combined,
    }))
    if not passed:
        raise SystemExit(1)


def _combined(arms: dict[str, dict[str, Any]]) -> dict[str, Any]:
    opportunities = [item for arm in arms.values() for item in arm["opportunities"]]
    successful = sum(arm["successful_blocks"] for arm in arms.values())
    triggered = sum(arm["triggered_blocks"] for arm in arms.values())
    target = [item for item in opportunities if item["next_action"]["eligible_first_time_fetches"]]
    return {
        "successful_blocks": successful,
        "triggered_blocks": triggered,
        "trigger_rate": triggered / successful if successful else 0.0,
        "actionable_opportunities": len(opportunities),
        "target_opportunities": len(target),
        "ranking": {
            ranking: _coverage(target, ranking)
            for ranking in ("graph", "recency", "first_seen")
        },
    }


def _coverage(values: list[dict[str, Any]], ranking: str) -> dict[str, Any]:
    hits = sum(
        bool(
            {entry["docid"] for entry in item["frontiers"][ranking]}
            & set(item["next_action"]["eligible_first_time_fetches"])
        )
        for item in values
    )
    return {
        "opportunity_hits": hits,
        "opportunity_coverage": hits / len(values) if values else 0.0,
    }


def _lineage_complete(arms: dict[str, dict[str, Any]]) -> bool:
    required = {"docid", "source_query", "source_turn", "retrieval_count", "first_seen_call", "last_seen_call", "best_rank"}
    return all(
        required <= set(entry) and bool(entry["docid"]) and bool(entry["source_query"])
        for arm in arms.values()
        for item in arm["opportunities"]
        for entries in item["frontiers"].values()
        for entry in entries
    )


def _json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


if __name__ == "__main__":
    main()
