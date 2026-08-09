from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from graphptc.graph_adaptation import load_jsonl, project_shadow_adaptation


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit Stage 7.6b shadow graph adaptation.")
    parser.add_argument("config_path", type=Path)
    parser.add_argument("output_path", type=Path)
    args = parser.parse_args()
    config = _json(args.config_path)
    acceptance = config["acceptance"]
    arms: dict[str, dict[str, Any]] = {}
    hash_checks: dict[str, bool] = {}
    deterministic_checks: dict[str, bool] = {}
    for name, value in config["inputs"].items():
        path = Path(value["events_path"])
        hash_checks[name] = _sha256(path) == value["events_sha256"]
        first = project_shadow_adaptation(
            load_jsonl(path), max_frontier_items=acceptance["maximum_frontier_items"]
        )
        second = project_shadow_adaptation(
            load_jsonl(path), max_frontier_items=acceptance["maximum_frontier_items"]
        )
        deterministic_checks[name] = _stable(first) == _stable(second)
        arms[name] = first
    successful = sum(arm["successful_blocks"] for arm in arms.values())
    triggered = sum(arm["triggered_blocks"] for arm in arms.values())
    trigger_rate = triggered / successful if successful else 0.0
    proposals = [item for arm in arms.values() for item in arm["proposals"]]
    checks = {
        "input_hash_match": all(hash_checks.values()),
        "expected_episodes_each": all(
            arm["episode_count"] == acceptance["expected_episodes_per_arm"]
            for arm in arms.values()
        ),
        "trigger_rate_selective": acceptance["minimum_trigger_rate"]
        <= trigger_rate
        <= acceptance["maximum_trigger_rate"],
        "deterministic_reprojection": all(deterministic_checks.values()),
        "shadow_only": all(
            arm["model_visible"] is acceptance["model_visible"]
            and arm["action_taken"] is acceptance["action_taken"]
            for arm in arms.values()
        ),
        "allowed_actions_only": all(
            item["proposed_action"]["action"] in acceptance["allowed_actions"]
            for item in proposals
        ),
        "no_action_executed": all(
            item["proposed_action"]["action_taken"] is None
            and item["proposed_action"]["model_visible"] is False
            for item in proposals
        ),
        "frontier_bounded": all(
            len(item["frontier"]) <= acceptance["maximum_frontier_items"]
            for item in proposals
        ),
        "frontier_lineage_complete": _lineage_complete(proposals),
        "no_gold_features": acceptance["gold_features_allowed"] is False,
    }
    passed = all(checks.values())
    report = {
        "schema_version": 1,
        "stage": "7.6b",
        "mode": config["mode"],
        "official_benchmark_result": False,
        "development_subset": True,
        "passed": passed,
        "online_adaptation_eligible": False,
        "checks": checks,
        "combined": {
            "successful_blocks": successful,
            "triggered_blocks": triggered,
            "trigger_rate": trigger_rate,
            "action_distribution": _action_distribution(proposals),
        },
        "arms": arms,
        "boundary": config["boundary"],
        "interpretation": {
            "result": "shadow framework structurally accepted" if passed else "shadow framework gate failed",
            "promotion": "passing this gate does not authorize model-visible or executed adaptation",
        },
    }
    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    args.output_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"passed": passed, "checks": checks, "combined": report["combined"]}))
    if not passed:
        raise SystemExit(1)


def _stable(value: dict[str, Any]) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def _lineage_complete(proposals: list[dict[str, Any]]) -> bool:
    required = {"docid", "source_query", "source_turn", "source_call", "result_rank", "retrieval_count"}
    return all(
        required <= set(item) and bool(item["docid"]) and bool(item["source_query"])
        for proposal in proposals for item in proposal["frontier"]
    )


def _action_distribution(proposals: list[dict[str, Any]]) -> dict[str, int]:
    values: dict[str, int] = {}
    for item in proposals:
        action = item["proposed_action"]["action"]
        values[action] = values.get(action, 0) + 1
    return values


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
