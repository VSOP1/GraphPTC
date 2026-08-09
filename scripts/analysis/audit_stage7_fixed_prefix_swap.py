from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit Stage 7.5b fixed-prefix capsule swaps.")
    parser.add_argument("gate_path", type=Path)
    parser.add_argument("preregistration_path", type=Path)
    parser.add_argument("selection_path", type=Path)
    parser.add_argument("swap_path", type=Path)
    parser.add_argument("output_path", type=Path)
    args = parser.parse_args()
    gate = _json(args.gate_path)
    preregistration = _json(args.preregistration_path)
    selection = _json(args.selection_path)
    pairs = _jsonl(args.swap_path)
    expected_ids = {item["prefix_id"] for item in selection["prefixes"]}
    actual_ids = {item["prefix_id"] for item in pairs}
    errors = sum(
        condition.get("status") != "success"
        for pair in pairs
        for condition in pair.get("conditions", {}).values()
    )
    checks = {
        "preregistration_passed": preregistration.get("passed") is True,
        "gate_hash_matches": preregistration.get("gate_sha256") == _sha256(args.gate_path),
        "minimum_paired_prefixes": len(pairs) >= gate["acceptance"]["minimum_paired_prefixes"],
        "selected_prefixes_complete": actual_ids == expected_ids,
        "two_conditions_each": all(set(pair["conditions"]) == {"graph", "placebo"} for pair in pairs),
        "non_capsule_prefix_exact": all(pair["non_capsule_prefix_match"] for pair in pairs),
        "fixed_capsule_length": all(pair["capsule_chars"] == gate["acceptance"]["capsule_chars"] for pair in pairs),
        "api_errors_bounded": errors <= gate["acceptance"]["maximum_api_errors"],
        "generated_tools_not_executed": gate["intervention"]["execute_generated_tools"] is False,
    }
    condition_summary = {
        condition: _condition_summary(pairs, condition)
        for condition in ("graph", "placebo")
    }
    divergences = [_pair_divergence(pair) for pair in pairs]
    integrity_passed = all(checks.values())
    report = {
        "schema_version": 1,
        "stage": "7.5b",
        "mode": gate["mode"],
        "official_benchmark_result": False,
        "development_subset": True,
        "passed": integrity_passed,
        "graph_effect_attribution_eligible": integrity_passed,
        "outcome_promotion_eligible": False,
        "checks": checks,
        "paired_prefixes": len(pairs),
        "api_errors": errors,
        "condition_summary": condition_summary,
        "action_divergence_count": sum(item["action_diverged"] for item in divergences),
        "divergences": divergences,
        "artifacts": {
            str(path): _sha256(path)
            for path in (args.gate_path, args.preregistration_path, args.selection_path, args.swap_path)
        },
        "boundary": gate["boundary"],
        "interpretation": {
            "scope": "causal evidence is limited to the immediate generated action under a fixed recorded prefix",
            "execution": "generated PTC code was not executed, so this does not estimate retrieval or answer outcomes",
        },
    }
    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    args.output_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "passed": integrity_passed,
        "checks": checks,
        "paired_prefixes": len(pairs),
        "condition_summary": condition_summary,
        "action_divergence_count": report["action_divergence_count"],
    }))
    if not integrity_passed:
        raise SystemExit(1)


def _condition_summary(pairs: list[dict[str, Any]], condition: str) -> dict[str, Any]:
    actions = [pair["conditions"][condition].get("action") for pair in pairs]
    actions = [item for item in actions if isinstance(item, dict)]
    return {
        "successful_actions": len(actions),
        "tool_decisions": sum(item["kind"] == "tool" for item in actions),
        "answer_decisions": sum(item["kind"] == "answer" for item in actions),
        "static_search_sites": sum(item["static_search_sites"] for item in actions),
        "static_fetch_sites": sum(item["static_fetch_sites"] for item in actions),
        "code_chars": sum(item["code_chars"] for item in actions),
    }


def _pair_divergence(pair: dict[str, Any]) -> dict[str, Any]:
    graph = pair["conditions"]["graph"].get("action")
    placebo = pair["conditions"]["placebo"].get("action")
    action_diverged = graph != placebo
    return {
        "prefix_id": pair["prefix_id"],
        "selection_reason": pair["selection_reason"],
        "graph_capsule": pair["graph_capsule"],
        "action_diverged": action_diverged,
        "graph_action": graph,
        "placebo_action": placebo,
    }


def _json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


if __name__ == "__main__":
    main()
