from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from graphptc.progress_consumption import load_jsonl, project_capsule_consumption


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit next-action consumption of Stage 7.4c capsules.")
    parser.add_argument("config_path", type=Path)
    parser.add_argument("output_path", type=Path)
    args = parser.parse_args()
    config = _json(args.config_path)

    arms: dict[str, dict[str, Any]] = {}
    hash_checks: dict[str, bool] = {}
    for name, inputs in config["inputs"].items():
        events_path = Path(inputs["events_path"])
        responses_path = Path(inputs["responses_path"])
        hash_checks[f"{name}_events"] = _sha256(events_path) == inputs["events_sha256"]
        hash_checks[f"{name}_responses"] = _sha256(responses_path) == inputs["responses_sha256"]
        projection = project_capsule_consumption(
            load_jsonl(events_path), max_tool_calls=1000
        )
        responses = _jsonl(responses_path)
        expected_snapshots = sum(
            int((item.get("graph_progress") or {}).get("snapshot_calls", 0))
            for item in responses
        )
        projection["expected_snapshot_calls"] = expected_snapshots
        projection["snapshot_count_matches"] = projection["successful_blocks"] == expected_snapshots
        projection["all_successful_blocks_linked"] = projection["transition_count"] == projection["successful_blocks"]
        arms[name] = projection

    acceptance = config["acceptance"]
    exact_prefix_available = all(
        any(Path(inputs["responses_path"]).parent.joinpath("checkpoints").glob("*.json"))
        for inputs in config["inputs"].values()
    )
    checks = {
        "input_hash_match": all(hash_checks.values()),
        "expected_episodes_each": all(
            arm["episode_count"] == acceptance["expected_episodes_per_arm"]
            for arm in arms.values()
        ),
        "snapshot_count_match": all(arm["snapshot_count_matches"] for arm in arms.values()),
        "all_successful_blocks_linked": all(
            arm["all_successful_blocks_linked"] for arm in arms.values()
        ),
        "exact_model_prefix_available": exact_prefix_available,
    }
    diagnostic_complete = all(
        checks[name]
        for name in (
            "input_hash_match",
            "expected_episodes_each",
            "snapshot_count_match",
            "all_successful_blocks_linked",
        )
    )
    report = {
        "schema_version": 1,
        "stage": "7.5a",
        "mode": config["mode"],
        "official_benchmark_result": False,
        "development_subset": True,
        "diagnostic_complete": diagnostic_complete,
        "stage7_5b_ready": diagnostic_complete and checks["exact_model_prefix_available"],
        "checks": checks,
        "hash_checks": hash_checks,
        "arms": arms,
        "comparison": _comparison(arms["placebo_auto"], arms["graph_auto"]),
        "boundary": config["boundary"],
        "interpretation": {
            "causal": "next-action associations do not prove that a capsule field caused an action",
            "prefix": "completed checkpoints are absent, so the current runs cannot support an exact fixed-prefix capsule swap",
        },
    }
    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    args.output_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({
        "diagnostic_complete": diagnostic_complete,
        "stage7_5b_ready": report["stage7_5b_ready"],
        "checks": checks,
        "comparison": report["comparison"],
    }))
    if not diagnostic_complete:
        raise SystemExit(1)


def _comparison(placebo: dict[str, Any], graph: dict[str, Any]) -> dict[str, Any]:
    left = placebo["summary"]["all"]
    right = graph["summary"]["all"]
    keys = (
        "terminal_rate",
        "mean_next_search_calls",
        "mean_next_fetch_calls",
        "next_exact_repeat_search_rate",
        "next_zero_novelty_search_rate",
        "next_repeated_fetch_rate",
        "next_known_unfetched_fetch_rate",
    )
    return {
        "placebo_auto": left,
        "graph_auto": right,
        "graph_minus_placebo": {
            key: _difference(left[key], right[key]) for key in keys
        },
        "graph_signal_slices": graph["summary"]["signals"],
    }


def _difference(left: float | None, right: float | None) -> float | None:
    return right - left if left is not None and right is not None else None


def _json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


if __name__ == "__main__":
    main()
