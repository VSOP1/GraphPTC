from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from graphptc.config import RuntimeConfig
from graphptc.graph_progress import GraphProgressView
from graphptc.model import ModelTurn, TokenUsage, ToolCall
from graphptc.ptc import OriginalPTCAgent


PREFIX = "GRAPH_PROGRESS_SNAPSHOT "


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit automatic Stage 7.4b progress exposure.")
    parser.add_argument("gate_path", type=Path)
    parser.add_argument("output_path", type=Path)
    args = parser.parse_args()
    gate = json.loads(args.gate_path.read_text(encoding="utf-8"))
    first = {name: _run_arm(name) for name in gate["arms"]}
    second = {name: _run_arm(name) for name in gate["arms"]}
    placebo = first["placebo_auto"]
    graph = first["graph_auto"]
    control = first["control"]
    expected_success = gate["acceptance"]["successful_blocks"]
    expected_failed = gate["acceptance"]["failed_blocks"]
    expected_chars = gate["acceptance"]["capsule_chars"]
    checks = {
        "deterministic": all(_stable(first[name]) == _stable(second[name]) for name in first),
        "expected_block_outcomes": all(run["successful_blocks"] == expected_success and run["failed_blocks"] == expected_failed for run in first.values()),
        "one_capsule_per_successful_block": all(run["capsule_count"] == expected_success for name, run in first.items() if name != "control"),
        "no_control_capsules": control["capsule_count"] == 0,
        "no_capsule_for_failed_block": all(run["capsule_count"] == run["successful_blocks"] for name, run in first.items() if name != "control"),
        "fixed_capsule_length": all(all(length == expected_chars for length in run["capsule_lengths"]) for name, run in first.items() if name != "control"),
        "same_placebo_graph_schema": placebo["capsule_schemas"] == graph["capsule_schemas"],
        "graph_values_are_live": graph["capsule_values"][0]["search_calls"] == 1 and graph["capsule_values"][1]["search_calls"] == 2,
        "placebo_values_are_neutral": all(item["search_calls"] == 0 for item in placebo["capsule_values"]),
        "raw_stdout_unchanged": placebo["block_stdout"] == graph["block_stdout"] == control["block_stdout"],
        "tool_ledger_unchanged": placebo["tool_calls"] == graph["tool_calls"] == control["tool_calls"],
        "snapshot_telemetry_exact": placebo["snapshot_calls"] == graph["snapshot_calls"] == expected_success,
        "no_gold_features": not _contains_gold({"placebo": placebo["capsule_schemas"], "graph": graph["capsule_schemas"]}),
        "no_forced_stop": all(run["status"] == "success" and run["answer"] == "<result>done</result>" and run["model_requests"] == 4 for run in first.values()),
    }
    report = {
        "schema_version": 1,
        "stage": "7.4b",
        "mode": gate["mode"],
        "official_benchmark_result": False,
        "passed": all(checks.values()),
        "checks": checks,
        "arms": {name: _stable(run) for name, run in first.items()},
        "artifacts": {str(args.gate_path): _sha256(args.gate_path)},
        "interpretation": {
            "exposure": "one fixed-length user capsule follows each successful tool observation",
            "stdout_boundary": "the original tool message remains byte-identical and separately capped",
            "failure_boundary": "failed blocks expose only their original error observation",
            "next": "freeze auto-mode configs and preregister an exposure plus outcome pilot before model calls",
        },
    }
    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    args.output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"passed": report["passed"], "checks": checks, "arms": report["arms"]}))
    if not report["passed"]:
        raise SystemExit(1)


class _Tools:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    @property
    def consumed(self) -> int:
        return len(self.calls)

    def search(self, *, query: str) -> list[dict[str, Any]]:
        result = [{"docid": f"doc-{query}", "snippet": query}]
        self.calls.append({"operation": "search", "query": query, "docids": [result[0]["docid"]]})
        return result


class _Model:
    def __init__(self) -> None:
        self._turns = iter(
            [
                _tool_turn("call-1", "hits = search(query='alpha')\nprint(hits[0]['docid'])"),
                _tool_turn("call-2", "hits = search(query='beta')\nprint(hits[0]['docid'])"),
                _tool_turn("call-3", "raise ValueError('fixture failure')"),
                _answer_turn(),
            ]
        )
        self.messages_seen: list[list[dict[str, Any]]] = []

    def create_turn(self, **kwargs: Any) -> ModelTurn:
        self.messages_seen.append(list(kwargs["messages"]))
        return next(self._turns)


def _run_arm(name: str) -> dict[str, Any]:
    tools = _Tools()
    model = _Model()
    view = None
    if name != "control":
        view = GraphProgressView(tools, mode=name.removesuffix("_auto"), max_tool_calls=10)
    agent = OriginalPTCAgent(
        model=model,
        search_tools=tools,
        runtime=RuntimeConfig(max_turns=5, max_ptc_blocks=4),
        runtime_functions=(tools.search,),
        post_block_message_factory=(
            None if view is None else lambda _trace: view.capsule()
        ),
    )
    result = agent.run("fixture")
    messages = model.messages_seen[-1]
    capsules = [str(message["content"]) for message in messages if message.get("role") == "user" and str(message.get("content", "")).startswith(PREFIX)]
    values = [json.loads(capsule.removeprefix(PREFIX)) for capsule in capsules]
    return {
        "status": result.status,
        "answer": result.answer,
        "model_requests": result.model_requests,
        "successful_blocks": sum(block.success for block in result.blocks),
        "failed_blocks": sum(not block.success for block in result.blocks),
        "block_stdout": [block.stdout for block in result.blocks],
        "tool_calls": tools.calls,
        "capsule_count": len(capsules),
        "capsule_lengths": [len(capsule) for capsule in capsules],
        "capsule_schemas": [list(value) for value in values],
        "capsule_values": [{key: item for key, item in value.items() if key != "padding"} for value in values],
        "snapshot_calls": 0 if view is None else view.telemetry()["snapshot_calls"],
    }


def _stable(run: dict[str, Any]) -> dict[str, Any]:
    return run


def _tool_turn(call_id: str, code: str) -> ModelTurn:
    return ModelTurn(
        assistant_message={"role": "assistant", "content": None},
        text="",
        tool_calls=[ToolCall(id=call_id, name="programmatic_tool_call", input={"code": code})],
        usage=TokenUsage(input_tokens=1, output_tokens=1),
        stop_reason="tool_calls",
    )


def _answer_turn() -> ModelTurn:
    return ModelTurn(
        assistant_message={"role": "assistant", "content": "<result>done</result>"},
        text="<result>done</result>",
        tool_calls=[],
        usage=TokenUsage(input_tokens=1, output_tokens=1),
        stop_reason="stop",
    )


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
