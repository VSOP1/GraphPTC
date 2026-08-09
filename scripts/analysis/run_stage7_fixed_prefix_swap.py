from __future__ import annotations

import argparse
import ast
import copy
import gzip
import hashlib
import json
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from dotenv import load_dotenv

from graphptc.browsecomp_plus_benchmark import _prompt_pair, _ptc_tool_spec
from graphptc.config import ExperimentConfig
from graphptc.graph_progress import GraphProgressView
from graphptc.model import OpenAIChatModel


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Stage 7.5b fixed-prefix capsule swaps.")
    parser.add_argument("config_path", type=Path)
    parser.add_argument("gate_path", type=Path)
    parser.add_argument("selection_path", type=Path)
    parser.add_argument("output_path", type=Path)
    args = parser.parse_args()
    load_dotenv(".env")
    config = ExperimentConfig.from_toml(args.config_path)
    gate = _json(args.gate_path)
    selection = _json(args.selection_path)
    model = OpenAIChatModel(
        config.model, config.require_api_key(config.model.api_key_env)
    )
    system_prompt, _ = _prompt_pair(config)
    tool_spec = _ptc_tool_spec(config)
    if tool_spec is None:
        raise ValueError("Stage 7.5b requires the PTC tool")
    placebo_capsule = GraphProgressView(
        SimpleNamespace(calls=[], consumed=0),
        mode="placebo",
        max_tool_calls=config.browsecomp_plus.max_tool_calls,
        target_chars=gate["acceptance"]["capsule_chars"],
    ).capsule()
    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    args.output_path.write_text("", encoding="utf-8")

    for prefix in selection["prefixes"]:
        archive_path = Path(prefix["archive_path"])
        if _sha256(archive_path) != prefix["archive_sha256"]:
            raise ValueError(f"archive hash mismatch: {archive_path}")
        payload = _gzip_json(archive_path)
        graph_messages = payload["messages"]
        index = int(prefix["capsule_message_index"])
        graph_capsule = str(graph_messages[index]["content"])
        if len(graph_capsule) != len(placebo_capsule):
            raise ValueError("graph/placebo capsule length mismatch")
        placebo_messages = copy.deepcopy(graph_messages)
        placebo_messages[index]["content"] = placebo_capsule
        order = (
            ["graph", "placebo"]
            if int(hashlib.sha256(prefix["prefix_id"].encode()).hexdigest(), 16) % 2 == 0
            else ["placebo", "graph"]
        )
        pair = {
            "schema_version": 1,
            "stage": "7.5b",
            "prefix_id": prefix["prefix_id"],
            "example_id": prefix["example_id"],
            "next_turn": prefix["next_turn"],
            "selection_reason": prefix["selection_reason"],
            "archive_path": str(archive_path),
            "archive_sha256": prefix["archive_sha256"],
            "capsule_message_index": index,
            "capsule_chars": len(graph_capsule),
            "graph_capsule_sha256": hashlib.sha256(graph_capsule.encode()).hexdigest(),
            "placebo_capsule_sha256": hashlib.sha256(placebo_capsule.encode()).hexdigest(),
            "graph_capsule": prefix["capsule"],
            "non_capsule_prefix_match": _without_capsule(graph_messages, index)
            == _without_capsule(placebo_messages, index),
            "condition_order": order,
            "conditions": {},
        }
        for condition in order:
            messages = graph_messages if condition == "graph" else placebo_messages
            try:
                turn = model.create_turn(
                    system=system_prompt,
                    messages=messages,
                    tools=[tool_spec],
                    timeout_seconds=config.model.timeout_seconds,
                )
                pair["conditions"][condition] = {
                    "status": "success",
                    "assistant_message": turn.assistant_message,
                    "text": turn.text,
                    "stop_reason": turn.stop_reason,
                    "usage": asdict(turn.usage),
                    "attempts": [asdict(item) for item in turn.attempts],
                    "action": _action(turn.tool_calls),
                }
            except Exception as exc:
                pair["conditions"][condition] = {
                    "status": "error",
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                }
        with args.output_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(pair, ensure_ascii=False) + "\n")
        print(json.dumps({
            "prefix_id": prefix["prefix_id"],
            "completed": len(pair["conditions"]),
        }))


def _action(tool_calls: list[Any]) -> dict[str, Any]:
    codes = [
        str(call.input.get("code", ""))
        for call in tool_calls
        if call.name == "programmatic_tool_call"
    ]
    search_sites = 0
    fetch_sites = 0
    syntax_valid = True
    for code in codes:
        try:
            tree = ast.parse(code)
        except SyntaxError:
            syntax_valid = False
            continue
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Name):
                continue
            search_sites += node.func.id == "search"
            fetch_sites += node.func.id == "fetch"
    return {
        "kind": "tool" if tool_calls else "answer",
        "tool_calls": len(tool_calls),
        "code_sha256": [hashlib.sha256(code.encode()).hexdigest() for code in codes],
        "code_chars": sum(len(code) for code in codes),
        "syntax_valid": syntax_valid,
        "static_search_sites": search_sites,
        "static_fetch_sites": fetch_sites,
    }


def _without_capsule(messages: list[dict[str, Any]], index: int) -> str:
    value = copy.deepcopy(messages)
    value[index]["content"] = "<CAPSULE>"
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _gzip_json(path: Path) -> dict[str, Any]:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        return json.load(handle)


def _json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


if __name__ == "__main__":
    main()
