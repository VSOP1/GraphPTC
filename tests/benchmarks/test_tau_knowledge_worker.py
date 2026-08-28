from __future__ import annotations

import json

from graphptc.tau_knowledge_worker import _runtime_metrics, _save_agent_artifacts


class _FakeAgent:
    def agent_artifact(self):
        return {
            "blocks": [
                {
                    "runtime_trace": {
                        "external_actions": [
                            {"name": "KB_search", "state_changed": False},
                            {
                                "name": "unlock_discoverable_agent_tool",
                                "state_changed": True,
                            },
                            {
                                "name": "call_discoverable_agent_tool",
                                "state_changed": True,
                            },
                        ]
                    }
                }
            ],
            "telemetry": {
                "model_requests": 2,
                "usage": {
                    "input_tokens": 100,
                    "output_tokens": 20,
                    "cached_input_tokens": 5,
                },
            },
        }

    def graph_artifact(self):
        return {"nodes": ["task"]}


def test_runtime_metrics_separate_retrieval_dynamic_tools_and_state_effects() -> None:
    metrics = _runtime_metrics(_FakeAgent(), 3.5)
    assert metrics["model_turns"] == 2
    assert metrics["ptc_blocks"] == 1
    assert metrics["tool_calls"] == 3
    assert metrics["retrieval_calls"] == 1
    assert metrics["unlock_calls"] == 1
    assert metrics["dynamic_tool_calls"] == 1
    assert metrics["state_change_calls"] == 2
    assert metrics["input_tokens"] == 100
    assert metrics["duration_seconds"] == 3.5


def test_agent_and_graph_artifacts_are_saved_for_partial_runs(tmp_path) -> None:
    agent_path = tmp_path / "trial.agent.json"
    graph_path = tmp_path / "trial.graph.json"
    _save_agent_artifacts(
        _FakeAgent(), {"agent_path": str(agent_path), "graph_path": str(graph_path)}
    )
    assert json.loads(agent_path.read_text(encoding="utf-8"))["blocks"]
    assert json.loads(graph_path.read_text(encoding="utf-8")) == {"nodes": ["task"]}
