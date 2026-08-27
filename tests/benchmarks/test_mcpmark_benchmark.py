from __future__ import annotations

import json
from pathlib import Path

from toolregistry import ToolRegistry

from graphptc.config import ExperimentConfig
from graphptc.mcpmark_benchmark import (
    _graph_delta_sequence,
    _paired_metrics,
    _prompt_bundle,
    _ptc_spec,
    _summarize,
    _terminal_records,
    _to_sdk_messages,
    _validate_config,
)
from graphptc.mcpmark_runtime import MCPMarkProgramRuntime, _official_server_spec
from graphptc.mcpmark_official_worker import OfficialSession


class FakeClient:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, object]]] = []
        self.closed = False

    def call_tool(self, name: str, arguments: dict[str, object]) -> dict[str, object]:
        self.calls.append((name, arguments))
        return {
            "content": [{"type": "text", "text": json.dumps(arguments)}],
            "isError": False,
        }

    def close(self) -> None:
        self.closed = True


TOOLS = [
    {
        "name": "read-record",
        "description": "Read a record.",
        "inputSchema": {
            "type": "object",
            "properties": {"record_id": {"type": "string"}},
            "required": ["record_id"],
        },
        "annotations": {"readOnlyHint": True},
    }
]


def test_runtime_calls_dynamic_wrapper_and_persists_namespace() -> None:
    client = FakeClient()
    runtime = MCPMarkProgramRuntime(client, TOOLS)  # type: ignore[arg-type]
    namespace = {function.__name__: function for function in runtime.functions}
    first = runtime.execute(
        "saved = read_record(record_id='a')\nprint(saved['isError'])",
        namespace=namespace,
    )
    second = runtime.execute("print(saved['content'][0]['text'])", namespace={})

    assert first.return_code == 0
    assert first.stdout == "False\n"
    assert second.stdout == '{"record_id": "a"}\n'
    assert client.calls == [("read-record", {"record_id": "a"})]
    assert runtime.last_execution_trace["external_actions"] == []
    assert runtime.telemetry()["mcp_calls"] == 1


def test_runtime_parameters_survive_toolregistry_ptc_projection() -> None:
    client = FakeClient()
    runtime = MCPMarkProgramRuntime(client, TOOLS)  # type: ignore[arg-type]
    registry = ToolRegistry()
    for function in runtime.functions:
        registry.register(function)
    registry.ptc.enable(runtime=runtime)

    output = registry.invoke(
        "programmatic_tool_call",
        {"code": "print(read_record(record_id='through-registry'))"},
    )

    assert "through-registry" in output
    assert client.calls == [("read-record", {"record_id": "through-registry"})]


def test_runtime_namespace_isolated_between_tasks() -> None:
    first = MCPMarkProgramRuntime(FakeClient(), TOOLS)  # type: ignore[arg-type]
    second = MCPMarkProgramRuntime(FakeClient(), TOOLS)  # type: ignore[arg-type]
    assert first.execute("saved = 7\nprint(saved)").return_code == 0
    result = second.execute("print(saved)")
    assert result.return_code == 1
    assert "NameError" in result.stderr


def test_sdk_final_message_shape_and_graph_delta_order() -> None:
    messages = [
        {"role": "user", "content": "task"},
        {
            "role": "assistant",
            "content": "working",
            "tool_calls": [
                {
                    "id": "call-1",
                    "function": {
                        "name": "programmatic_tool_call",
                        "arguments": '{"code":"print(1)"}',
                    },
                }
            ],
        },
        {"role": "tool", "tool_call_id": "call-1", "content": "1\n\nGRAPH_DELTA {}"},
        {
            "role": "assistant",
            "content": "next",
            "tool_calls": [
                {
                    "id": "call-2",
                    "function": {
                        "name": "programmatic_tool_call",
                        "arguments": '{"code":"print(2)"}',
                    },
                }
            ],
        },
        {"role": "assistant", "content": "Task completed."},
    ]
    sdk = _to_sdk_messages(messages)
    assert sdk[-1]["type"] == "message"
    assert sdk[-1]["content"][0]["text"] == "Task completed."
    sequence = _graph_delta_sequence(messages)
    assert sequence["temporal_exposure_verified"] is True
    assert sequence["causal_influence_established"] is False


def test_two_arms_freeze_all_requested_model_and_runtime_values() -> None:
    graph = ExperimentConfig.from_toml("configs/mcpmark/graphptc.toml")
    baseline = ExperimentConfig.from_toml("configs/mcpmark/fewshot-ptc.toml")
    _validate_config(graph)
    _validate_config(baseline)

    assert graph.model == baseline.model
    graph_runtime = vars(graph.runtime) | {"graph_adaptation_mode": "off"}
    assert graph_runtime == vars(baseline.runtime)
    assert graph.runtime.graph_adaptation_mode == "generic"
    assert baseline.runtime.graph_adaptation_mode == "off"
    assert graph.runtime.graph_inspection_enabled is False
    assert graph.model.max_completion_tokens == 32768
    assert graph.model.temperature == 0
    assert graph.model.thinking == "disabled"
    assert graph.runtime.max_turns == 100
    assert graph.runtime.task_timeout_seconds == 3600
    assert graph.runtime.max_compactions == 0
    assert graph.mcpmark.workers == baseline.mcpmark.workers == 1
    assert graph.mcpmark.k == baseline.mcpmark.k == 1
    assert graph.mcpmark.npx_command == baseline.mcpmark.npx_command
    assert graph.mcpmark.npm_cache_dir == baseline.mcpmark.npm_cache_dir
    assert graph.mcpmark.npm_dependency_cutoff == baseline.mcpmark.npm_dependency_cutoff
    assert graph.mcpmark.pipx_command == baseline.mcpmark.pipx_command
    assert graph.mcpmark.docker_command == baseline.mcpmark.docker_command
    assert graph.mcpmark.postgres_pip_constraints == baseline.mcpmark.postgres_pip_constraints
    assert graph.mcpmark.platform_provenance_path == baseline.mcpmark.platform_provenance_path

    graph_smoke = ExperimentConfig.from_toml("configs/mcpmark/graphptc-smoke5.toml")
    baseline_smoke = ExperimentConfig.from_toml("configs/mcpmark/fewshot-ptc-smoke5.toml")
    selection = json.loads(Path("data/mcpmark/smoke5-selection.json").read_text())
    assert graph_smoke.mcpmark.task_ids == baseline_smoke.mcpmark.task_ids
    assert graph_smoke.mcpmark.task_ids == tuple(selection["task_ids"])


def test_absolute_npx_command_adds_native_node_directory_to_path() -> None:
    command, _, env = _official_server_spec(
        "filesystem",
        {"test_directory": "/tmp/mcpmark"},
        commands={"npx": "/opt/node/bin/npx"},
        npm_cache_dir="/frozen/npm-cache",
        npm_dependency_cutoff="2026-06-12T10:57:10Z",
    )
    assert command == "/opt/node/bin/npx"
    assert env["PATH"].startswith("/opt/node/bin")
    assert env["NPM_CONFIG_CACHE"] == "/frozen/npm-cache"
    assert env["NPM_CONFIG_BEFORE"] == "2026-06-12T10:57:10Z"


def test_postgres_constraints_are_forwarded_to_official_pipx_launcher() -> None:
    _, _, env = _official_server_spec(
        "postgres",
        {
            "host": "localhost",
            "port": 5432,
            "username": "postgres",
            "password": "password",
            "database": "postgres",
        },
        postgres_pip_constraints="/frozen/postgres-constraints.txt",
    )
    assert env["PIP_CONSTRAINT"] == "/frozen/postgres-constraints.txt"


def test_fewshot_content_is_arm_invariant() -> None:
    graph = ExperimentConfig.from_toml("configs/mcpmark/graphptc.toml")
    baseline = ExperimentConfig.from_toml("configs/mcpmark/fewshot-ptc.toml")
    manifest = [
        {
            "wrapper": "read_record",
            "mcp_tool": "read-record",
            "description": "Read a record.",
            "input_schema": TOOLS[0]["inputSchema"],
        }
    ]

    assert _prompt_bundle(graph, manifest) == _prompt_bundle(baseline, manifest)


def test_graph_arm_adds_control_fields_without_inspection_api() -> None:
    graph = ExperimentConfig.from_toml("configs/mcpmark/graphptc.toml")
    baseline = ExperimentConfig.from_toml("configs/mcpmark/fewshot-ptc.toml")
    graph_properties = _ptc_spec(graph)["function"]["parameters"]["properties"]
    baseline_properties = _ptc_spec(baseline)["function"]["parameters"]["properties"]

    assert set(graph_properties) - set(baseline_properties) == {
        "action",
        "target",
        "expected_change",
    }
    assert not any("inspect" in name.lower() for name in graph_properties)


def test_summary_failure_categories_track_lifecycle_phases() -> None:
    records = [
        {"setup": None, "agent": None, "verification": None, "cleanup": {"success": True}},
        {
            "status": "finished",
            "setup": {"setup_success": True},
            "agent": {"status": "success", "usage": {}, "runtime_session": {}},
            "verification": {"result": {"success": False}},
            "cleanup": {"success": True},
        },
        {
            "status": "failed",
            "setup": {"setup_success": True},
            "agent": {"status": "success", "usage": {}, "runtime_session": {}},
            "evaluator_attempted": True,
            "verification": None,
            "cleanup": {"success": True},
        },
        {
            "status": "failed",
            "setup": {"setup_success": True},
            "agent": {"status": "success", "usage": {}, "runtime_session": {}},
            "verification": {"result": {"success": True}},
            "cleanup": {"success": False},
        },
    ]
    summary = _summarize([{}, {}, {}, {}], records, "signature")
    assert summary.setup_failures == 1
    assert summary.execution_failures == 0
    assert summary.evaluator_failures == 1
    assert summary.verifier_failures == 1
    assert summary.cleanup_failures == 1


def test_paired_metrics_separate_wins_losses_and_ties() -> None:
    passed = {"verification": {"result": {"success": True}}}
    failed = {"verification": {"result": {"success": False}}}
    metrics = _paired_metrics(
        [(passed, failed), (failed, passed), (passed, passed), (failed, failed)]
    )
    assert metrics["graph_wins"] == 1
    assert metrics["graph_losses"] == 1
    assert metrics["ties"] == 2


def test_terminal_ledger_rejects_duplicate_task_records(tmp_path: Path) -> None:
    ledger = tmp_path / "results.jsonl"
    record = {"task_id": "filesystem:file_property/example"}
    ledger.write_text(
        json.dumps(record) + "\n" + json.dumps(record) + "\n",
        encoding="utf-8",
    )

    try:
        _terminal_records(ledger)
    except ValueError as exc:
        assert "duplicate terminal" in str(exc)
    else:
        raise AssertionError("duplicate MCPMark records must be rejected")


def test_official_cleanup_is_attempted_exactly_once() -> None:
    class FailingStateManager:
        def __init__(self) -> None:
            self.calls = 0

        def clean_up(self, task: object) -> bool:
            self.calls += 1
            raise RuntimeError("cleanup failed")

    manager = FailingStateManager()
    session = OfficialSession()
    session.state_manager = manager
    session.task = object()
    session.setup_attempted = True

    first = session.cleanup()
    second = session.cleanup()

    assert first["success"] is False
    assert first["error"] == "RuntimeError: cleanup failed"
    assert second["already_attempted"] is True
    assert manager.calls == 1
