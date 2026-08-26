from __future__ import annotations

from graphptc.tau3_runtime import BlockComplete, Tau3ProgramRuntime, ToolRequest


def test_ptc_program_is_resumed_through_official_tool_results() -> None:
    runtime = Tau3ProgramRuntime(("lookup",), max_stdout_chars=8_000, timeout_seconds=2)
    event = runtime.start("value = lookup(key='x')\nprint(value['answer'])")

    assert isinstance(event, ToolRequest)
    assert event.name == "lookup"
    assert event.arguments == {"key": "x"}

    completed = runtime.resume('{"answer": 7}', error=False)
    assert isinstance(completed, BlockComplete)
    assert completed.success
    assert completed.stdout == "7\n"
    assert completed.calls[0]["name"] == "lookup"
    runtime.close()


def test_ptc_program_can_make_conditional_sequential_calls() -> None:
    runtime = Tau3ProgramRuntime(("read", "write"), max_stdout_chars=8_000, timeout_seconds=2)
    first = runtime.start(
        "row = read(item_id='1')\n"
        "if row['status'] == 'open':\n"
        "    changed = write(item_id='1', status='closed')\n"
        "    print(changed)"
    )
    assert isinstance(first, ToolRequest) and first.name == "read"
    second = runtime.resume('{"status":"open"}', error=False)
    assert isinstance(second, ToolRequest) and second.name == "write"
    completed = runtime.resume('{"ok":true}', error=False)
    assert isinstance(completed, BlockComplete)
    assert completed.success
    assert len(completed.calls) == 2
    runtime.close()


def test_stdout_is_truncated_without_losing_original_length() -> None:
    runtime = Tau3ProgramRuntime((), max_stdout_chars=8, timeout_seconds=2)
    completed = runtime.start("print('abcdefghijk')")
    assert isinstance(completed, BlockComplete)
    assert completed.stdout_truncated
    assert completed.stdout_chars == 12
    assert completed.stdout.startswith("abcdefgh")
    runtime.close()


def test_tool_and_program_errors_are_failures() -> None:
    runtime = Tau3ProgramRuntime(("lookup",), max_stdout_chars=8_000, timeout_seconds=2)
    request = runtime.start("lookup(key='missing')")
    assert isinstance(request, ToolRequest)
    completed = runtime.resume("not found", error=True)
    assert isinstance(completed, BlockComplete)
    assert not completed.success
    assert completed.error_type == "Tau3ToolError"
    assert completed.calls[0]["success"] is False
    runtime.close()


def test_official_database_hash_delta_marks_a_write_effect() -> None:
    runtime = Tau3ProgramRuntime(("update",), max_stdout_chars=8_000, timeout_seconds=2)
    assert isinstance(runtime.start("print(update(item_id='1'))"), ToolRequest)
    completed = runtime.resume(
        '{"ok":true}', error=False, state_changed=True, declared_effect="write"
    )
    assert isinstance(completed, BlockComplete)
    assert completed.calls[0]["effect"] == "write"
    assert completed.calls[0]["state_changed"] is True
    assert completed.calls[0]["effect_basis"] == "official_tool_metadata_and_db_hash"
    runtime.close()


def test_close_is_per_episode_and_unblocks_pending_program() -> None:
    runtime = Tau3ProgramRuntime(("lookup",), max_stdout_chars=8_000, timeout_seconds=2)
    assert isinstance(runtime.start("lookup(key='x')"), ToolRequest)
    runtime.close()
    assert runtime.closed
