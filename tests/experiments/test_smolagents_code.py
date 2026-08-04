from smolagents.default_tools import FinalAnswerTool

from graphptc.persistent_runtime import PersistentIpcRuntime
from graphptc.experiments.smolagents_code import (
    FetchTool,
    PersistentSmolExecutor,
    SearchTool,
    _keywordize_tool_calls,
)


def test_keywordize_tool_calls_uses_manifest_parameter_names() -> None:
    tools = {
        "search": SearchTool(lambda **_: []),
        "fetch": FetchTool(lambda **kwargs: kwargs),
        "final_answer": FinalAnswerTool(),
    }

    code = _keywordize_tool_calls(
        'hits = search("needle")\npage = fetch(hits[0]["docid"])\nfinal_answer(page)',
        tools,
    )

    assert "search(query='needle')" in code
    assert "fetch(docid=hits[0]['docid'])" in code
    assert "final_answer(answer=page)" in code


def test_search_prompt_exposes_array_output_schema() -> None:
    prompt = SearchTool(lambda **_: []).to_code_prompt()

    assert "-> list[dict]" in prompt
    assert '"type": "array"' in prompt
    assert '"docid"' in prompt
    assert '"snippet"' in prompt


def test_search_prompt_can_reproduce_unstructured_control() -> None:
    prompt = SearchTool(lambda **_: [], expose_output_schema=False).to_code_prompt()

    assert "-> array" in prompt
    assert '"type": "array"' not in prompt


def test_persistent_executor_runs_multiple_calls_and_preserves_state() -> None:
    calls: list[dict[str, str]] = []

    def search(*, query: str):  # type: ignore[no-untyped-def]
        calls.append({"operation": "search", "query": query})
        return [{"docid": query, "score": 1.0, "snippet": query}]

    runtime = PersistentIpcRuntime()
    executor = PersistentSmolExecutor(
        runtime=runtime,
        search_calls=calls,
        timeout_seconds=10,
        max_stdout_chars=100,
    )
    executor.send_tools(
        {
            "search": SearchTool(search),
            "fetch": FetchTool(lambda **kwargs: kwargs),
            "final_answer": FinalAnswerTool(),
        }
    )
    try:
        first = executor(
            'hits = []\nfor query in ["alpha", "beta"]:\n'
            '    hits.extend(search(query))\nprint(len(hits))'
        )
        second = executor("final_answer(str(len(hits)))")
    finally:
        runtime.close()

    assert first.logs == "2\n"
    assert first.is_final_answer is False
    assert second.output == "2"
    assert second.is_final_answer is True
    assert len(executor.blocks) == 1
    assert executor.blocks[0].runtime_calls == 2
    assert executor.blocks[0].program_analysis["has_loop"] is True


def test_persistent_executor_propagates_stdout_truncation() -> None:
    calls: list[dict[str, str]] = []
    runtime = PersistentIpcRuntime()
    executor = PersistentSmolExecutor(
        runtime=runtime,
        search_calls=calls,
        timeout_seconds=10,
        max_stdout_chars=120,
    )
    executor.send_tools({"final_answer": FinalAnswerTool()})
    try:
        output = executor('print("x" * 500)')
    finally:
        runtime.close()

    assert len(output.logs) == 120
    assert output.logs.startswith("PTC_STDOUT_TRUNCATED")
    assert executor.blocks[0].stdout_truncated is True
    assert executor.blocks[0].stdout_chars == 501
