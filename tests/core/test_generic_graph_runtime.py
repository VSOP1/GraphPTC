from __future__ import annotations

from graphptc.episode_graph import EpisodeGraph
from graphptc.graph_agent import GraphContextProjector, GraphProgressTracker
from graphptc.tool_effects import ToolEffectContract, ToolGraphRuntime


def test_generic_graph_runtime_tracks_non_retrieval_tools_and_effects() -> None:
    graph = EpisodeGraph(task="Compute a total and update inventory")
    runtime = ToolGraphRuntime(graph)
    calls = {"lookup": 0, "aggregate": 0, "update": 0}
    inventory = {"widget": 10}

    def lookup_rows(*, table: str) -> list[int]:
        calls["lookup"] += 1
        return [2, 3, 5] if table == "orders" else []

    def aggregate_values(*, values: list[int]) -> int:
        calls["aggregate"] += 1
        return sum(values)

    def update_inventory(*, item: str, delta: int) -> int:
        calls["update"] += 1
        inventory[item] += delta
        return inventory[item]

    runtime.register(
        lookup_rows,
        ToolEffectContract(
            name="lookup_rows",
            effect="read",
            deterministic=True,
            cacheable=True,
            artifact_kind="table_rows",
        ),
    )
    runtime.register(
        aggregate_values,
        ToolEffectContract(
            name="aggregate_values",
            effect="pure",
            deterministic=True,
            cacheable=True,
            artifact_kind="aggregate",
        ),
    )
    runtime.register(
        update_inventory,
        ToolEffectContract(name="update_inventory", effect="write"),
    )

    rows = runtime.invoke("lookup_rows", target="task", table="orders")
    reused_rows = runtime.invoke("lookup_rows", target="task", table="orders")
    total = runtime.invoke(
        "aggregate_values",
        target="task",
        consumes=(rows.artifact_id or "",),
        values=rows.value,
    )
    first_write = runtime.invoke(
        "update_inventory", target="task", item="widget", delta=-total.value
    )
    second_write = runtime.invoke(
        "update_inventory", target="task", item="widget", delta=total.value
    )

    assert reused_rows.reused is True
    assert calls == {"lookup": 1, "aggregate": 1, "update": 2}
    assert inventory["widget"] == 10
    assert first_write.state_after == second_write.state_before
    assert any(
        edge["type"] == "consumes"
        and edge["source"] == rows.artifact_id
        and edge["target"] == total.action_id
        for edge in graph.edges
    )
    assert any(edge["type"] == "reuses" for edge in graph.edges)
    assert any(edge["type"] == "supersedes" for edge in graph.edges)


def test_graph_context_projector_archives_only_inactive_old_observations() -> None:
    graph = EpisodeGraph(task="Use several dependent tools")
    graph.add_node("goal:active", "GOAL", {})
    active = ["goal:active"]
    projector = GraphContextProjector(
        graph,
        active_nodes=lambda: tuple(active),
        retain_recent=2,
        retain_relevant=1,
        relevance_depth=2,
    )
    messages: list[dict[str, str]] = []

    for index in range(1, 6):
        block_id = f"block:{index}"
        action_id = f"action:{index}"
        graph.add_node(block_id, "BLOCK", {})
        graph.add_node(action_id, "ACTION", {})
        graph.add_edge("implements", action_id, block_id)
        if index == 1:
            graph.add_edge("targets", action_id, "goal:active")
        graph.put_artifact(
            f"artifact:{block_id}:stdout",
            f"full observation {index}",
            kind="observation",
        )
        messages.append(
            {"role": "tool", "tool_call_id": f"call-{index}", "content": f"full {index}"}
        )
        projector.project(messages, block_id=block_id)

    assert messages[0]["content"] == "full 1"
    assert str(messages[1]["content"]).startswith("GRAPH_MEMORY_REF ")
    assert messages[-2]["content"] == "full 4"
    assert messages[-1]["content"] == "full 5"


def test_graph_progress_tracker_detects_equivalent_artifact_loop() -> None:
    graph = EpisodeGraph(task="Repeat a read")
    runtime = ToolGraphRuntime(graph)

    def read_value(*, key: str) -> dict[str, str]:
        return {"key": key, "value": "same"}

    runtime.register(read_value, ToolEffectContract(name="read_value", effect="read"))
    tracker = GraphProgressTracker(graph)
    observations = []
    for index in range(1, 4):
        call = runtime.invoke("read_value", target="task", key="x")
        block_id = f"block:{index}"
        graph.add_node(block_id, "BLOCK", {})
        graph.add_edge("executes", block_id, call.action_id)
        observations.append(tracker.observe(block_id, target="task"))

    assert observations[0]["progressed"] is True
    assert observations[1]["progressed"] is False
    assert observations[2]["stagnant_streak"] == 2
