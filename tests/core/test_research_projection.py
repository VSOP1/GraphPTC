from graphptc.research_projection import project_research_graph


def test_research_projection_tracks_query_document_and_evidence_lineage() -> None:
    events = [
        {"episode_id": "e1", "type": "episode.started", "sequence": 1},
        {
            "episode_id": "e1",
            "type": "tool.called",
            "sequence": 2,
            "block_id": "b1",
            "data": {"tool": "search", "arguments": {"query": "A  B"}, "result": [{"docid": "d1"}]},
        },
        {
            "episode_id": "e1",
            "type": "tool.called",
            "sequence": 3,
            "block_id": "b1",
            "data": {"tool": "search", "arguments": {"query": "a b"}, "result": [{"docid": "d1"}]},
        },
        {
            "episode_id": "e1",
            "type": "tool.called",
            "sequence": 4,
            "block_id": "b1",
            "data": {"tool": "fetch", "arguments": {"docid": "d1"}, "result": {"content": "evidence"}},
        },
    ]
    graph = project_research_graph(events)
    assert graph["metrics"] == {
        "search_count": 2,
        "unique_queries": 1,
        "repeated_queries": 1,
        "fetch_count": 1,
        "unique_fetched_docids": 1,
        "repeated_fetches": 0,
        "unique_docids": 1,
        "repeated_result_docids": 1,
    }
    assert {node["kind"] for node in graph["nodes"]} >= {
        "BLOCK",
        "QUERY",
        "RESULT_SET",
        "DOCUMENT",
        "EVIDENCE",
    }
    node_ids = {node["id"] for node in graph["nodes"]}
    assert all(
        edge["source"] in node_ids and edge["target"] in node_ids
        for edge in graph["edges"]
    )
