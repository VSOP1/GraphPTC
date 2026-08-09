from __future__ import annotations

from typing import Any, Iterable

from graphptc.actionable_frontier import project_actionable_frontier
from graphptc.graph_adaptation import project_shadow_adaptation


def project_actionable_frontier_r2(
    events: Iterable[dict[str, Any]], *, max_items: int
) -> dict[str, Any]:
    values = list(events)
    r1 = project_actionable_frontier(iter(values), max_items=max_items)
    shadow = project_shadow_adaptation(iter(values), max_frontier_items=max_items)
    r1_by_source = {
        item["source_block_id"]: item for item in r1["opportunities"]
    }
    opportunities = []
    for proposal in shadow["proposals"]:
        r1_item = r1_by_source.get(proposal["source_block_id"])
        if r1_item is None or not proposal["frontier"]:
            continue
        opportunities.append(
            {
                "task_id": proposal["task_id"],
                "source_block_id": proposal["source_block_id"],
                "source_turn": proposal["source_turn"],
                "trigger_reasons": list(proposal["trigger"]["reasons"]),
                "frontiers": {
                    "lineage_recency": proposal["frontier"],
                    "r1_graph": r1_item["frontiers"]["graph"],
                    "recency": r1_item["frontiers"]["recency"],
                    "first_seen": r1_item["frontiers"]["first_seen"],
                },
                "next_action": r1_item["next_action"],
            }
        )
    return {
        "episode_count": shadow["episode_count"],
        "successful_blocks": shadow["successful_blocks"],
        "triggered_blocks": shadow["triggered_blocks"],
        "trigger_rate": shadow["trigger_rate"],
        "actionable_opportunities": len(opportunities),
        "opportunities": opportunities,
    }
