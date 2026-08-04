from __future__ import annotations

import json
from typing import Any


def _tool_call(call_id: str, code: str, text: str) -> dict[str, Any]:
    return {
        "role": "assistant",
        "content": text,
        "tool_calls": [
            {
                "id": call_id,
                "type": "function",
                "function": {
                    "name": "programmatic_tool_call",
                    "arguments": json.dumps({"code": code}),
                },
            }
        ],
    }


_BATCH_FILTER = [
    {
        "role": "user",
        "content": """Answer the following question using the research environment when evidence is needed.

<question>Which of the Alder, Bracken, and Cobalt depots passed inspection and stored coolant K7?</question>

Return the final concise answer inside <result> and </result> tags; separate multiple answers with commas.""",
    },
    _tool_call(
        "demo_batch_filter",
        """import json
depots = ["Alder", "Bracken", "Cobalt"]
hits = {}
for depot in depots:
    for hit in search(query=f"{depot} depot inspection coolant"):
        hits.setdefault(hit["docid"], hit)
records = []
for hit in hits.values():
    page = fetch(docid=hit["docid"])
    text = page["content"].lower()
    if "inspection passed" in text and "coolant k7" in text:
        records.append({"docid": hit["docid"], "evidence": page["content"][-240:]})
print(json.dumps(records, ensure_ascii=False))""",
        "I will inspect the named candidates together and return only records satisfying both conditions.",
    ),
    {
        "role": "tool",
        "tool_call_id": "demo_batch_filter",
        "content": '[{"docid":"depot-bracken","evidence":"Bracken depot. Coolant K7. Inspection passed."}]',
    },
    {"role": "assistant", "content": "<result>Bracken</result>"},
]


_JOIN_AND_AGGREGATE = [
    {
        "role": "user",
        "content": """Answer the following question using the research environment when evidence is needed.

<question>What was the combined verified output of the Larch and Rowan stations?</question>

Return the final concise answer inside <result> and </result> tags; separate multiple answers with commas.""",
    },
    _tool_call(
        "demo_join_sum",
        """import json, re
values = {}
for station in ["Larch", "Rowan"]:
    station_hits = search(query=f"{station} station verified output")
    for hit in station_hits[:3]:
        page = fetch(docid=hit["docid"])
        match = re.search(r"verified output:\\s*(\\d+)", page["content"], re.I)
        if match and station.lower() in page["content"].lower():
            values[station] = int(match.group(1))
            break
print(json.dumps({"values": values, "total": sum(values.values())}, ensure_ascii=False))""",
        "I will retrieve both station records and perform the deterministic extraction and sum in Python.",
    ),
    {
        "role": "tool",
        "tool_call_id": "demo_join_sum",
        "content": '{"values":{"Larch":18,"Rowan":24},"total":42}',
    },
    {"role": "assistant", "content": "<result>42</result>"},
]


PTC_FEW_SHOT_MESSAGES: tuple[dict[str, Any], ...] = tuple(
    _BATCH_FILTER + _JOIN_AND_AGGREGATE
)
