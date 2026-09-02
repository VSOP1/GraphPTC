from __future__ import annotations

import json
from typing import Any


def _tool_call(
    call_id: str,
    *,
    code: str,
    expected_change: str,
    text: str,
) -> dict[str, Any]:
    return {
        "role": "assistant",
        "content": text,
        "tool_calls": [
            {
                "id": call_id,
                "type": "function",
                "function": {
                    "name": "programmatic_tool_call",
                    "arguments": json.dumps(
                        {
                            "code": code,
                            "action": "CONTINUE",
                            "target": "task",
                            "expected_change": expected_change,
                        }
                    ),
                },
            }
        ],
    }


APPWORLD_PTC_FEW_SHOT_MESSAGES: tuple[dict[str, Any], ...] = (
    {
        "role": "user",
        "content": """Demonstration task: How many playlists are in my Spotify playlist library?

The people, credentials, tokens, outputs, and answer below are demonstration-only values. For a real
task, retrieve every value from that task's AppWorld APIs.""",
    },
    _tool_call(
        "appworld_demo_docs",
        code="""import json
descriptions = apis.api_docs.show_api_descriptions(app_name="spotify")
available = {item["name"] for item in descriptions}
wanted = ["login", "show_playlist_library"]
docs = {}
for api_name in wanted:
    if api_name in available:
        spec = apis.api_docs.show_api_doc(app_name="spotify", api_name=api_name)
        docs[api_name] = {
            "parameters": [
                {"name": item["name"], "required": item["required"]}
                for item in spec["parameters"]
            ]
        }
print(json.dumps(docs, ensure_ascii=False))""",
        expected_change="identify the documented login and playlist-library parameters",
        text=(
            "I will inspect the relevant API names and parameter contracts together, then print "
            "only the compact fields needed for the execution phase."
        ),
    ),
    {
        "role": "tool",
        "tool_call_id": "appworld_demo_docs",
        "content": (
            '{"login":{"parameters":[{"name":"username","required":true},'
            '{"name":"password","required":true}]},'
            '"show_playlist_library":{"parameters":['
            '{"name":"access_token","required":true},'
            '{"name":"page_index","required":false},'
            '{"name":"page_limit","required":false}]}}\n\n'
            'GRAPH_DELTA {"declared_action":{"action":"CONTINUE","target":"task"},'
            '"action_verification":{"realized":true}}'
        ),
    },
    _tool_call(
        "appworld_demo_execute",
        code="""profile = apis.supervisor.show_profile()
passwords = {
    item["account_name"]: item["password"]
    for item in apis.supervisor.show_account_passwords()
}
login = apis.spotify.login(
    username=profile["email"],
    password=passwords["spotify"],
)
access_token = login["access_token"]
playlists = []
for page_index in range(100):
    page = apis.spotify.show_playlist_library(
        access_token=access_token,
        page_index=page_index,
        page_limit=20,
    )
    if not page:
        break
    playlists.extend(page)
answer = len(playlists)
print({"playlist_count": answer})
apis.supervisor.complete_task(answer=answer)""",
        expected_change="read every playlist page and submit the derived count",
        text=(
            "The remaining calls and count are mechanically determined, so I will obtain the "
            "credentials, pass the returned token explicitly, paginate, aggregate, and complete "
            "the task in one program."
        ),
    ),
    {
        "role": "tool",
        "tool_call_id": "appworld_demo_execute",
        "content": '{"playlist_count":7}\nExecution successful.',
    },
)
