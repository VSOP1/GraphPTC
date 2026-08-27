from __future__ import annotations

import json
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

from dotenv import dotenv_values


def _get(url: str, headers: dict[str, str]) -> dict[str, Any]:
    request = urllib.request.Request(url, headers=headers)
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            payload = json.load(response)
            return {
                "status": response.status,
                "login": payload.get("login"),
                "state": payload.get("state"),
                "role": payload.get("role"),
                "oauth_scopes": response.headers.get("X-OAuth-Scopes", ""),
            }
    except urllib.error.HTTPError as exc:
        try:
            message = json.load(exc).get("message")
        except Exception:
            message = None
        return {"status": exc.code, "message": message}


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    env = dotenv_values(root / ".mcp_env")
    token = str(env.get("GITHUB_TOKENS") or "").split(",", 1)[0].strip()
    organization = str(env.get("GITHUB_EVAL_ORG") or "").strip()
    if not token or not organization:
        raise SystemExit("GITHUB_TOKENS and GITHUB_EVAL_ORG must be configured")
    headers = {
        "Authorization": f"Bearer {token}",
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
        "User-Agent": "graphptc-mcpmark-auth-audit",
    }
    result = {
        "user": _get("https://api.github.com/user", headers),
        "organization": _get(
            f"https://api.github.com/orgs/{organization}", headers
        ),
        "membership": _get(
            f"https://api.github.com/user/memberships/orgs/{organization}", headers
        ),
    }
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0 if all(item["status"] == 200 for item in result.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
