from __future__ import annotations

import json
import subprocess
from pathlib import Path


def git_commit() -> str:
    root = repository_root()
    completed = subprocess.run(
        ("git", "rev-parse", "HEAD"),
        cwd=root,
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode == 0 and completed.stdout.strip():
        return completed.stdout.strip()
    manifest = _release_manifest(root)
    return str(manifest.get("git_commit", "unavailable"))


def git_dirty() -> bool:
    root = repository_root()
    completed = subprocess.run(
        ("git", "status", "--porcelain", "--untracked-files=all"),
        cwd=root,
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode == 0:
        return bool(completed.stdout.strip())
    return False


def repository_root() -> Path:
    for parent in Path(__file__).resolve().parents:
        if (parent / "pyproject.toml").is_file() or (parent / "RELEASE.json").is_file():
            return parent
    return Path(__file__).resolve().parents[3]


def _release_manifest(root: Path) -> dict[str, object]:
    path = root / "RELEASE.json"
    if not path.is_file():
        return {}
    value = json.loads(path.read_text(encoding="utf-8"))
    return value if isinstance(value, dict) else {}
