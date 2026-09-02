#!/usr/bin/env python3
"""Build a secret-free source archive and a cloneable Git bundle."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import zipfile
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Iterable, Sequence


REPO_ROOT = Path(__file__).resolve().parents[2]
RESULT_PATHS = (
    "runs/agent_diff/graphptc",
    "runs/agent_diff/fewshot-ptc",
    "runs/alfworld/valid-seen",
    "runs/alfworld/valid-unseen",
    "runs/apiflow/graphptc",
    "runs/apiflow/fewshot-ptc",
    "runs/apiflow/temperature1-epoch1",
    "runs/appworld/graphptc-test-normal",
    "runs/appworld/fewshot-ptc-test-normal",
    "runs/appworld/graphptc-test-challenge",
    "runs/appworld/fewshot-ptc-test-challenge",
    "runs/browsecomp_plus/graphptc-stdout8k-fold1",
    "runs/browsecomp_plus/graphptc-stdout8k-fold2",
    "runs/browsecomp_plus/graphptc-stdout8k-fold3",
    "runs/browsecomp_plus/graphptc-stdout8k-remaining530",
    "runs/browsecomp_plus/fewshot-ptc-v1-stdout8k-fold1",
    "runs/browsecomp_plus/fewshot-ptc-v1-stdout8k-fold2",
    "runs/browsecomp_plus/fewshot-ptc-v1-stdout8k-fold3",
    "runs/browsecomp_plus/fewshot-ptc-v1-stdout8k-remaining530",
    "runs/fanoutqa/dev/graphptc",
    "runs/fanoutqa/dev/fewshot-ptc",
    "runs/frames/test/graphptc",
    "runs/frames/test/fewshot-ptc",
    "runs/intercode/graphptc",
    "runs/intercode/baseline",
    "runs/toolhop/mandatory-temperature0-epoch1",
    "runs/toolsandbox/graphptc",
    "runs/toolsandbox/fewshot-ptc",
)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("dist"))
    parser.add_argument(
        "--include-results",
        action="store_true",
        help="Also create a multi-gigabyte archive containing only runs/README.md allowlisted results.",
    )
    args = parser.parse_args(argv)
    output_dir = args.output_dir
    if not output_dir.is_absolute():
        output_dir = REPO_ROOT / output_dir

    _require_clean_repository()
    commit = _git("rev-parse", "HEAD").strip()
    label = commit[:12]
    source_zip = output_dir / f"GraphPTC-{label}-source.zip"
    bundle = output_dir / f"GraphPTC-{label}.bundle"
    results_zip = output_dir / f"GraphPTC-{label}-results.zip"
    checksum_path = output_dir / f"GraphPTC-{label}-SHA256SUMS.txt"
    targets = [source_zip, bundle, checksum_path]
    if args.include_results:
        targets.append(results_zip)
    existing = [str(path) for path in targets if path.exists()]
    if existing:
        raise FileExistsError(
            "refusing to overwrite package files: " + ", ".join(existing)
        )
    output_dir.mkdir(parents=True, exist_ok=True)

    tracked = _tracked_files()
    unsafe = [path for path in tracked if not _is_safe_source_path(path)]
    if unsafe:
        raise ValueError(
            "unsafe tracked paths would enter the package: " + ", ".join(unsafe)
        )
    release = {
        "schema_version": 1,
        "git_commit": commit,
        "git_dirty": False,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "tracked_files": len(tracked),
    }
    with zipfile.ZipFile(
        source_zip, "x", compression=zipfile.ZIP_DEFLATED, allowZip64=True
    ) as archive:
        for relative in tracked:
            archive.write(
                REPO_ROOT / relative, str(PurePosixPath("GraphPTC") / relative)
            )
        archive.writestr(
            "GraphPTC/RELEASE.json",
            json.dumps(release, ensure_ascii=False, indent=2) + "\n",
        )

    subprocess.run(
        ("git", "bundle", "create", str(bundle), "HEAD", "--branches", "--tags"),
        cwd=REPO_ROOT,
        check=True,
    )
    created = [source_zip, bundle]
    if args.include_results:
        _write_results_archive(results_zip)
        created.append(results_zip)
    checksum_path.write_text(
        "".join(f"{_sha256(path)}  {path.name}\n" for path in created),
        encoding="utf-8",
    )
    for path in (*created, checksum_path):
        print(path)
    return 0


def _require_clean_repository() -> None:
    status = _git("status", "--porcelain", "--untracked-files=all")
    if status.strip():
        raise RuntimeError(
            "repository is not clean; commit the reviewed delivery changes first"
        )


def _tracked_files() -> list[str]:
    output = subprocess.run(
        ("git", "ls-files", "-z"),
        cwd=REPO_ROOT,
        capture_output=True,
        check=True,
    ).stdout
    return sorted(part.decode("utf-8") for part in output.split(b"\0") if part)


def _is_safe_source_path(value: str) -> bool:
    path = PurePosixPath(value)
    parts = set(path.parts)
    if path.name in {".env", ".mcp_env"}:
        return False
    if parts & {
        ".git",
        ".venv",
        "__pycache__",
        ".pytest_cache",
        ".ruff_cache",
        "dist",
        "external",
    }:
        return False
    if path.parts and path.parts[0] == "runs" and value != "runs/README.md":
        return False
    return True


def _write_results_archive(path: Path) -> None:
    roots = [
        REPO_ROOT / "runs" / "README.md",
        *(REPO_ROOT / item for item in RESULT_PATHS),
    ]
    missing = [str(item) for item in roots if not item.exists()]
    if missing:
        raise FileNotFoundError(
            "allowlisted result paths are missing: " + ", ".join(missing)
        )
    with zipfile.ZipFile(
        path, "x", compression=zipfile.ZIP_DEFLATED, allowZip64=True
    ) as archive:
        for file_path in _result_files(roots):
            relative = file_path.relative_to(REPO_ROOT)
            archive.write(
                file_path, str(PurePosixPath("GraphPTC") / relative.as_posix())
            )


def _result_files(roots: Iterable[Path]) -> list[Path]:
    files: set[Path] = set()
    for root in roots:
        if root.is_file():
            files.add(root)
        else:
            files.update(path for path in root.rglob("*") if path.is_file())
    return sorted(files)


def _git(*args: str) -> str:
    return subprocess.run(
        ("git", *args),
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=True,
    ).stdout


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FileExistsError, FileNotFoundError, RuntimeError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc
