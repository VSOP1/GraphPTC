#!/usr/bin/env python3
"""Create a model profile and orchestrate the complete three-arm evaluation suite."""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import tomllib
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Sequence


REPO_ROOT = Path(__file__).resolve().parents[2]
PROFILE_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
OUTPUT_KEYS = {
    "responses_path",
    "results_path",
    "grades_path",
    "report_path",
    "submission_path",
    "progress_path",
    "artifact_dir",
    "graph_dir",
    "wiki_cache_dir",
}
OUTPUT_DIRECTORIES = {
    "artifact_dir": "artifacts",
    "graph_dir": "graphs",
    "wiki_cache_dir": "wiki-cache",
}


@dataclass(frozen=True)
class Arm:
    benchmark: str
    name: str
    config: str
    run_command: str
    evaluate_command: str


ARMS = (
    Arm("browsecomp_plus", "graphptc", "browsecomp_plus.graphptc-full.toml", "run-browsecomp-plus", "evaluate-browsecomp-plus"),
    Arm("browsecomp_plus", "fewshot-ptc", "browsecomp_plus.fewshot-ptc-full.toml", "run-browsecomp-plus", "evaluate-browsecomp-plus"),
    Arm("browsecomp_plus", "direct-tools", "browsecomp_plus.direct-tools-full.toml", "run-browsecomp-plus", "evaluate-browsecomp-plus"),
    Arm("appworld", "graphptc-test-normal", "appworld.graphptc-test-normal.toml", "run-appworld", "evaluate-appworld"),
    Arm("appworld", "fewshot-ptc-test-normal", "appworld.fewshot-ptc-test-normal.toml", "run-appworld", "evaluate-appworld"),
    Arm("appworld", "direct-tools-test-normal", "appworld.direct-tools-test-normal.toml", "run-appworld", "evaluate-appworld"),
    Arm("appworld", "graphptc-test-challenge", "appworld.graphptc-test-challenge.toml", "run-appworld", "evaluate-appworld"),
    Arm("appworld", "fewshot-ptc-test-challenge", "appworld.fewshot-ptc-test-challenge.toml", "run-appworld", "evaluate-appworld"),
    Arm("appworld", "direct-tools-test-challenge", "appworld.direct-tools-test-challenge.toml", "run-appworld", "evaluate-appworld"),
    Arm("toolsandbox", "graphptc", "graphptc.toml", "run-toolsandbox", "evaluate-toolsandbox"),
    Arm("toolsandbox", "fewshot-ptc", "fewshot-ptc.toml", "run-toolsandbox", "evaluate-toolsandbox"),
    Arm("toolsandbox", "direct-tools", "direct-tools.toml", "run-toolsandbox", "evaluate-toolsandbox"),
    Arm("agent_diff", "graphptc", "graphptc.toml", "run-agent-diff", "evaluate-agent-diff"),
    Arm("agent_diff", "fewshot-ptc", "fewshot-ptc.toml", "run-agent-diff", "evaluate-agent-diff"),
    Arm("agent_diff", "direct-tools", "direct-tools.toml", "run-agent-diff", "evaluate-agent-diff"),
    Arm("fanoutqa", "graphptc", "graphptc-dev.toml", "run-fanoutqa", "evaluate-fanoutqa"),
    Arm("fanoutqa", "fewshot-ptc", "fewshot-ptc-dev.toml", "run-fanoutqa", "evaluate-fanoutqa"),
    Arm("fanoutqa", "direct-tools", "direct-tools-dev.toml", "run-fanoutqa", "evaluate-fanoutqa"),
    Arm("frames", "graphptc", "graphptc-test.toml", "run-frames", "evaluate-frames"),
    Arm("frames", "fewshot-ptc", "fewshot-ptc-test.toml", "run-frames", "evaluate-frames"),
    Arm("frames", "direct-tools", "direct-tools-test.toml", "run-frames", "evaluate-frames"),
)


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    os.chdir(REPO_ROOT)
    if args.command == "create-profile":
        profile_dir = create_profile(
            args.profile,
            model=args.model,
            base_url=args.base_url,
            api_key_env=args.api_key_env,
            thinking=args.thinking,
        )
        print(profile_dir)
        return 0

    profile_dir = _profile_dir(args.profile)
    _load_profile(profile_dir)
    if args.command == "preflight":
        return _preflight(profile_dir, dry_run=args.dry_run)
    if args.command == "run":
        return _execute_arms(profile_dir, phase="run", dry_run=args.dry_run)
    if args.command == "evaluate":
        return _execute_arms(profile_dir, phase="evaluate", dry_run=args.dry_run)
    if args.command == "all":
        for phase in ("preflight", "run", "evaluate"):
            result = (
                _preflight(profile_dir, dry_run=args.dry_run)
                if phase == "preflight"
                else _execute_arms(profile_dir, phase=phase, dry_run=args.dry_run)
            )
            if result:
                return result
        return 0
    parser.error(f"unsupported command: {args.command}")
    return 2


def create_profile(
    profile: str,
    *,
    model: str,
    base_url: str,
    api_key_env: str,
    thinking: str | None,
) -> Path:
    profile_dir = _profile_dir(profile)
    profile_root = profile_dir.parent
    if profile_root.exists():
        raise FileExistsError(
            f"profile already exists: {profile_root}; choose a new profile name"
        )
    for arm in ARMS:
        source = REPO_ROOT / "configs" / arm.benchmark / arm.config
        target = profile_dir / arm.benchmark / arm.config
        target.parent.mkdir(parents=True, exist_ok=True)
        output_dir = PurePosixPath("runs") / arm.benchmark / profile / arm.name
        rewritten = _rewrite_config(
            source.read_text(encoding="utf-8"),
            model=model,
            base_url=base_url,
            api_key_env=api_key_env,
            thinking=thinking,
            output_dir=output_dir,
            appworld_experiment=f"{profile}-{arm.name}",
        )
        tomllib.loads(rewritten)
        target.write_text(rewritten, encoding="utf-8", newline="\n")

    manifest = {
        "schema_version": 1,
        "profile": profile,
        "model": model,
        "base_url": base_url,
        "api_key_env": api_key_env,
        "thinking": thinking,
        "configs": [str(_arm_config(profile_dir, arm).relative_to(REPO_ROOT)).replace("\\", "/") for arm in ARMS],
    }
    (profile_root / "profile.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    _validate_profile(profile_dir, profile)
    return profile_dir


def _rewrite_config(
    text: str,
    *,
    model: str,
    base_url: str,
    api_key_env: str,
    thinking: str | None,
    output_dir: PurePosixPath,
    appworld_experiment: str,
) -> str:
    section = ""
    output: list[str] = []
    assignment = re.compile(r"^(\s*)([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(.+?)\s*$")
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("[") and stripped.endswith("]"):
            section = stripped[1:-1]
            output.append(line)
            continue
        match = assignment.match(line)
        if not match:
            output.append(line)
            continue
        indent, key, raw_value = match.groups()
        if section == "model" and key in {"model", "base_url", "api_key_env", "thinking"}:
            replacements = {
                "model": model,
                "base_url": base_url,
                "api_key_env": api_key_env,
                "thinking": thinking,
            }
            value = replacements[key]
            if value is not None:
                output.append(f"{indent}{key} = {_toml_string(value)}")
            continue
        if section == "appworld" and key == "experiment_name":
            output.append(f"{indent}{key} = {_toml_string(appworld_experiment)}")
            continue
        if key in OUTPUT_KEYS:
            old_value = tomllib.loads(f"value = {raw_value}")["value"]
            suffix = OUTPUT_DIRECTORIES.get(key, PurePosixPath(old_value).name)
            output.append(f"{indent}{key} = {_toml_string(str(output_dir / suffix))}")
            continue
        output.append(line)
    return "\n".join(output) + "\n"


def _preflight(profile_dir: Path, *, dry_run: bool) -> int:
    _validate_profile(profile_dir, profile_dir.parent.name)
    missing = _missing_environment(profile_dir)
    if missing and not dry_run:
        print("missing environment variables: " + ", ".join(missing), file=sys.stderr)
        return 2
    commands = [
        _command("inspect-browsecomp-plus", profile_dir / "browsecomp_plus" / "browsecomp_plus.graphptc-full.toml"),
        _command("inspect-appworld", profile_dir / "appworld" / "appworld.graphptc-test-normal.toml"),
        _command("inspect-appworld", profile_dir / "appworld" / "appworld.graphptc-test-challenge.toml"),
        _command("inspect-toolsandbox", profile_dir / "toolsandbox" / "graphptc.toml"),
        _command("inspect-agent-diff", profile_dir / "agent_diff" / "graphptc.toml"),
        _command("inspect-fanoutqa", profile_dir / "fanoutqa" / "graphptc-dev.toml"),
        _command("probe-fanoutqa-wikipedia", profile_dir / "fanoutqa" / "graphptc-dev.toml"),
        _command("inspect-frames", profile_dir / "frames" / "graphptc-test.toml"),
        _command("probe-frames-wikipedia", profile_dir / "frames" / "graphptc-test.toml"),
    ]
    return _run_commands(commands, dry_run=dry_run)


def _execute_arms(profile_dir: Path, *, phase: str, dry_run: bool) -> int:
    commands = []
    for arm in ARMS:
        command = arm.run_command if phase == "run" else arm.evaluate_command
        commands.append(_command(command, _arm_config(profile_dir, arm)))
    return _run_commands(commands, dry_run=dry_run)


def _run_commands(commands: Sequence[list[str]], *, dry_run: bool) -> int:
    for index, command in enumerate(commands, start=1):
        print(f"[{index}/{len(commands)}] " + subprocess.list2cmdline(command), flush=True)
        if dry_run:
            continue
        completed = subprocess.run(command, cwd=REPO_ROOT, check=False)
        if completed.returncode:
            return completed.returncode
    return 0


def _command(name: str, config: Path) -> list[str]:
    return [sys.executable, "-m", "graphptc", name, "--config", str(config)]


def _validate_profile(profile_dir: Path, profile: str) -> None:
    expected_outputs: set[str] = set()
    for arm in ARMS:
        path = _arm_config(profile_dir, arm)
        if not path.is_file():
            raise FileNotFoundError(f"profile config is missing: {path}")
        raw = tomllib.loads(path.read_text(encoding="utf-8"))
        for section in raw.values():
            if not isinstance(section, dict):
                continue
            for key, value in section.items():
                if key not in OUTPUT_KEYS:
                    continue
                if not isinstance(value, str) or f"/{profile}/" not in value.replace("\\", "/"):
                    raise ValueError(f"profile output path is not isolated: {path}: {key}")
                if value in expected_outputs:
                    raise ValueError(f"duplicate profile output path: {value}")
                expected_outputs.add(value)


def _missing_environment(profile_dir: Path) -> list[str]:
    names: set[str] = {"RAPID_API_KEY"}
    for arm in ARMS:
        raw = tomllib.loads(_arm_config(profile_dir, arm).read_text(encoding="utf-8"))
        for section_name in ("model", "grader", "user_model"):
            section = raw.get(section_name, {})
            if isinstance(section, dict) and section.get("api_key_env"):
                names.add(str(section["api_key_env"]))
        if arm.benchmark == "agent_diff":
            section = raw.get("agent_diff", {})
            for key in ("api_key_env", "base_url_env"):
                if section.get(key):
                    names.add(str(section[key]))
    return sorted(name for name in names if not os.getenv(name))


def _load_profile(profile_dir: Path) -> dict[str, object]:
    path = profile_dir.parent / "profile.json"
    if not path.is_file():
        raise FileNotFoundError(f"profile manifest is missing: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _arm_config(profile_dir: Path, arm: Arm) -> Path:
    return profile_dir / arm.benchmark / arm.config


def _profile_dir(profile: str) -> Path:
    if not PROFILE_NAME.fullmatch(profile):
        raise ValueError("profile must contain only letters, digits, dot, underscore, and hyphen")
    return REPO_ROOT / "runs" / "profiles" / profile / "configs"


def _toml_string(value: str) -> str:
    return json.dumps(value, ensure_ascii=False)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    create = subparsers.add_parser("create-profile", help="Create 21 isolated configs for a new model.")
    create.add_argument("--profile", required=True)
    create.add_argument("--model", required=True)
    create.add_argument("--base-url", required=True)
    create.add_argument("--api-key-env", default="TEACHER_MODEL_API_KEY")
    create.add_argument("--thinking", help="Optional provider-specific thinking mode; omitted by default.")
    for name in ("preflight", "run", "evaluate", "all"):
        command = subparsers.add_parser(name)
        command.add_argument("--profile", required=True)
        command.add_argument("--dry-run", action="store_true")
    return parser


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FileExistsError, FileNotFoundError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc
