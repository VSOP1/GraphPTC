from __future__ import annotations

import importlib.util
import sys
import tomllib
from pathlib import Path, PurePosixPath
from types import ModuleType


ROOT = Path(__file__).resolve().parents[2]


def _load_script(name: str, relative: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def test_full_suite_rewrites_only_evaluated_model_and_outputs() -> None:
    suite = _load_script("graphptc_full_suite_test", "scripts/evaluation/full_suite.py")
    template = (ROOT / "configs/toolsandbox/graphptc.toml").read_text(encoding="utf-8")

    rewritten = suite._rewrite_config(
        template,
        model="teacher-model",
        base_url="https://provider.example/v1",
        api_key_env="TEACHER_MODEL_API_KEY",
        thinking=None,
        output_dir=PurePosixPath("runs/toolsandbox/profile/graphptc"),
        appworld_experiment="unused",
    )
    raw = tomllib.loads(rewritten)

    assert len(suite.ARMS) == 21
    assert raw["model"]["model"] == "teacher-model"
    assert raw["model"]["api_key_env"] == "TEACHER_MODEL_API_KEY"
    assert "thinking" not in raw["model"]
    assert raw["user_model"]["model"] == "mimo-v2.5"
    assert raw["user_model"]["api_key_env"] == "MIMO_API_KEY"
    assert raw["toolsandbox"]["results_path"].startswith("runs/toolsandbox/profile/")


def test_release_package_rejects_secrets_and_runtime_state() -> None:
    release = _load_script("graphptc_release_test", "scripts/release/build_package.py")

    assert release._is_safe_source_path("src/graphptc/config.py")
    assert release._is_safe_source_path(".env.example")
    assert not release._is_safe_source_path(".env")
    assert not release._is_safe_source_path(".venv/bin/python")
    assert not release._is_safe_source_path("external/toolsandbox/.venv/bin/python")
    assert not release._is_safe_source_path("runs/appworld/results.jsonl")
