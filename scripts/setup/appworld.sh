#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
target="$repo_root/external/appworld"
python_bin="${GRAPHPTC_PYTHON311:-python3.11}"

mkdir -p "$target"
if [[ ! -x "$target/.venv/bin/python" ]]; then
  "$python_bin" -m venv "$target/.venv"
fi
"$target/.venv/bin/python" -m pip install --upgrade pip
"$target/.venv/bin/python" -m pip install "appworld==0.1.3.post1"
"$target/.venv/bin/python" -m appworld.cli install
"$target/.venv/bin/python" -m appworld.cli download data --root "$target"
"$target/.venv/bin/python" -m appworld.cli verify tests --root "$target"
"$target/.venv/bin/python" -m appworld.cli verify tasks --root "$target"
