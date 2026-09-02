#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
target="$repo_root/external/agent_diff"
python_bin="${GRAPHPTC_PYTHON311:-python3.11}"

mkdir -p "$target"
if [[ ! -x "$target/.venv/bin/python" ]]; then
  "$python_bin" -m venv "$target/.venv"
fi
"$target/.venv/bin/python" -m pip install --upgrade pip
"$target/.venv/bin/python" -m pip install "agent-diff==1.0.6"
"$target/.venv/bin/python" -m pip check
