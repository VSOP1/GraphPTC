#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
python_bin="${GRAPHPTC_PYTHON311:-python3.11}"

cd "$repo_root"
if [[ ! -x ".venv/bin/python" ]]; then
  "$python_bin" -m venv .venv
fi
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -e ".[dev,browsecomp-plus,agent-diff,fanoutqa]"
bash scripts/setup/appworld.sh
bash scripts/setup/toolsandbox.sh
bash scripts/setup/agent_diff.sh
