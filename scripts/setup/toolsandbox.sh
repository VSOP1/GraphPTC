#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
target="$repo_root/external/toolsandbox"
python_bin="${GRAPHPTC_PYTHON310:-python3.10}"
commit="165848b9a78cead7ca7fe7c89c688b58e6501219"

if [[ ! -d "$target/.git" ]]; then
  if [[ -e "$target" ]]; then
    printf 'error: %s exists but is not the ToolSandbox Git checkout\n' "$target" >&2
    exit 2
  fi
  git clone https://github.com/apple/ToolSandbox.git "$target"
fi
current="$(git -C "$target" rev-parse HEAD)"
if [[ "$current" != "$commit" ]]; then
  git -C "$target" fetch origin "$commit"
  git -C "$target" checkout --detach "$commit"
fi
if [[ ! -x "$target/.venv/bin/python" ]]; then
  "$python_bin" -m venv "$target/.venv"
fi
"$target/.venv/bin/python" -m pip install --upgrade pip
"$target/.venv/bin/python" -m pip install -e "${target}[dev]"
"$target/.venv/bin/python" -m pip install \
  "httpx==0.27.2" \
  "codecell==0.2.1" \
  "toolregistry[ptc]==0.14.0" \
  "tomli==2.2.1" \
  "pytest==8.3.5"
"$target/.venv/bin/python" -m pip check
