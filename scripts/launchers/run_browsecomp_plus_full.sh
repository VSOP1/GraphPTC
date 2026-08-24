#!/usr/bin/env bash
set -u

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
run_dir="$repo_dir/runs/browsecomp_plus/original-ptc-v1-turn30-full"
mkdir -p "$run_dir"

if ! curl --fail --silent --show-error --max-time 10 \
  http://127.0.0.1:8765/health >/dev/null; then
  printf '[%s] retriever health check failed\n' "$(date --iso-8601=seconds)" \
    >> "$run_dir/full-run.stderr.log"
  exit 2
fi

printf '[%s] starting full BrowseComp-Plus baseline\n' "$(date --iso-8601=seconds)" \
  >> "$run_dir/full-run.stderr.log"
cd "$repo_dir"
"$repo_dir/.venv/Scripts/graphptc.exe" run-browsecomp-plus \
  --config configs/browsecomp_plus/browsecomp_plus.example.toml \
  >> "$run_dir/full-run.stdout.log" \
  2>> "$run_dir/full-run.stderr.log"
exit_code=$?
printf '[%s] runner exited with code %s\n' "$(date --iso-8601=seconds)" "$exit_code" \
  >> "$run_dir/full-run.stderr.log"
exit "$exit_code"
