#!/usr/bin/env bash
set -u

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
run_dir="$repo_dir/runs/browsecomp_plus/graphptc-stage1"
mkdir -p "$run_dir"

printf '[%s] starting GraphPTC Stage 1 smoke example 769\n' \
  "$(date --iso-8601=seconds)" >> "$run_dir/smoke.stderr.log"
cd "$repo_dir"
"$repo_dir/.venv/Scripts/graphptc.exe" run-graphptc-browsecomp-plus \
  --example-id 769 --restart \
  >> "$run_dir/smoke.stdout.log" \
  2>> "$run_dir/smoke.stderr.log"
exit_code=$?
printf '[%s] runner exited with code %s\n' "$(date --iso-8601=seconds)" "$exit_code" \
  >> "$run_dir/smoke.stderr.log"
exit "$exit_code"
