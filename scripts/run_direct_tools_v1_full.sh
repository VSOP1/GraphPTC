#!/usr/bin/env bash
set -u

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
run_dir="$repo_dir/runs/browsecomp_plus/direct-tools-v1-turn30-full"
config="configs/browsecomp_plus.direct-tools-v1-turn30-full.toml"
mkdir -p "$run_dir"

if ! curl --fail --silent --show-error --max-time 10 \
  http://127.0.0.1:8765/health >/dev/null; then
  printf '[%s] retriever health check failed\n' "$(date --iso-8601=seconds)" \
    >> "$run_dir/run.status.log"
  exit 2
fi

cd "$repo_dir"
printf '[%s] generation started\n' "$(date --iso-8601=seconds)" \
  >> "$run_dir/run.status.log"
"$repo_dir/.venv/Scripts/graphptc.exe" run-browsecomp-plus \
  --config "$config" \
  >> "$run_dir/generation.stdout.log" \
  2>> "$run_dir/generation.stderr.log"
generation_exit=$?
printf '[%s] generation exited with code %s\n' \
  "$(date --iso-8601=seconds)" "$generation_exit" >> "$run_dir/run.status.log"
if [[ "$generation_exit" -ne 0 ]]; then
  exit "$generation_exit"
fi

completed="$(wc -l < "$run_dir/responses.jsonl")"
if [[ "$completed" -ne 830 ]]; then
  printf '[%s] refusing to grade incomplete responses: %s/830\n' \
    "$(date --iso-8601=seconds)" "$completed" >> "$run_dir/run.status.log"
  exit 3
fi

printf '[%s] grading started\n' "$(date --iso-8601=seconds)" \
  >> "$run_dir/run.status.log"
"$repo_dir/.venv/Scripts/graphptc.exe" evaluate-browsecomp-plus \
  --config "$config" \
  >> "$run_dir/grading.stdout.log" \
  2>> "$run_dir/grading.stderr.log"
grader_exit=$?
printf '[%s] grading exited with code %s\n' \
  "$(date --iso-8601=seconds)" "$grader_exit" >> "$run_dir/run.status.log"
exit "$grader_exit"
