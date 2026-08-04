#!/usr/bin/env bash
set -u

session_name="${1:-graphptc-direct-full-v1}"
repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
run_dir="$repo_dir/runs/browsecomp_plus/direct-tools-v1-turn30-full"
responses="$run_dir/responses.jsonl"
grades="$run_dir/grades.jsonl"
monitor_log="$run_dir/monitor.log"
mkdir -p "$run_dir"

while true; do
  runner_dead="$(tmux display-message -p -t "$session_name:runner" '#{pane_dead}' 2>/dev/null || printf '1')"
  responses_count=0
  grades_count=0
  [[ -f "$responses" ]] && responses_count="$(wc -l < "$responses")"
  [[ -f "$grades" ]] && grades_count="$(wc -l < "$grades")"
  last_status="$(tail -n 1 "$run_dir/run.status.log" 2>/dev/null || true)"
  printf '[%s] responses=%s/830 grades=%s/830 runner_dead=%s last=%s\n' \
    "$(date --iso-8601=seconds)" "$responses_count" "$grades_count" \
    "$runner_dead" "$last_status" >> "$monitor_log"
  if [[ "$runner_dead" == "1" ]]; then
    break
  fi
  sleep 60
done

exec sleep infinity
