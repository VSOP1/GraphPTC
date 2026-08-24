#!/usr/bin/env bash
set -u

session_name="${1:-graphptc-bcp-full}"
repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
run_dir="$repo_dir/runs/browsecomp_plus/original-ptc-v1-turn30-full"
responses="$run_dir/responses.jsonl"
progress_log="$run_dir/full-run.monitor.log"

while true; do
  pane_dead="$(tmux display-message -p -t "$session_name:runner" '#{pane_dead}' 2>/dev/null || printf '1')"
  if [[ -f "$responses" ]]; then
    completed="$(wc -l < "$responses")"
  else
    completed=0
  fi
  last_status="$(tail -n 1 "$run_dir/full-run.stderr.log" 2>/dev/null || true)"
  printf '[%s] completed=%s/830 runner_dead=%s last=%s\n' \
    "$(date --iso-8601=seconds)" "$completed" "$pane_dead" "$last_status" \
    >> "$progress_log"
  if [[ "$pane_dead" == "1" ]]; then
    if [[ "$completed" == "830" ]]; then
      printf '[%s] generation complete; starting MiMo grader\n' \
        "$(date --iso-8601=seconds)" >> "$progress_log"
      cd "$repo_dir"
      "$repo_dir/.venv/Scripts/graphptc.exe" evaluate-browsecomp-plus \
        --config configs/browsecomp_plus/browsecomp_plus.example.toml \
        >> "$run_dir/evaluate.stdout.log" \
        2>> "$run_dir/evaluate.stderr.log"
      grader_exit=$?
      printf '[%s] grader exited with code %s\n' \
        "$(date --iso-8601=seconds)" "$grader_exit" >> "$progress_log"
    else
      printf '[%s] generation stopped incomplete; grader not started\n' \
        "$(date --iso-8601=seconds)" >> "$progress_log"
    fi
    break
  fi
  sleep 60
done

exec sleep infinity
