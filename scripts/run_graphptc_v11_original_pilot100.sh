#!/usr/bin/env bash
set -euo pipefail
cd /mnt/d/GraphPTC
export MIMO_API_KEY="$(grep '^MIMO_API_KEY=' .env | cut -d= -f2-)"
export TAVILY_API_KEY="$(grep '^TAVILY_API_KEY=' .env | cut -d= -f2-)"
exec /mnt/d/GraphPTC/.venv/Scripts/graphptc.exe run-browsecomp-plus \
  --config configs/browsecomp_plus.graphptc-adapt-v11-original-pilot100.toml \
  > runs/browsecomp_plus/graphptc-adapt-v11-original-pilot100/run.log \
  2> runs/browsecomp_plus/graphptc-adapt-v11-original-pilot100/run.err.log
