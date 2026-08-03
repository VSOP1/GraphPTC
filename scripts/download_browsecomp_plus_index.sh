#!/usr/bin/env bash
set -euo pipefail

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
hf_cli="/home/agent/.venvs/graphptc-retriever/bin/hf"
index_revision="b3f37f70c33829eb09d04784a54277a31871fd63"

"$hf_cli" download \
  Tevatron/browsecomp-plus-indexes \
  --repo-type dataset \
  --revision "$index_revision" \
  --include 'bm25/*' \
  --local-dir "$repo_dir/data/browsecomp_plus/official_indexes"
