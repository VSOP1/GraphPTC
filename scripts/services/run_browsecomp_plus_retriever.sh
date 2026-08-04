#!/usr/bin/env bash
set -euo pipefail

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export JAVA_HOME=/usr/lib/jvm/java-21-openjdk-amd64
export PATH="$JAVA_HOME/bin:$PATH"
export OPENAI_API_KEY=unused-local-retriever
exec /home/agent/.venvs/graphptc-retriever/bin/python \
  "$repo_dir/scripts/services/browsecomp_plus_retriever.py" \
  --index-path "$repo_dir/data/browsecomp_plus/official_indexes/bm25" \
  --tokenizer-path "$repo_dir/data/browsecomp_plus/qwen3-tokenizer" \
  --index-revision b3f37f70c33829eb09d04784a54277a31871fd63 \
  --tokenizer-revision c1899de289a04d12100db370d81485cdf75e47ca \
  --top-k 5 \
  --snippet-max-tokens 512
