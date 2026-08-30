#!/usr/bin/env bash
set -euo pipefail

repo_dir="/mnt/d/GraphPTC"
export JAVA_HOME=/usr/lib/jvm/java-21-openjdk-amd64
export PATH="$JAVA_HOME/bin:$PATH"
exec /opt/frames-retriever/bin/python \
  "$repo_dir/scripts/services/frames_retriever.py" \
  --index-path /mnt/d/GraphPTC/data/frames/wikipedia-20230601/index \
  --host 0.0.0.0 \
  --port 8890 \
  --top-k 10 \
  --snippet-max-chars 1200 \
  --corpus-snapshot wikipedia/20230601.en \
  --document-count 6672479
