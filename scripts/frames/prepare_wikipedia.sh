#!/usr/bin/env bash
set -euo pipefail

export JAVA_HOME=/usr/lib/jvm/java-21-openjdk-amd64
export PATH="$JAVA_HOME/bin:$PATH"
repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
corpus_root="$repo_dir/data/frames/wikipedia-20230601"
mkdir -p "$corpus_root/tfrecord"
if [[ -d /var/lib/frames/wikipedia-20230601/tfrecord ]]; then
  find /var/lib/frames/wikipedia-20230601/tfrecord \
    -maxdepth 1 -type f -name 'wikipedia-train.tfrecord-*' ! -name '*.part' \
    -exec cp -n '{}' "$corpus_root/tfrecord/" \;
fi
exec /opt/frames-retriever/bin/python \
  "$repo_dir/scripts/frames/prepare_wikipedia.py" \
  --root "$corpus_root"
