#!/usr/bin/env bash
set -euo pipefail

export JAVA_HOME=/usr/lib/jvm/java-21-openjdk-amd64
export PATH="$JAVA_HOME/bin:$PATH"
corpus_root=/mnt/d/GraphPTC/data/frames/wikipedia-20230601
mkdir -p "$corpus_root/tfrecord"
if [[ -d /var/lib/frames/wikipedia-20230601/tfrecord ]]; then
  find /var/lib/frames/wikipedia-20230601/tfrecord \
    -maxdepth 1 -type f -name 'wikipedia-train.tfrecord-*' ! -name '*.part' \
    -exec cp -n '{}' "$corpus_root/tfrecord/" \;
fi
exec /opt/frames-retriever/bin/python \
  /mnt/d/GraphPTC/scripts/data/prepare_frames_wikipedia.py \
  --root "$corpus_root"
