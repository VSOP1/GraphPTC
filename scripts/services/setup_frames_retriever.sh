#!/usr/bin/env bash
set -euo pipefail

apt-get update -qq
DEBIAN_FRONTEND=noninteractive apt-get install -y -qq \
  openjdk-21-jdk-headless \
  python3-venv
python3 -m venv /opt/frames-retriever
/opt/frames-retriever/bin/pip install --no-deps \
  pyserini==1.2.0 \
  tfrecord==1.14.6
/opt/frames-retriever/bin/pip install \
  pyjnius==1.7.0 \
  numpy==1.26.4 \
  pandas==2.3.3 \
  requests==2.32.5 \
  tqdm==4.67.1 \
  crc32c==2.7.1 \
  protobuf==5.29.5
