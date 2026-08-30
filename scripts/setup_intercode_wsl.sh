#!/usr/bin/env bash
set -euo pipefail

official_root=/home/agent/graphptc-intercode
venv_root=/home/agent/graphptc-intercode/.venv
adapter_root=/mnt/d/GraphPTC

python3 -m venv "$venv_root"
"$venv_root/bin/python" -m pip install --upgrade pip
"$venv_root/bin/python" -m pip install \
  docker gymnasium mysql-connector-python numpy pandas rich rpyc scikit-learn scipy

for filesystem in 1 2 3 4; do
  docker build \
    --build-arg "file_system_version=$filesystem" \
    --tag "intercode-nl2bash-fs$filesystem" \
    --file "$adapter_root/infra/intercode/nl2bash.Dockerfile" \
    "$official_root"
done

docker build \
  --tag docker-env-sql \
  --file "$adapter_root/infra/intercode/sql-spider.Dockerfile" \
  "$official_root"

if docker container inspect docker-env-sql_ic_ctr >/dev/null 2>&1; then
  docker start docker-env-sql_ic_ctr >/dev/null
else
  docker run \
    --detach \
    --name docker-env-sql_ic_ctr \
    --publish 3307:3306 \
    --env MYSQL_ROOT_PASSWORD=password \
    docker-env-sql \
    --lower_case_table_names=1 >/dev/null
fi

for _ in $(seq 1 120); do
  if docker exec docker-env-sql_ic_ctr mysqladmin ping --silent; then
    exit 0
  fi
  sleep 1
done

exit 1
